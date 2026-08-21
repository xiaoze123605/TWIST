"""Inference-only branch ablation evaluation for the Dual AnyAdapter.

Loads ONE Dual AnyAdapter checkpoint (a = a_base + adapter_gain * delta) and
rolls it out three ways by masking which residual branches are added at
inference time:

    full      -> a_base + adapter_gain * (dyn_gain * delta_dyn + err_gain * delta_err)
    dyn_only  -> a_base + adapter_gain * (dyn_gain * delta_dyn)
    err_only  -> a_base + adapter_gain * (err_gain * delta_err)

The mask is applied only through TwistAnyAdapterActorCritic.adapter_branch_mode;
network weights, the world model, and the optimizer are never touched, so a
single checkpoint serves all three modes.  Fairness: identical task config,
checkpoint, seed, motion file, and domain randomization across modes; only the
branch mask changes (run each mode in a separate process with the same flags).

Results are written to <out_dir>/<branch_mode>.json (full report) and
<out_dir>/<branch_mode>.csv (flat summary row).
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# NOTE: isaacgym must be imported before torch; legged_gym.envs does that.
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks, imports isaacgym)
from legged_gym.gym_utils import task_registry
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill

import torch
from isaacgym import gymapi

BRANCH_MODES = ("full", "dyn_only", "err_only")

# Termination reason labels (priority order matches the env's reset semantics).
REASON_NONE = "none"
REASON_TIMEOUT = "timeout"
REASON_MOTION_END = "motion_end"
REASON_POSE_FAIL = "pose_fail"
REASON_CONTACT = "contact"
REASON_ROLL_PITCH = "roll_pitch"
REASON_HEIGHT = "height"
REASON_VELOCITY = "velocity"
REASON_OTHER = "other"


def instrument_termination_reasons() -> None:
    """Monkey-patch check_termination to record a per-env reason buffer.

    Pure read-only instrumentation: the original function runs first and its
    behavior is unchanged; the wrapper only fills env._term_reason_buf with
    the first condition that triggered reset_buf for each env.
    """
    original = G1MimicDistill.check_termination

    def instrumented_check_termination(self):
        original(self)

        num_envs = self.num_envs
        reason = [REASON_NONE] * num_envs

        time_out = self.time_out_buf
        motion_end = (
            self.episode_length_buf * self.dt
            >= self._motion_lib.get_motion_length(self._motion_ids)
        )
        contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        height = (
            torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2])
            > self.cfg.rewards.root_height_diff_threshold
        )
        velocity = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.0

        pose_fail = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._pose_termination:
            body_pos = (
                self.rigid_body_states[:, self._key_body_ids, 0:3]
                - self.rigid_body_states[:, 0:1, 0:3]
            )
            tar_body_pos = (
                self._ref_body_pos[:, self._key_body_ids]
                - self._ref_root_pos[:, None, :]
            )
            if not self.global_obs:
                body_pos = convert_to_local_root_body_pos(
                    self.root_states[:, 3:7], body_pos
                )
                tar_body_pos = convert_to_local_root_body_pos(
                    self._ref_root_rot, tar_body_pos
                )
            body_pos_dist = (
                (tar_body_pos - body_pos).square().sum(dim=-1).max(dim=-1)[0]
            )
            pose_fail = body_pos_dist > self._pose_termination_dist ** 2
            if self._track_root:
                root_dist = (
                    self._ref_root_pos - self.root_states[:, 0:3]
                ).square().sum(dim=-1)
                pose_fail |= (
                    root_dist.squeeze(-1)
                    > self._root_tracking_termination_dist ** 2
                )

        reset = self.reset_buf.cpu().numpy()
        for i in range(num_envs):
            if not reset[i]:
                continue
            if time_out[i]:
                reason[i] = REASON_TIMEOUT
            elif motion_end[i]:
                reason[i] = REASON_MOTION_END
            elif pose_fail[i]:
                reason[i] = REASON_POSE_FAIL
            elif contact[i]:
                reason[i] = REASON_CONTACT
            elif roll_cut[i] or pitch_cut[i]:
                reason[i] = REASON_ROLL_PITCH
            elif height[i]:
                reason[i] = REASON_HEIGHT
            elif velocity[i]:
                reason[i] = REASON_VELOCITY
            else:
                reason[i] = REASON_OTHER

        self._term_reason_buf = reason

    G1MimicDistill.check_termination = instrumented_check_termination


def build_eval_args(args: argparse.Namespace, headless: bool) -> argparse.Namespace:
    """Namespace with every field task_registry/parse_sim_params expects."""
    return argparse.Namespace(
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device_type="cuda" if "cuda" in args.device else "cpu",
        sim_device_id=0,
        compute_device_id=0,
        graphics_device_id=0,
        sim_device="cuda" if "cuda" in args.device else "cpu",
        rl_device=args.device,
        device=args.device,
        headless=headless,
        use_gpu=True,
        use_gpu_pipeline=True,
        subscenes=0,
        num_threads=0,
        teleop_mode=False,
        rows=None,
        cols=None,
        record_video=False,
        no_rand=args.no_dr,
        resume=False,
        max_iterations=None,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        fix_action_std=False,
        teacher_exptid="mimic",
        teacher_checkpoint=-1,
        eval_student=False,
        proj_name="g1",
        exptid=None,
        resumeid=None,
    )


def checkpoint_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_branches(actor_critic, delta_dyn, delta_err, mode):
    """Mirror of TwistAnyAdapterActorCritic.action_delta for the dual adapter."""
    if mode == "dyn_only":
        return actor_critic.dynamics_branch_gain * delta_dyn
    if mode == "err_only":
        return actor_critic.tracking_branch_gain * delta_err
    return (
        actor_critic.dynamics_branch_gain * delta_dyn
        + actor_critic.tracking_branch_gain * delta_err
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="g1_stu_anyadapter_dual")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained Dual AnyAdapter checkpoint (*.pt).")
    parser.add_argument("--branch_mode", type=str, choices=BRANCH_MODES,
                        default="full")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Number of control steps per rollout.")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="Override env episode length (seconds). Default: task config.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--show_viewer", action="store_true",
                        help="Attach a viewer instead of running headless.")
    parser.add_argument("--no_dr", action="store_true",
                        help="Disable domain randomization (controlled runs).")
    parser.add_argument("--out_dir", type=str,
                        default=str(REPO_ROOT / "results" / "dual_ablation"))
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument_termination_reasons()
    eval_args = build_eval_args(args, headless=not args.show_viewer)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Identical env config across modes; only the branch mask changes.
    env_cfg.env.record_video = False
    env_cfg.env.rand_reset = False
    if args.episode_length_s is not None:
        env_cfg.env.episode_length_s = args.episode_length_s
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=args.task, args=eval_args, env_cfg=env_cfg)
    print(f"[evaluate_dual_branch] env num_envs={env.num_envs} "
          f"dt={env.dt} num_actions={env.num_actions}")

    train_cfg.runner.resume = False
    ppo_runner, _ = task_registry.make_alg_runner(
        log_root=None, env=env, name=args.task, args=eval_args,
        train_cfg=train_cfg, init_wandb=False,
    )
    ppo_runner.load(args.checkpoint, load_optimizer=False)

    actor_critic = ppo_runner.get_actor_critic(device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic.adapter_branch_mode = args.branch_mode
    print(f"[evaluate_dual_branch] branch mode: {actor_critic.adapter_branch_mode}")

    obs = env.get_observations()
    if env.cfg.env.normalize_obs:
        normalizer = ppo_runner.get_normalizer(device=env.device)
    else:
        normalizer = None

    # --- accumulators -----------------------------------------------------
    num_envs = env.num_envs
    episode_return = torch.zeros(num_envs, device=env.device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    completed_returns = []
    completed_lengths = []
    fall_count = 0
    timeout_count = 0
    motion_end_count = 0
    reason_counts = {label: 0 for label in (
        REASON_TIMEOUT, REASON_MOTION_END, REASON_POSE_FAIL, REASON_CONTACT,
        REASON_ROLL_PITCH, REASON_HEIGHT, REASON_VELOCITY, REASON_OTHER,
    )}
    mean_rew = 0.0
    mean_root_pos_err = 0.0
    mean_roll_err = 0.0
    mean_pitch_err = 0.0
    mean_joint_err = 0.0
    mean_keybody_err = 0.0
    mean_dyn_res = 0.0
    mean_err_res = 0.0
    mean_applied_res = 0.0
    max_applied_res = 0.0
    # Env's own episode logger (extras), weighted by number of resets per step.
    extras_metric_sums = {}
    extras_metric_weight = 0.0

    for step in range(args.max_steps):
        if normalizer is not None:
            normalized_obs = normalizer.normalize(obs.detach())
        else:
            normalized_obs = obs.detach()
        actions = policy(normalized_obs)
        with torch.inference_mode():
            delta_dyn, delta_err = actor_critic.get_adapter_delta_components(normalized_obs)
            applied = combine_branches(actor_critic, delta_dyn, delta_err,
                                       args.branch_mode)
            dyn_res = torch.norm(delta_dyn, dim=-1).mean()
            err_res = torch.norm(delta_err, dim=-1).mean()
            applied_res = torch.norm(applied, dim=-1).mean()

        obs, _, rews, dones, infos = env.step(actions.detach())

        episode_return += rews
        episode_steps += 1

        # Tracking errors from the env's current state vs reference motion.
        root_pos_err = torch.norm(env.root_states[:, :3] - env._ref_root_pos, dim=-1)
        roll_ref, pitch_ref, _ = euler_from_quaternion(env._ref_root_rot)
        roll_err = torch.abs(env.roll - roll_ref)
        pitch_err = torch.abs(env.pitch - pitch_ref)
        joint_err = torch.abs(env.dof_pos - env._ref_dof_pos).mean(dim=-1)
        keybody_err = torch.norm(
            env.rigid_body_states[:, env._key_body_ids, 0:3]
            - env._ref_body_pos[:, env._key_body_ids],
            dim=-1,
        ).mean(dim=-1)

        mean_rew += float(rews.mean().detach().cpu())
        mean_root_pos_err += float(root_pos_err.mean().detach().cpu())
        mean_roll_err += float(roll_err.mean().detach().cpu())
        mean_pitch_err += float(pitch_err.mean().detach().cpu())
        mean_joint_err += float(joint_err.mean().detach().cpu())
        mean_keybody_err += float(keybody_err.mean().detach().cpu())
        mean_dyn_res += float(dyn_res.detach().cpu())
        mean_err_res += float(err_res.detach().cpu())
        mean_applied_res += float(applied_res.detach().cpu())
        max_applied_res = max(
            max_applied_res, float(torch.norm(applied, dim=-1).max().detach().cpu())
        )

        done_mask = dones.bool()
        if done_mask.any():
            completed_returns.extend(episode_return[done_mask].detach().cpu().tolist())
            completed_lengths.extend(episode_steps[done_mask].detach().cpu().tolist())
            timeout_count += int((done_mask & env.time_out_buf).sum())
            fall_count += int((done_mask & ~env.time_out_buf).sum())
            done_ids = done_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
            for label in reason_counts:
                reason_counts[label] += sum(
                    1 for i in done_ids if env._term_reason_buf[i] == label
                )
            episode_return[done_mask] = 0.0
            episode_steps[done_mask] = 0

            # Env's own per-episode reward-component logger (extras["episode"]).
            # Only consumed on reset steps: reset_idx refills it exclusively for
            # the envs that just reset, and it stays stale between resets.
            if "episode" in infos and infos["episode"]:
                weight = float(done_mask.sum())
                for key, value in infos["episode"].items():
                    if value is not None:
                        extras_metric_sums[key] = (
                            extras_metric_sums.get(key, 0.0)
                            + float(value.mean().detach().cpu()) * weight
                        )
                extras_metric_weight += weight

    steps = args.max_steps
    num_episodes = len(completed_returns)
    motion_end_count = reason_counts[REASON_MOTION_END]
    results = {
        "meta": {
            "task": args.task,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_md5": checkpoint_md5(args.checkpoint),
            "branch_mode": args.branch_mode,
            "branch_mask": {
                "full": "delta_dyn + delta_err",
                "dyn_only": "delta_dyn",
                "err_only": "delta_err",
            }[args.branch_mode],
            "adapter_gain": float(actor_critic.adapter_gain),
            "dynamics_branch_gain": float(actor_critic.dynamics_branch_gain),
            "tracking_branch_gain": float(actor_critic.tracking_branch_gain),
            "seed": args.seed,
            "num_envs": num_envs,
            "max_steps": steps,
            "dt": float(env.dt),
            "episode_length_s": float(env.cfg.env.episode_length_s),
            "domain_rand_enabled": bool(env.cfg.domain_rand.domain_rand_general),
            "motion_file": str(env.cfg.motion.motion_file),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "performance": {
            "mean_reward_per_step": mean_rew / steps,
            "num_episodes": num_episodes,
            "mean_episode_return": (sum(completed_returns) / num_episodes) if num_episodes else None,
            "std_episode_return": float(torch.tensor(completed_returns).float().std()) if num_episodes > 1 else None,
            "mean_episode_length_s": (sum(completed_lengths) / num_episodes * env.dt) if num_episodes else None,
            "episode_return": completed_returns,
            "episode_length_frames": completed_lengths,
        },
        "tracking": {
            "mean_root_pos_error_m": mean_root_pos_err / steps,
            "mean_roll_error_rad": mean_roll_err / steps,
            "mean_pitch_error_rad": mean_pitch_err / steps,
            "mean_joint_pos_error_rad": mean_joint_err / steps,
            "mean_keybody_pos_error_m": mean_keybody_err / steps,
            # Reward-component metrics from the env's own episode logger.
            "env_episode_reward_components": {
                key: value / extras_metric_weight if extras_metric_weight > 0 else None
                for key, value in extras_metric_sums.items()
            },
        },
        "stability": {
            "termination_reason_counts": reason_counts,
            "fall_count": fall_count,
            "timeout_count": timeout_count,
            "motion_end_count": motion_end_count,
            "fall_rate_per_episode": (fall_count / num_episodes) if num_episodes else None,
        },
        "residual": {
            "mean_delta_dyn_l2": mean_dyn_res / steps,
            "mean_delta_err_l2": mean_err_res / steps,
            "mean_applied_residual_l2": mean_applied_res / steps,
            "max_applied_residual_l2": max_applied_res,
        },
    }

    json_path = out_dir / f"{args.branch_mode}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    flat = {
        "branch_mode": args.branch_mode,
        "checkpoint": results["meta"]["checkpoint"],
        "seed": args.seed,
        "num_envs": num_envs,
        "max_steps": steps,
        "mean_reward_per_step": results["performance"]["mean_reward_per_step"],
        "mean_episode_return": results["performance"]["mean_episode_return"],
        "mean_episode_length_s": results["performance"]["mean_episode_length_s"],
        "mean_root_pos_error_m": results["tracking"]["mean_root_pos_error_m"],
        "mean_roll_error_rad": results["tracking"]["mean_roll_error_rad"],
        "mean_pitch_error_rad": results["tracking"]["mean_pitch_error_rad"],
        "mean_joint_pos_error_rad": results["tracking"]["mean_joint_pos_error_rad"],
        "mean_keybody_pos_error_m": results["tracking"]["mean_keybody_pos_error_m"],
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "motion_end_count": motion_end_count,
        "mean_delta_dyn_l2": results["residual"]["mean_delta_dyn_l2"],
        "mean_delta_err_l2": results["residual"]["mean_delta_err_l2"],
        "mean_applied_residual_l2": results["residual"]["mean_applied_residual_l2"],
        "max_applied_residual_l2": results["residual"]["max_applied_residual_l2"],
    }
    for label in reason_counts:
        flat[f"termination_{label}"] = reason_counts[label]

    csv_path = out_dir / f"{args.branch_mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    print(f"[evaluate_dual_branch] mode={args.branch_mode} "
          f"episodes={num_episodes} falls={fall_count} timeouts={timeout_count} "
          f"mean_episode_return={results['performance']['mean_episode_return']}")
    print(f"[evaluate_dual_branch] results -> {json_path}")
    print(f"[evaluate_dual_branch] results -> {csv_path}")


if __name__ == "__main__":
    main()
