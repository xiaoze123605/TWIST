"""Collect one cumulative DAgger round with episode-level teacher mixing."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "legged_gym"), str(ROOT / "rsl_rl"), str(ROOT / "pose")]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--report", required=True)
    custom, remaining = parser.parse_known_args()
    if custom.round < 0 or custom.steps < 1 or not 0 <= custom.beta <= 1:
        raise ValueError("invalid round, steps or beta")
    if custom.beta < 1 and not custom.student_checkpoint:
        raise ValueError("student checkpoint required when beta < 1")
    sys.argv = [sys.argv[0]] + remaining

    import isaacgym  # noqa: F401 - import order required by Isaac Gym
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    from legged_gym.envs.base.legged_robot import euler_from_quaternion
    from rsl_rl.datasets.dynamics_dagger_buffer import (
        DynamicsDaggerBuffer, SOURCE_CURRENT_STUDENT, SOURCE_TEACHER, build_actor_input,
    )

    dataset_path = Path(custom.dataset)
    replay = DynamicsDaggerBuffer.load(dataset_path) if dataset_path.exists() else DynamicsDaggerBuffer()

    args = get_args()
    if args.task not in ("g1_dynamics_tracker_wide", "g1_dynamics_tracker_adaptive_wide"):
        raise ValueError("DAgger collection requires a wide DynamicsTracker task")
    args.headless = True
    cfg, training_cfg = task_registry.get_cfgs(args.task)
    cfg.noise.add_noise = False
    cfg.env.randomize_start_pos = False
    cfg.domain_rand.domain_rand_general = False
    for name in ("randomize_gravity", "randomize_friction", "randomize_base_mass",
                 "randomize_base_com", "push_robots", "push_end_effector",
                 "randomize_motor", "action_delay", "randomize_mimic_obs"):
        setattr(cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.dagger_randomize_phase = True
    teacher = torch.jit.load(custom.teacher, map_location=env.device).eval()
    runner = None
    if custom.student_checkpoint:
        runner, _ = task_registry.make_alg_runner(
            env, name=args.task, args=args, train_cfg=training_cfg,
            init_wandb=False, log_root=None)
        runner.load(custom.student_checkpoint, load_optimizer=False,
                    warm_start=args.task == "g1_dynamics_tracker_adaptive_wide")
        runner.alg.actor_critic.eval()

    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()

    teacher_history = None
    previous_executed_legacy_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    episode_base = (int(replay.tensors["episode_id"].max()) + 1) if len(replay) else 0
    episode_ids = torch.arange(episode_base, episode_base+env.num_envs,
                               dtype=torch.long, device=env.device)
    next_episode_id = episode_base + env.num_envs
    teacher_control = torch.rand(env.num_envs, device=env.device) < custom.beta
    tensors = {name: [] for name in (
        "actor_input", "teacher_target", "motion_id", "motion_time", "motion_phase",
        "episode_id", "source", "failure_margin_m", "seed",
        "action_mode", "target_scale_version", "collection_round", "env_id",
        "physical_failure",
    )}
    scale_text = json.dumps(list(env.cfg.control.target_scales), separators=(",", ":"))
    scale_version = int(hashlib.sha256(scale_text.encode()).hexdigest()[:15], 16)

    def legacy_current_observation():
        times = env._get_motion_times() + env.dt
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = env._motion_lib.calc_motion_frame(
            env._motion_ids, times)
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        mimic = torch.cat((root_pos[:, 2:3], roll[:, None], pitch[:, None], yaw[:, None],
                           quat_rotate_inverse(root_rot, root_vel),
                           quat_rotate_inverse(root_rot, root_ang_vel)[:, 2:3], dof_pos), -1)
        proprio = torch.cat((env.base_ang_vel * env.obs_scales.ang_vel,
                             torch.stack((env.roll, env.pitch), -1),
                             (env.dof_pos-env.default_dof_pos_all)*env.obs_scales.dof_pos,
                             env.dof_vel*env.obs_scales.dof_vel,
                             previous_executed_legacy_action), -1)
        velocity_start = 5 + env.num_actions
        proprio[:, [velocity_start+i for i in (4, 5, 10, 11)]] = 0.
        return torch.cat((mimic, proprio), -1)

    with torch.no_grad():
        for _ in range(custom.steps):
            observation = env.get_observations()
            actor_input = build_actor_input(observation, runner.alg.actor_critic if runner else
                                             _zero_latent_actor(env, training_cfg, task_registry, args))
            # The temporary actor is cached by _zero_latent_actor; it is used
            # only to apply the existing observation-to-actor-input contract.
            legacy_current = legacy_current_observation()
            if teacher_history is None:
                teacher_history = legacy_current[:, None].repeat(1, 10, 1)
            teacher_obs = torch.cat((legacy_current, teacher_history.flatten(1)), -1)
            legacy_action = teacher(teacher_obs).clamp(
                -env.cfg.normalization.clip_actions/env.cfg.control.action_scale,
                env.cfg.normalization.clip_actions/env.cfg.control.action_scale)
            legacy_target = env.default_dof_pos_all + legacy_action*env.cfg.control.action_scale
            normalized = ((legacy_target-env.current_reference[:, 8:31]) /
                          env.target_transform.scale).clamp(-.999, .999)
            teacher_raw = torch.atanh(normalized)
            teacher_target = env.target_transform(teacher_raw, env.current_reference[:, 8:31])
            if runner:
                student_raw = runner.alg.actor_critic.act_inference(observation)
                raw = torch.where(teacher_control[:, None], teacher_raw, student_raw)
            else:
                raw = teacher_raw
            executed_target = env.target_transform(raw, env.current_reference[:, 8:31])
            executed_legacy = (executed_target-env.default_dof_pos_all)/env.cfg.control.action_scale
            motion_time = env._get_motion_times().clone()
            motion_length = env._motion_lib.get_motion_length(env._motion_ids)
            tensors["actor_input"].append(actor_input.cpu())
            tensors["teacher_target"].append(teacher_target.cpu())
            tensors["motion_id"].append(env._motion_ids.cpu().clone())
            tensors["motion_time"].append(motion_time.cpu())
            tensors["motion_phase"].append((motion_time/motion_length.clamp_min(1e-6)).clamp(0, 1).cpu())
            tensors["episode_id"].append(episode_ids.cpu().clone())
            tensors["source"].append(torch.where(teacher_control,
                torch.full_like(episode_ids, SOURCE_TEACHER),
                torch.full_like(episode_ids, SOURCE_CURRENT_STUDENT)).cpu())
            tensors["seed"].append(torch.full_like(episode_ids, int(env.cfg.seed)).cpu())
            tensors["action_mode"].append(torch.zeros_like(episode_ids).cpu())
            tensors["target_scale_version"].append(torch.full_like(episode_ids, scale_version).cpu())
            tensors["collection_round"].append(torch.full_like(episode_ids, custom.round).cpu())
            tensors["env_id"].append(all_ids.cpu())
            _, _, _, done, info = env.step(raw)
            failure = info["physical_failure"].bool()
            tensors["physical_failure"].append(failure.cpu())
            margin = env.cfg.rewards.root_height_diff_threshold-info["root_height_abs_error"]
            tensors["failure_margin_m"].append(margin.cpu())
            teacher_history = torch.cat((teacher_history[:, 1:], legacy_current[:, None]), 1)
            previous_executed_legacy_action.copy_(executed_legacy)
            reset_ids = done.nonzero(as_tuple=False).flatten()
            if reset_ids.numel():
                new_current = legacy_current_observation()
                teacher_history[reset_ids] = new_current[reset_ids, None]
                previous_executed_legacy_action[reset_ids] = 0.
                count = reset_ids.numel()
                episode_ids[reset_ids] = torch.arange(next_episode_id, next_episode_id+count,
                                                       device=env.device)
                next_episode_id += count
                teacher_control[reset_ids] = torch.rand(count, device=env.device) < custom.beta

    stacked = {name: torch.stack(value).flatten(0, 1) for name, value in tensors.items()}
    # Compute steps-to-failure within each (environment, episode) trajectory.
    steps_to_failure = torch.full_like(stacked["episode_id"], -1)
    for env_id in range(env.num_envs):
        ids = (stacked["env_id"] == env_id).nonzero(as_tuple=False).flatten()
        episodes = stacked["episode_id"][ids]
        for episode in episodes.unique():
            episode_rows = ids[episodes == episode]
            failed = stacked["physical_failure"][episode_rows].nonzero(as_tuple=False).flatten()
            if failed.numel():
                failure_row = int(failed[0])
                steps_to_failure[episode_rows[:failure_row+1]] = torch.arange(
                    failure_row, -1, -1, dtype=torch.long)
    stacked["steps_to_failure"] = steps_to_failure
    del stacked["env_id"], stacked["physical_failure"]

    round_metadata = dict(round=custom.round, beta=custom.beta, samples=stacked["actor_input"].shape[0],
                          seed=int(env.cfg.seed), steps=custom.steps,
                          student_checkpoint=(str(Path(custom.student_checkpoint).resolve())
                                              if custom.student_checkpoint else None),
                          random_phase=True)
    replay.append_round(stacked, round_metadata)
    file_sha = replay.save(dataset_path)
    report = dict(dataset=str(dataset_path.resolve()), dataset_sha256=file_sha,
                  content_sha256=replay.content_sha256(), sample_count=len(replay),
                  source_counts=replay.source_counts(), latest_round=round_metadata,
                  phase_min=float(stacked["motion_phase"].min()),
                  phase_max=float(stacked["motion_phase"].max()),
                  failure_adjacent_samples=int((stacked["steps_to_failure"] >= 0).sum()))
    report_path = Path(custom.report)
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


_CACHED_ACTOR = None


def _zero_latent_actor(env, training_cfg, task_registry, args):
    global _CACHED_ACTOR
    if _CACHED_ACTOR is None:
        runner, _ = task_registry.make_alg_runner(env, name=args.task, args=args,
                                                  train_cfg=training_cfg,
                                                  init_wandb=False, log_root=None)
        runner.alg.actor_critic.use_dynamics_latent = False
        runner.alg.actor_critic.eval()
        _CACHED_ACTOR = runner.alg.actor_critic
    return _CACHED_ACTOR


if __name__ == "__main__":
    main()
