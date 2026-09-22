"""Run the legacy TWIST teacher through the new tracker's target interface.

This is a diagnostic, not a deployment dependency.  It answers whether the
new simulator/reset/PD/target contract can remain upright when supplied with
an already competent controller.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "legged_gym"), str(ROOT / "rsl_rl"), str(ROOT / "pose")]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--distill-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-control-fraction", type=float, default=0.0)
    parser.add_argument("--distill-lr", type=float, default=3e-4)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--distill-batch-size", type=int, default=1024)
    custom, remaining = parser.parse_known_args()
    output = Path(custom.output)
    if output.exists():
        raise FileExistsError(output)
    sys.argv = [sys.argv[0]] + remaining

    import isaacgym  # noqa: F401 - must precede torch
    import torch
    from torch.nn import functional as F
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    from legged_gym.envs.base.legged_robot import euler_from_quaternion
    from rsl_rl.modules.dynamics_tracker import CURRENT_DIM, STATE_DIM, TASK_DIM

    args = get_args()
    if not args.task.startswith("g1_dynamics_tracker"):
        raise ValueError("select a g1_dynamics_tracker task")
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
    teacher = torch.jit.load(custom.teacher, map_location=env.device).eval()
    runner = optimizer = None
    if custom.distill_checkpoint:
        if training_cfg.policy.use_dynamics_latent:
            raise ValueError("teacher pretraining requires a zero-latent Stage-A task")
        runner, _ = task_registry.make_alg_runner(
            env, name=args.task, args=args, train_cfg=training_cfg,
            init_wandb=False, log_root=None)
        optimizer = torch.optim.Adam(runner.alg.actor_critic.actor.parameters(), lr=custom.distill_lr)
        if custom.student_checkpoint:
            runner.load(custom.student_checkpoint, load_optimizer=False)
    elif custom.student_checkpoint:
        raise ValueError("--student-checkpoint requires --distill-checkpoint")
    if not 0.0 <= custom.teacher_control_fraction <= 1.0:
        raise ValueError("teacher control fraction must be in [0, 1]")

    env._eval_scenario_motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.num_motions()
    env._eval_scenario_motion_times = torch.zeros(env.num_envs, device=env.device)
    env._eval_scenario_dof_pos_scale = torch.ones_like(env.dof_pos)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()

    # Legacy student history repeats the first observation at startup.
    old_history = None
    previous_teacher_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(ended)
    durations = torch.full((env.num_envs,), custom.steps, dtype=torch.long, device=env.device)
    failures = completions = 0
    imitation_loss = []
    actor_inputs, actor_targets = [], []

    def teacher_current_observation():
        times = env._get_motion_times() + env.dt
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = env._motion_lib.calc_motion_frame(
            env._motion_ids, times)
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        root_vel = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        mimic = torch.cat((root_pos[:, 2:3], roll[:, None], pitch[:, None], yaw[:, None],
                           root_vel, root_ang_vel[:, 2:3], dof_pos), dim=-1)
        proprio = torch.cat((env.base_ang_vel * env.obs_scales.ang_vel,
                             torch.stack((env.roll, env.pitch), dim=-1),
                             (env.dof_pos - env.default_dof_pos_all) * env.obs_scales.dof_pos,
                             env.dof_vel * env.obs_scales.dof_vel,
                             previous_teacher_action), dim=-1)
        # Match the legacy student's masked ankle velocities.
        vel_start = 5 + env.num_actions
        for joint in (4, 5, 10, 11):
            proprio[:, vel_start + joint] = 0.
        return torch.cat((mimic, proprio), dim=-1)

    for step in range(custom.steps):
        with torch.no_grad():
            current = teacher_current_observation()
            if old_history is None:
                old_history = current[:, None].repeat(1, 10, 1)
            teacher_obs = torch.cat((current, old_history.flatten(1)), dim=-1)
            teacher_action = teacher(teacher_obs)
            clipped = teacher_action.clamp(-env.cfg.normalization.clip_actions /
                                             env.cfg.control.action_scale,
                                             env.cfg.normalization.clip_actions /
                                             env.cfg.control.action_scale)
            target = env.default_dof_pos_all + clipped * env.cfg.control.action_scale
            normalized = ((target - env.current_reference[:, 8:31]) /
                          env.target_transform.scale).clamp(-0.999, 0.999)
            teacher_raw = torch.atanh(normalized)
        if runner is not None:
            # Stage A uses a zero latent. Store the compact actor input instead
            # of the 6k observation so shuffled replay stays inexpensive.
            current_new = env.get_observations()[:, :CURRENT_DIM]
            state = current_new[:, :STATE_DIM]
            task = current_new[:, STATE_DIM:STATE_DIM + TASK_DIM]
            q_error = task[:, 8:31] - state[:, 6:29]
            dq_error = task[:, 31:] - state[:, 29:52]
            latent = torch.zeros(current_new.shape[0], runner.alg.actor_critic.latent_dim,
                                 device=env.device)
            actor_inputs.append(torch.cat((current_new, q_error, dq_error, latent), -1).cpu())
            actor_targets.append(normalized.cpu())
        with torch.no_grad():
            if custom.student_checkpoint:
                student_raw = runner.alg.actor_critic.act_inference(env.get_observations())
                teacher_count = int(env.num_envs * custom.teacher_control_fraction)
                mask = torch.arange(env.num_envs, device=env.device) < teacher_count
                raw = torch.where(mask[:, None], teacher_raw, student_raw)
            else:
                raw = teacher_raw
            executed_target = env.target_transform(raw, env.current_reference[:, 8:31])
            executed_legacy_action = ((executed_target - env.default_dof_pos_all) /
                                      env.cfg.control.action_scale)
            _, _, _, done, info = env.step(raw)
            completion = info["motion_completed"].bool() & done.bool()
            first = done.bool() & ~ended
            durations[first] = step + 1
            success[first] = completion[first]
            ended |= done.bool()
            failures += int(info["physical_failure"].sum())
            completions += int(info["motion_completed"].sum())
            reset_ids = done.nonzero(as_tuple=False).flatten()
            old_history = torch.cat((old_history[:, 1:], current[:, None]), dim=1)
            previous_teacher_action.copy_(executed_legacy_action)
            if reset_ids.numel():
                next_current = teacher_current_observation()
                old_history[reset_ids] = next_current[reset_ids, None]
                previous_teacher_action[reset_ids] = 0.

    if runner is not None:
        inputs = torch.cat(actor_inputs)
        targets = torch.cat(actor_targets)
        for _ in range(custom.distill_epochs):
            order = torch.randperm(inputs.shape[0])
            for start in range(0, inputs.shape[0], custom.distill_batch_size):
                ids = order[start:start + custom.distill_batch_size]
                prediction = runner.alg.actor_critic.actor(inputs[ids].to(env.device))
                loss = F.smooth_l1_loss(torch.tanh(prediction), targets[ids].to(env.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(runner.alg.actor_critic.actor.parameters(), 1.0)
                optimizer.step()
                imitation_loss.append(float(loss.detach()))
        checkpoint = Path(custom.distill_checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        runner.save(checkpoint)
    report = dict(task=args.task, teacher=str(Path(custom.teacher).resolve()),
                  motion_file=env.cfg.motion.motion_file,
                  first_episode_completion_rate=float(success.float().mean()),
                  first_episode_duration_s=(durations.float() * env.dt).cpu().tolist(),
                  physical_failures=failures, motion_completions=completions,
                  first_episode_censored=int((~ended).sum()),
                  distilled_checkpoint=(str(Path(custom.distill_checkpoint).resolve())
                                        if custom.distill_checkpoint else None),
                  mean_imitation_loss=(sum(imitation_loss) / len(imitation_loss)
                                       if imitation_loss else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
