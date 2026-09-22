"""Evaluate the legacy teacher through the new environment/target contract only."""
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
    custom, remaining = parser.parse_known_args()
    output = Path(custom.output)
    if output.exists():
        raise FileExistsError(output)
    sys.argv = [sys.argv[0]] + remaining
    import isaacgym  # noqa: F401
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    from legged_gym.envs.base.legged_robot import euler_from_quaternion

    args = get_args()
    if not args.task.startswith("g1_dynamics_tracker"):
        raise ValueError("select a g1_dynamics_tracker task")
    args.headless = True
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.noise.add_noise = False
    cfg.env.randomize_start_pos = False
    cfg.domain_rand.domain_rand_general = False
    for name in ("randomize_gravity", "randomize_friction", "randomize_base_mass",
                 "randomize_base_com", "push_robots", "push_end_effector",
                 "randomize_motor", "action_delay", "randomize_mimic_obs"):
        setattr(cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    teacher = torch.jit.load(custom.teacher, map_location=env.device).eval()
    env._eval_scenario_motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.num_motions()
    env._eval_scenario_motion_times = torch.zeros(env.num_envs, device=env.device)
    env._eval_scenario_dof_pos_scale = torch.ones_like(env.dof_pos)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()
    history = None
    previous_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(ended)
    durations = torch.full((env.num_envs,), custom.steps, dtype=torch.long, device=env.device)
    failures = completions = 0

    def legacy_observation():
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
                             env.dof_vel*env.obs_scales.dof_vel, previous_action), -1)
        velocity_start = 5 + env.num_actions
        proprio[:, [velocity_start+i for i in (4, 5, 10, 11)]] = 0.
        return torch.cat((mimic, proprio), -1)

    with torch.no_grad():
        for step in range(custom.steps):
            current = legacy_observation()
            if history is None:
                history = current[:, None].repeat(1, 10, 1)
            action = teacher(torch.cat((current, history.flatten(1)), -1)).clamp(
                -env.cfg.normalization.clip_actions/env.cfg.control.action_scale,
                env.cfg.normalization.clip_actions/env.cfg.control.action_scale)
            target = env.default_dof_pos_all + action*env.cfg.control.action_scale
            normalized = ((target-env.current_reference[:, 8:31]) /
                          env.target_transform.scale).clamp(-.999, .999)
            raw = torch.atanh(normalized)
            executed = env.target_transform(raw, env.current_reference[:, 8:31])
            _, _, _, done, info = env.step(raw)
            first = done.bool() & ~ended
            durations[first] = step + 1
            success[first] = info["motion_completed"][first]
            ended |= done.bool()
            failures += int(info["physical_failure"].sum())
            completions += int(info["motion_completed"].sum())
            history = torch.cat((history[:, 1:], current[:, None]), 1)
            previous_action.copy_((executed-env.default_dof_pos_all)/env.cfg.control.action_scale)
            reset_ids = done.nonzero(as_tuple=False).flatten()
            if reset_ids.numel():
                new_current = legacy_observation()
                history[reset_ids] = new_current[reset_ids, None]
                previous_action[reset_ids] = 0.
    report = dict(task=args.task, teacher=str(Path(custom.teacher).resolve()),
                  motion_file=env.cfg.motion.motion_file,
                  first_episode_completion_rate=float(success.float().mean()),
                  first_episode_duration_s=(durations.float()*env.dt).cpu().tolist(),
                  physical_failures=failures, motion_completions=completions,
                  first_episode_censored=int((~ended).sum()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
