"""Evaluate the frozen TWIST baseline on a fixed motion from phase zero."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'legged_gym'), str(ROOT/'rsl_rl'), str(ROOT/'pose')]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=400)
    custom, remaining = parser.parse_known_args()
    output = Path(custom.output)
    if output.exists():
        raise FileExistsError(output)
    sys.argv = [sys.argv[0]] + remaining
    import isaacgym
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    from legged_gym.envs.base.legged_robot import euler_from_quaternion
    args = get_args()
    if args.task != 'g1_stu_rl':
        raise ValueError('This evaluator requires --task g1_stu_rl')
    args.headless = True
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.noise.add_noise = False
    cfg.env.randomize_start_pos = False
    cfg.domain_rand.domain_rand_general = False
    for name in ('randomize_gravity', 'randomize_friction', 'randomize_base_mass',
                 'randomize_base_com', 'push_robots', 'push_end_effector',
                 'randomize_motor', 'action_delay', 'randomize_mimic_obs'):
        setattr(cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    policy = torch.jit.load(custom.policy, map_location=env.device).eval()
    env._eval_scenario_motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.num_motions()
    env._eval_scenario_motion_times = torch.zeros(env.num_envs, device=env.device)
    env._eval_scenario_dof_pos_scale = torch.ones_like(env.dof_pos)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(ended)
    durations = torch.full((env.num_envs,), custom.steps, dtype=torch.long, device=env.device)
    resets = failures = completions = 0
    corrections = []
    new_scales = torch.tensor([0.5, 0.3, 0.3, 0.5, 0.25, 0.2] * 2 +
                              [0.3] * 3 + [0.5] * 8, device=env.device)
    with torch.no_grad():
        for step in range(custom.steps):
            obs = env.get_observations()
            action = policy(obs.detach())
            clipped = action.clamp(-env.cfg.normalization.clip_actions / env.cfg.control.action_scale,
                                   env.cfg.normalization.clip_actions / env.cfg.control.action_scale)
            target = env.default_dof_pos_all + clipped * env.cfg.control.action_scale
            if (~ended).any():
                corrections.append((target[~ended] - env._ref_dof_pos[~ended]).cpu())
            _, _, _, done, info = env.step(action)
            completion = info['time_outs'].bool() & done.bool()
            first = done.bool() & ~ended
            durations[first] = step + 1
            success[first] = completion[first]
            ended |= done.bool()
            resets += int(done.sum())
            completions += int(completion.sum())
            failures += int((done.bool() & ~completion).sum())
    correction = torch.cat(corrections).abs()
    report = dict(task=args.task, checkpoint=str(Path(custom.policy).resolve()),
                  motion_file=env.cfg.motion.motion_file, frames=custom.steps*env.num_envs,
                  resets=resets, physical_failures=failures, motion_completions=completions,
                  first_episode_completion_rate=float(success.float().mean()),
                  first_episode_censored=int((~ended).sum()),
                  first_episode_duration_s=(durations.float()*env.dt).cpu().tolist(),
                  joint_names=list(env.dof_names),
                  baseline_target_minus_reference_abs_p95_rad=torch.quantile(correction, .95, dim=0).tolist(),
                  baseline_target_minus_reference_abs_max_rad=correction.amax(0).tolist(),
                  new_reference_scale_rad=new_scales.cpu().tolist(),
                  fraction_baseline_correction_outside_new_scale=(correction > new_scales.cpu()).float().mean(0).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
