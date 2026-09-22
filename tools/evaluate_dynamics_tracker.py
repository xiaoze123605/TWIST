"""Headless deterministic-action evaluation of a new tracker checkpoint.

Metrics are episode-frame aggregates, not a success-rate benchmark against
TWIST. Failed episodes restart; reports include the number of physical failures.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'legged_gym'), str(ROOT/'rsl_rl'), str(ROOT/'pose')]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--zero-latent', action='store_true', help='Inference-only dynamics conditioning ablation')
    custom, remaining = parser.parse_known_args()
    if custom.steps < 1:
        raise ValueError('steps must be positive')
    output = Path(custom.output)
    if output.exists():
        raise FileExistsError(output)
    sys.argv = [sys.argv[0]] + remaining
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    from rsl_rl.modules.dynamics_tracker import DynamicsTrackerActorCritic
    from rsl_rl.modules.dynamics_tracker_runtime import DeploymentPolicy
    args = get_args()
    if not args.task.startswith('g1_dynamics_tracker'):
        raise ValueError('requires a dynamics tracker task')
    args.headless = True
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.noise.add_noise = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    checkpoint = torch.load(custom.model, map_location=env.device)
    spec = env.deployment_spec()
    spec['use_dynamics_latent'] = checkpoint['train_cfg']['policy']['use_dynamics_latent']
    if spec != checkpoint['deployment_spec']:
        raise ValueError('checkpoint and evaluation environment contracts differ')
    ac = DynamicsTrackerActorCritic(env.num_obs, env.num_privileged_obs,
                                    **checkpoint['train_cfg']['policy']).to(env.device).eval()
    ac.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if custom.zero_latent:
        ac.use_dynamics_latent = False
        spec['use_dynamics_latent'] = False
    deployment = torch.jit.script(DeploymentPolicy(ac, spec).to(env.device).eval())
    # Fixed clip and phase assignment for each environment, rather than RSI.
    env._eval_scenario_motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.num_motions()
    env._eval_scenario_motion_times = torch.zeros(env.num_envs, device=env.device)
    env._eval_scenario_dof_pos_scale = torch.ones_like(env.dof_pos)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.compute_observations()
    totals = dict(joint_squared=0., height_squared=0., saturation=0., physical_failures=0,
                  motion_completions=0, timeouts=0, resets=0)
    failure_names = ('contact', 'height', 'tilt', 'speed', 'pose')
    reason_counts = {name: 0 for name in failure_names}
    first_end = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_success = torch.zeros_like(first_end)
    first_steps = torch.full((env.num_envs,), custom.steps, device=env.device, dtype=torch.long)
    first_failure_snapshots = [None] * env.num_envs
    with torch.no_grad():
        for step in range(custom.steps):
            obs = env.get_observations()
            raw = ac(obs)
            torch.testing.assert_close(deployment(obs), env.target_transform(raw, env.current_reference[:,8:31]))
            _, _, _, done, info = env.step(raw)
            first_done = done.bool() & ~first_end
            first_steps[first_done] = step + 1
            first_success[first_done] = info['motion_completed'][first_done]
            for env_id in first_done.nonzero(as_tuple=False).flatten().tolist():
                first_failure_snapshots[env_id] = dict(
                    step=step + 1,
                    height_error_m=float(info['tracking_height_error'][env_id]),
                    roll_pitch_rad=info['base_roll_pitch'][env_id].cpu().tolist(),
                    joint_error_rad=info['tracking_joint_error'][env_id].cpu().tolist(),
                    joint_velocity_error_rad_s=info['tracking_joint_velocity_error'][env_id].cpu().tolist(),
                    sent_reference_offset_rad=info['sent_reference_offset'][env_id].cpu().tolist(),
                    torque_limit_ratio=info['torque_limit_ratio'][env_id].cpu().tolist())
            first_end |= done.bool()
            totals['joint_squared'] += float(info['tracking_joint_error'].square().sum())
            totals['height_squared'] += float(info['tracking_height_error'].square().sum())
            totals['saturation'] += float(info['torque_saturation_fraction'].sum())
            totals['physical_failures'] += int(info['physical_failure'].sum())
            totals['motion_completions'] += int(info['motion_completed'].sum())
            totals['timeouts'] += int(info['time_outs'].sum())
            totals['resets'] += int(done.sum())
            for name in failure_names:
                reason_counts[name] += int(info['failure_' + name].sum())
    samples = custom.steps * env.num_envs
    report = dict(task=args.task, checkpoint=str(Path(custom.model).resolve()),
                  motion_file=env.cfg.motion.motion_file, seed=env.cfg.seed,
                  frames=samples, noise=False, resets=totals['resets'],
                  latent_enabled=ac.use_dynamics_latent, zero_latent_ablation=custom.zero_latent,
                  physical_failures=totals['physical_failures'],
                  motion_completions=totals['motion_completions'], timeouts=totals['timeouts'],
                  joint_rmse_rad=(totals['joint_squared']/samples/23)**.5,
                  root_height_rmse_m=(totals['height_squared']/samples)**.5,
                  torque_saturation_fraction=totals['saturation']/samples,
                  first_episode_completion_rate=float(first_success.float().mean()),
                  first_episode_censored=int((~first_end).sum()),
                  first_episode_duration_s=(first_steps.float()*env.dt).cpu().tolist(),
                  failure_reason_counts=reason_counts,
                  first_failure_snapshots=first_failure_snapshots,
                  scripted_action_parity=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
