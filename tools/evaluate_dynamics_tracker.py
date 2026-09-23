"""Deterministic-action evaluation of a dynamics tracker checkpoint.

Metrics are episode-frame aggregates, not a success-rate benchmark against
TWIST. Failed episodes restart; reports include the number of physical failures.
The Isaac Gym viewer is enabled by default and can be disabled with --headless.
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
    parser.add_argument('--episode-length-s', type=float, default=120.0,
                        help='Evaluation horizon; must exceed long reference clips')
    parser.add_argument('--random-phase', action='store_true')
    parser.add_argument('--require-all-motions', action='store_true',
                        help='Fail unless at least one environment is assigned to every configured clip')
    latent = parser.add_mutually_exclusive_group()
    latent.add_argument('--zero-latent', action='store_true', help='Inference-only dynamics conditioning ablation')
    latent.add_argument('--shuffled-latent', action='store_true', help='Shuffle latent vectors across environments')
    custom, remaining = parser.parse_known_args()
    if custom.steps < 1 or custom.episode_length_s <= 0:
        raise ValueError('steps and episode length must be positive')
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
    from rsl_rl.datasets.dynamics_dagger_buffer import build_actor_input
    args = get_args()
    if not args.task.startswith('g1_dynamics_tracker'):
        raise ValueError('requires a dynamics tracker task')
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.noise.add_noise = False
    cfg.env.episode_length_s = custom.episode_length_s
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if custom.require_all_motions and env.num_envs < env._motion_lib.num_motions():
        raise ValueError(
            f'full-corpus evaluation requires --num_envs >= {env._motion_lib.num_motions()}, '
            f'got {env.num_envs}')
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
    # Fixed clip assignment. Validation defaults to phase zero; random-phase
    # validation is explicit and reproducible from the configured seed.
    env._eval_scenario_motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.num_motions()
    if custom.random_phase:
        lengths = env._motion_lib.get_motion_length(env._eval_scenario_motion_ids)
        env._eval_scenario_motion_times = torch.rand(env.num_envs, device=env.device) * lengths
    else:
        env._eval_scenario_motion_times = torch.zeros(env.num_envs, device=env.device)
    env._eval_scenario_dof_pos_scale = torch.ones_like(env.dof_pos)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.compute_observations()
    totals = dict(joint_squared=0., height_squared=0., saturation=0., physical_failures=0,
                  motion_completions=0, timeouts=0, resets=0)
    per_env_joint_squared = torch.zeros(env.num_envs, device=env.device)
    per_env_height_squared = torch.zeros(env.num_envs, device=env.device)
    per_env_saturation = torch.zeros(env.num_envs, device=env.device)
    per_env_failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    failure_names = ('contact', 'height', 'tilt', 'speed', 'pose')
    reason_counts = {name: 0 for name in failure_names}
    first_end = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_success = torch.zeros_like(first_end)
    first_steps = torch.full((env.num_envs,), custom.steps, device=env.device, dtype=torch.long)
    first_failure_snapshots = [None] * env.num_envs
    with torch.no_grad():
        for step in range(custom.steps):
            obs = env.get_observations()
            raw = (ac.actor(build_actor_input(obs, ac, 'shuffled'))
                   if custom.shuffled_latent else ac(obs))
            if not custom.shuffled_latent:
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
            per_env_joint_squared += info['tracking_joint_error'].square().sum(-1)
            per_env_height_squared += info['tracking_height_error'].square()
            per_env_saturation += info['torque_saturation_fraction']
            per_env_failures += info['physical_failure'].long()
            for name in failure_names:
                reason_counts[name] += int(info['failure_' + name].sum())
    samples = custom.steps * env.num_envs
    per_motion = []
    motion_paths = list(env._motion_lib._motion_files)
    for motion_id, motion_path in enumerate(motion_paths):
        ids = (env._eval_scenario_motion_ids == motion_id).nonzero(as_tuple=False).flatten()
        if not ids.numel():
            continue
        motion_frames = custom.steps * ids.numel()
        per_motion.append(dict(
            motion_id=motion_id, motion_path=str(Path(motion_path).resolve()),
            environments=int(ids.numel()),
            first_episode_completion_rate=float(first_success[ids].float().mean()),
            first_episode_mean_duration_s=float(first_steps[ids].float().mean() * env.dt),
            first_episode_duration_s=(first_steps[ids].float() * env.dt).cpu().tolist(),
            first_episode_censored=int((~first_end[ids]).sum()),
            physical_failures=int(per_env_failures[ids].sum()),
            joint_rmse_rad=float((per_env_joint_squared[ids].sum()/motion_frames/23).sqrt()),
            root_height_rmse_m=float((per_env_height_squared[ids].sum()/motion_frames).sqrt()),
            torque_saturation_fraction=float(per_env_saturation[ids].sum()/motion_frames)))
    report = dict(task=args.task, checkpoint=str(Path(custom.model).resolve()),
                  motion_file=env.cfg.motion.motion_file, seed=env.cfg.seed,
                  episode_length_s=custom.episode_length_s,
                  random_phase=custom.random_phase,
                  frames=samples, noise=False, resets=totals['resets'],
                  latent_enabled=ac.use_dynamics_latent, zero_latent_ablation=custom.zero_latent,
                  shuffled_latent_ablation=custom.shuffled_latent,
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
                  per_motion=per_motion,
                  scripted_action_parity=(None if custom.shuffled_latent else True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
