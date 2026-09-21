"""Short, actual PPO proof: restored references used; WM/base frozen; DTERA updated.

Uses standard legged_gym CLI arguments. Forces WM-only episodes to exercise the
post-24-frame path. This is an integration test, not the mixed-input pilot.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'legged_gym'), str(ROOT/'rsl_rl'), str(ROOT/'pose')]
import isaacgym  # must precede torch
import torch
import wandb
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main():
    args = get_args()
    if args.task not in ('g1_motion_wm_dtera', 'g1_motion_wm_dtera_v2',
                         'g1_motion_wm_dtera_legs', 'g1_motion_wm_dtera_deploy_v3',
                         'g1_motion_wm_dtera_deploy_v4'):
        raise ValueError('Specify a Motion-WM DTERA training task')
    if args.num_envs is None or args.num_envs > 8 or args.max_iterations not in (2,3):
        raise ValueError('Smoke proof requires <=8 environments and 2 or 3 iterations')
    args.headless = True
    output = ROOT/'legged_gym/logs'/args.proj_name/args.exptid
    output.mkdir(parents=True, exist_ok=False)
    wandb.init(mode='disabled')
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.motion_wm.mode_probabilities = (0.,0.,1.)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                             train_cfg=train_cfg, log_root=str(output), init_wandb=False)
    pipeline = env.motion_reference_pipeline
    actor = runner.alg.actor_critic
    wm_before, base_before, adapter_before = snapshot(pipeline.model), snapshot(actor.base_actor), snapshot(actor.adapter)
    encoders_before = {name: snapshot(getattr(actor, name)) for name in
                       ('history_encoder', 'tracking_error_history_encoder')}
    # Validate clean reward buffers are not modified by the student processor.
    reward_before = env._ref_dof_pos.clone()
    privileged, clean = env._get_mimic_obs()
    clean_before, privileged_before = clean.clone(), privileged.clone()
    env._randomize_mimic_obs(clean)
    raw_delta = pipeline.corrupted - clean
    raw_delta[:, 3] = torch.atan2(torch.sin(raw_delta[:, 3]),
                                 torch.cos(raw_delta[:, 3]))
    raw_reference_max_error = float(raw_delta.abs().max())
    raw_reference_used = raw_reference_max_error < 1e-6
    torch.testing.assert_close(env._ref_dof_pos, reward_before, rtol=0, atol=0)
    torch.testing.assert_close(clean, clean_before, rtol=0, atol=0)
    torch.testing.assert_close(privileged, privileged_before, rtol=0, atol=0)
    # Undo the extra observation used by the preceding contract check.
    env.reset()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    wm_after, base_after, adapter_after = snapshot(pipeline.model), snapshot(actor.base_actor), snapshot(actor.adapter)
    wm_equal = all(torch.equal(v, wm_after[k]) for k,v in wm_before.items())
    base_equal = all(torch.equal(v, base_after[k]) for k,v in base_before.items())
    adapter_changed = any(not torch.equal(v, adapter_after[k]) for k,v in adapter_before.items())
    branch_updates = {name: sum(float((adapter_after[k]-v).square().sum())
                               for k,v in adapter_before.items() if k.startswith(name+'.'))**.5
                      for name in ('dynamics_branch','tracking_branch')}
    encoder_updates = {name: sum(float((snapshot(getattr(actor,name))[k]-v).square().sum())
                                for k,v in before.items())**.5
                       for name,before in encoders_before.items()}
    metrics = runner.alg.anyadapter_metrics
    with torch.no_grad():
        diagnostics = actor.action_diagnostics(env.obs_buf)
    processed_error = pipeline.processed[:,8:] - env.dof_pos
    torch.testing.assert_close(env.tracking_error_history[:,-1,:23], processed_error)
    proof = dict(task=args.task, iterations=args.max_iterations, environments=env.num_envs,
                 restored_frames=pipeline.total_refined_frames, wm_weights_unchanged=wm_equal,
                 base_weights_unchanged=base_equal, adapter_weights_changed=adapter_changed,
                 wm_has_no_grad=all(p.grad is None for p in pipeline.model.parameters()),
                 policy_obs_dim=int(env.obs_buf.shape[-1]),
                 finite_observations=bool(torch.isfinite(env.obs_buf).all()),
                 clean_reward_buffers_unchanged_by_pipeline=True,
                 artificial_corruption=pipeline.artificial_corruption,
                 raw_reference_used=raw_reference_used,
                 raw_reference_max_error=raw_reference_max_error,
                 starts_at_motion_frame_zero=bool(torch.all(env._motion_time_offsets == 0)),
                 motion=args.motion_file, seed=args.seed,
                 branch_weight_update_l2=branch_updates, encoder_weight_update_l2=encoder_updates,
                 finite_training_metrics=all(math.isfinite(float(v)) for v in metrics.values()),
                 applied_residual_abs_max=float(diagnostics['applied_delta'].abs().max()),
                 processed_reference_used_by_tracking_history=True,
                 optimizer_ownership=runner.alg.optimizer_ownership,
                 training_metrics=metrics)
    (output/'smoke_proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))
    assert wm_equal and base_equal and adapter_changed and proof['wm_has_no_grad']
    assert proof['restored_frames'] > 0 and proof['policy_obs_dim'] == 3695 and proof['finite_observations']
    assert proof['finite_training_metrics'] and all(value > 0 for value in branch_updates.values())
    assert all(value > 0 for value in encoder_updates.values())
    if args.task in ('g1_motion_wm_dtera_deploy_v3',
                     'g1_motion_wm_dtera_deploy_v4'):
        assert not proof['artificial_corruption']
        assert proof['raw_reference_used'] and proof['starts_at_motion_frame_zero']
    wandb.finish()


if __name__ == '__main__':
    main()
