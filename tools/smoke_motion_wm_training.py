"""Short, actual PPO proof: restored references used; WM/base frozen; DTERA updated.

Uses standard legged_gym CLI arguments. Forces WM-only episodes to exercise the
post-24-frame path. This is an integration test, not the mixed-input pilot.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaacgym  # must precede torch
import torch
import wandb
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main():
    args = get_args()
    if args.task != 'g1_motion_wm_dtera' or not args.resumeid:
        raise ValueError('Specify --task g1_motion_wm_dtera and --resumeid for a verified DTERA checkpoint')
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
    # Validate clean reward buffers are not modified by the student processor.
    reward_before = env._ref_dof_pos.clone()
    privileged, clean = env._get_mimic_obs()
    clean_before, privileged_before = clean.clone(), privileged.clone()
    env._randomize_mimic_obs(clean)
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
    proof = dict(task=args.task, iterations=args.max_iterations, environments=env.num_envs,
                 restored_frames=pipeline.total_refined_frames, wm_weights_unchanged=wm_equal,
                 base_weights_unchanged=base_equal, adapter_weights_changed=adapter_changed,
                 wm_has_no_grad=all(p.grad is None for p in pipeline.model.parameters()),
                 policy_obs_dim=int(env.obs_buf.shape[-1]),
                 finite_observations=bool(torch.isfinite(env.obs_buf).all()),
                 clean_reward_buffers_unchanged_by_pipeline=True,
                 motion=args.motion_file, seed=args.seed)
    (output/'smoke_proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))
    assert wm_equal and base_equal and adapter_changed and proof['wm_has_no_grad']
    assert proof['restored_frames'] > 0 and proof['policy_obs_dim'] == 3695 and proof['finite_observations']
    wandb.finish()


if __name__ == '__main__':
    main()
