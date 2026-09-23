"""Fail fast if the baseline-anchored adapter changes TWIST's tested contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl')]

import isaacgym
import torch

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicStuRLCfg
from legged_gym.envs.g1.g1_twist_baseline_adapter_config import (
    G1TwistBaselineAdapterCfg, G1TwistBaselineAdapterCfgPPO,
    G1TwistBaselineAdapterRefineCfgPPO,
)
from legged_gym.gym_utils.helpers import class_to_dict
from rsl_rl.modules.actor_critic_twist_anyadapter_opentrack import (
    TwistAnyAdapterOpenTrackActorCritic,
)
from rsl_rl.algorithms.ppo_twist_baseline_adapter import PPOTwistBaselineAdapter


def main():
    base = G1MimicStuRLCfg()
    cfg = G1TwistBaselineAdapterCfg()
    ppo = G1TwistBaselineAdapterCfgPPO()
    for section in ('control', 'rewards', 'domain_rand', 'noise', 'terrain', 'init_state'):
        assert class_to_dict(getattr(cfg, section)) == class_to_dict(getattr(base, section)), section
    for name in ('tar_obs_steps', 'n_proprio', 'n_mimic_obs', 'history_len',
                 'num_observations', 'num_actions', 'episode_length_s', 'obs_type'):
        assert getattr(cfg.env, name) == getattr(base.env, name), name
    assert cfg.env.num_observations == 1155
    assert cfg.env.anyadapter_final_policy_obs_dim == 7001
    assert Path(cfg.motion.motion_file).is_file()
    assert Path(ppo.policy.base_actor_jit_path).is_file()

    torch.manual_seed(7)
    actor = TwistAnyAdapterOpenTrackActorCritic(
        num_actions=23, num_critic_observations=cfg.env.num_privileged_obs,
        **class_to_dict(ppo.policy),
    ).eval()
    base_jit = torch.jit.load(ppo.policy.base_actor_jit_path, map_location='cpu').eval()
    obs = torch.randn(4, cfg.env.anyadapter_final_policy_obs_dim)
    with torch.no_grad():
        actual = actor.act_inference(obs)
        expected = base_jit(obs[:, :1155])
    difference = float((actual - expected).abs().max())
    assert difference < 2e-5, difference
    assert not any(p.requires_grad for p in actor.layerwise_actor.base_layers.parameters())
    assert ppo.runner.algorithm_class_name == 'PPOTwistBaselineAdapter'
    algorithm = PPOTwistBaselineAdapter(None, actor, device='cpu', **class_to_dict(ppo.algorithm))
    assert algorithm.adapter_reg_coef > 0.0
    assert algorithm.effective_adapter_reg_coef(0) == 4.0
    assert algorithm.effective_adapter_reg_coef(300) == 2.0
    assert all(group['weight_decay'] == 0.0 for group in algorithm.wm_optimizer.param_groups)
    refine = G1TwistBaselineAdapterRefineCfgPPO()
    assert refine.runner.algorithm_class_name == 'PPOTwistBaselineAdapter'
    assert refine.policy.adapter_gain == 0.25
    assert refine.algorithm.policy_learning_rate == 2e-6
    assert refine.algorithm.world_model_loss_coef == 0.0
    assert refine.algorithm.freeze_world_model
    assert Path(refine.algorithm.policy_anchor_checkpoint).is_file()
    assert refine.algorithm.policy_anchor_coef == 10.0
    assert refine.algorithm.adapter_reg_initial_coef == 4.0
    assert refine.algorithm.adapter_reg_coef == 4.0
    assert refine.algorithm.adapter_reg_anneal_iterations == 0
    assert refine.algorithm.adapter_tail_threshold == 0.12
    assert refine.algorithm.adapter_tail_coef == 8.0
    refinement = PPOTwistBaselineAdapter(
        None, actor, device='cpu', **class_to_dict(refine.algorithm)
    )
    actor.load_state_dict(torch.load(
        refine.algorithm.policy_anchor_checkpoint, map_location='cpu'
    )['model_state_dict'])
    with torch.no_grad():
        anchored, absolute = refinement._policy_anchor_penalty(obs, actor.actor_mean(obs))
    assert anchored.item() < 1e-12 and absolute.item() < 1e-12
    assert not any(p.requires_grad for p in actor.history_encoder.parameters())
    assert not any(p.requires_grad for p in actor.world_model.parameters())
    print(f'TWIST contract preserved; initial adapter/base max_abs_diff={difference:.3e}')


if __name__ == '__main__':
    main()
