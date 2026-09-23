"""Baseline-anchored AnyAdapter, preserving TWIST's tracking environment.

The original 1155-D student observation, current reference, rewards, motion
curriculum, PD controller and reset logic all come from G1MimicStuRLCfg.
Only a separate 79-frame dynamics history is appended for the adapter and
its action world model. The TWIST actor is frozen and the layer adapters start
at exactly zero, so the initial deterministic policy equals the base JIT.
"""

from pathlib import Path

from .g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicPrivCfgPPO
from .g1_mimic_distill_anyadapter_config import ANYADAPTER_STATE_INDICES


_ROOT = Path(__file__).resolve().parents[4]


class G1TwistBaselineAdapterCfg(G1MimicStuRLCfg):
    class env(G1MimicStuRLCfg.env):
        normalize_obs = False  # the frozen TWIST JIT owns its normalizer
        use_anyadapter = True
        anyadapter_history_len = 79
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74
        anyadapter_added_obs_dim = 79 * 74
        anyadapter_final_policy_obs_dim = 1155 + anyadapter_added_obs_dim
        anyadapter_fill_history_on_reset = True
        anyadapter_context_dim = 0

    class motion(G1MimicStuRLCfg.motion):
        motion_file = str(_ROOT / 'legged_gym/motion_data_configs/'
                          'wm_dtera_prepared_20260916_local/train.yaml')


class G1TwistBaselineAdapterCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'TwistAnyAdapterOpenTrackActorCritic'
        # PPOAny2Track keeps the same dynamics-WM path while allowing a
        # baseline-preserving adapter penalty. The OpenTrack specialization
        # explicitly forbids that penalty and is unsuitable for this run.
        algorithm_class_name = 'PPOTwistBaselineAdapter'
        runner_class_name = 'OnPolicyRunnerMimic'
        experiment_name = 'g1_twist_baseline_adapter'
        run_name = ''
        num_steps_per_env = 24
        max_iterations = 30000
        save_interval = 50
        constant_save_interval = True
        init_at_random_ep_len = True

    class policy(G1MimicPrivCfgPPO.policy):
        base_actor_jit_path = str(_ROOT / 'legged_gym/logs/g1_stu_rl/'
                                  '0529_twist_rlbcstu/traced/'
                                  '0529_twist_rlbcstu-36500-jit.pt')
        base_obs_dim = 1155
        history_len = 79
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        latent_dim = 128
        world_model_hidden_dims = [512, 512, 256, 256, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'silu'
        init_noise_std = 0.05
        fix_action_std = False
        adapter_gain = 1.0
        freeze_base = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        num_mini_batches = 16
        schedule = 'fixed'
        entropy_coef = 5e-4
        policy_learning_rate = 1e-5
        world_model_learning_rate = 3e-5
        action_std_min = 0.03
        action_std_max = 0.2
        world_model_loss_coef = 1.0
        world_model_loss_type = 'smooth_l1'
        world_model_sequence_length = 20
        world_model_num_epochs = 1
        world_model_component_weights = [5.0, 5.0, 1.0, 0.5]
        adapter_reg_initial_coef = 4.0
        adapter_reg_coef = 2.0
        adapter_reg_anneal_iterations = 300
        adapter_bias_reg_coef = 0.0
        stand_anchor_coef = 0.0
        synthetic_stand_anchor_coef = 0.0
        weight_decay = 0.0


class G1TwistBaselineAdapterRefineCfgPPO(G1TwistBaselineAdapterCfgPPO):
    """Short guarded experiment from the selected 1600-update checkpoint."""

    class runner(G1TwistBaselineAdapterCfgPPO.runner):
        experiment_name = 'g1_twist_baseline_adapter_refine'
        # The first held-out screen found a fall by update 100. Keep this a
        # short diagnostic task until a revised run passes paired validation.
        max_iterations = 100

    class policy(G1TwistBaselineAdapterCfgPPO.policy):
        adapter_gain = 0.25

    class algorithm(G1TwistBaselineAdapterCfgPPO.algorithm):
        policy_learning_rate = 2e-6
        world_model_loss_coef = 0.0
        freeze_world_model = True
        policy_anchor_checkpoint = str(
            _ROOT / 'legged_gym/logs/g1_twist_baseline_adapter/'
                    'anchored_opt_v2_long/model_1600.pt')
        policy_anchor_coef = 10.0
        adapter_reg_initial_coef = 4.0
        adapter_reg_coef = 4.0
        adapter_reg_anneal_iterations = 0
        adapter_tail_threshold = 0.12
        adapter_tail_coef = 8.0
