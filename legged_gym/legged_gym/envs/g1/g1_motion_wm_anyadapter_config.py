"""Independent Motion-WM + paper-style AnyAdapter configuration."""

from pathlib import Path

from .g1_mimic_distill_anyadapter_config import (
    ANYADAPTER_STATE_INDICES,
    G1MimicStuAnyAdapterV6Cfg,
)
from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfgPPO


class G1MotionWMAnyAdapterCfg(G1MimicStuAnyAdapterV6Cfg):
    class env(G1MimicStuAnyAdapterV6Cfg.env):
        num_envs = 64
        use_anyadapter = True
        normalize_obs = False
        anyadapter_history_len = 79
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74
        anyadapter_added_obs_dim = 79 * 74
        anyadapter_final_policy_obs_dim = 7001
        anyadapter_fill_history_on_reset = True
        use_tracking_error_history = False
        anyadapter_context_dim = 0

    class motion_wm:
        checkpoint = str(Path(__file__).resolve().parents[3] /
                         "logs/motion_world_model/full_stable_v2/best.pt")
        mode_probabilities = (1 / 3, 1 / 3, 1 / 3)
        corruption_preset = "formal"
        artificial_corruption = True

    class domain_rand(G1MimicStuAnyAdapterV6Cfg.domain_rand):
        randomize_mimic_obs = False


class G1MotionWMAnyAdapterCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterOpenTrackActorCritic"
        algorithm_class_name = "PPOAnyAdapterOpenTrack"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_motion_wm_anyadapter"
        run_name = "opentrack_fresh_seed42_v1"
        init_at_random_ep_len = False
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 100
        constant_save_interval = True
        resume = False

    class policy(G1MimicPrivCfgPPO.policy):
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
        base_obs_dim = 1155
        history_len = 79
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        latent_dim = 128
        world_model_hidden_dims = [512, 512, 256, 256, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "silu"
        init_noise_std = 0.05
        fix_action_std = False
        adapter_gain = 1.0
        freeze_base = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        num_mini_batches = 16
        # Continuation fine-tuning from model_4900: late training had reached
        # the adaptive LR floor while learned exploration std kept increasing.
        schedule = "fixed"
        entropy_coef = 5e-4
        policy_learning_rate = 5e-6
        world_model_learning_rate = 3e-5
        action_std_min = 0.03
        action_std_max = 0.35
        world_model_loss_coef = 1.0
        world_model_loss_type = "l1"
        world_model_sequence_length = 20
        world_model_num_epochs = 1
        world_model_component_weights = [5.0, 5.0, 1.0, 0.5]
        adapter_reg_coef = 0.0
        stand_anchor_coef = 0.0
        synthetic_stand_anchor_coef = 0.0
