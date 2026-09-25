"""Independent Motion-WM + paper-style AnyAdapter configuration."""

from pathlib import Path

from .g1_mimic_distill_anyadapter_config import (
    ANYADAPTER_STATE_INDICES,
    G1MimicStuAnyAdapterV6Cfg,
)
from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfgPPO
from .g1_twist_baseline_adapter_config import (
    G1TwistBaselineAdapterCfg, G1TwistBaselineAdapterCfgPPO,
)


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


class G1MotionWMAnyAdapterCleanCfg(G1MotionWMAnyAdapterCfg):
    """Motion-WM + AnyAdapter on the audited, disjoint training split."""

    class motion(G1MotionWMAnyAdapterCfg.motion):
        motion_file = str(Path(__file__).resolve().parents[4] /
                          "legged_gym/motion_data_configs/"
                          "wm_dtera_prepared_20260916_local/train.yaml")

    class motion_wm(G1MotionWMAnyAdapterCfg.motion_wm):
        # Most deployments use raw references; retain corrupted and WM
        # references so the same actor can handle both at inference.
        mode_probabilities = (0.5, 0.25, 0.25)


class G1MotionWMAnyAdapterCleanCfgPPO(G1MotionWMAnyAdapterCfgPPO):
    class runner(G1MotionWMAnyAdapterCfgPPO.runner):
        algorithm_class_name = "PPOTwistBaselineAdapter"
        experiment_name = "g1_motion_wm_anyadapter_clean"
        run_name = ""
        max_iterations = 30000
        save_interval = 50
        init_at_random_ep_len = False
        resume = False

    class algorithm(G1MotionWMAnyAdapterCfgPPO.algorithm):
        # The previous anchored run used gain 0.25, coefficient 10 on the
        # source policy, and a frozen dynamics encoder. Here the action WM
        # keeps learning and the actor can make useful root/yaw corrections.
        policy_learning_rate = 5e-6
        world_model_learning_rate = 3e-5
        world_model_loss_type = "l1"
        world_model_loss_coef = 1.0
        freeze_world_model = False
        policy_anchor_coef = 0.0
        adapter_reg_initial_coef = 0.05
        adapter_reg_coef = 0.05
        adapter_reg_anneal_iterations = 0
        adapter_tail_threshold = 0.35
        adapter_tail_coef = 1.0
        action_std_min = 0.03
        action_std_max = 0.2


_CLEAN_1000 = str(Path(__file__).resolve().parents[3] /
                  "logs/g1_motion_wm_anyadapter_clean/"
                  "clean_motionwm_pilot_500_20260924_r400/model_1000.pt")


class G1MotionWMAnyAdapterCleanGuardedCfgPPO(G1MotionWMAnyAdapterCleanCfgPPO):
    """Short paired pilot initialized from clean update 1000."""

    class runner(G1MotionWMAnyAdapterCleanCfgPPO.runner):
        max_iterations = 200

    class policy(G1MotionWMAnyAdapterCleanCfgPPO.policy):
        adapter_gain = 0.25

    class algorithm(G1MotionWMAnyAdapterCleanCfgPPO.algorithm):
        adapter_reg_initial_coef = 0.5
        adapter_reg_coef = 0.5
        adapter_tail_threshold = 0.2
        adapter_tail_coef = 2.0
        policy_anchor_checkpoint = _CLEAN_1000
        policy_anchor_coef = 1.0


class G1MotionWMAnyAdapterCleanGuardedFrozenCfgPPO(
        G1MotionWMAnyAdapterCleanGuardedCfgPPO):
    """Same guarded pilot with the trained dynamics representation fixed."""

    class algorithm(G1MotionWMAnyAdapterCleanGuardedCfgPPO.algorithm):
        freeze_world_model = True
        world_model_loss_coef = 0.0


class G1MotionWMAnyAdapterCleanRawCfg(G1MotionWMAnyAdapterCleanCfg):
    """Use the same audited split with raw references only."""

    class motion_wm(G1MotionWMAnyAdapterCleanCfg.motion_wm):
        mode_probabilities = (1.0, 0.0, 0.0)


_GUARDED_200 = str(Path(__file__).resolve().parents[3] /
                   "logs/g1_twist_baseline_adapter/"
                   "anchored_guarded_1600_pilot_v1_200_20260923/model_200.pt")


class G1MotionWMAnyAdapterBaselineCfg(G1TwistBaselineAdapterCfg):
    """Original TWIST rewards/curriculum with Motion-WM reference processing."""

    class motion_wm(G1MotionWMAnyAdapterCleanCfg.motion_wm):
        mode_probabilities = (0.5, 0.25, 0.25)


class G1MotionWMAnyAdapterBaselineRawCfg(G1MotionWMAnyAdapterBaselineCfg):
    class motion_wm(G1MotionWMAnyAdapterBaselineCfg.motion_wm):
        mode_probabilities = (1.0, 0.0, 0.0)


class G1MotionWMAnyAdapterBaselineCfgPPO(G1TwistBaselineAdapterCfgPPO):
    """Guarded continuation of the selected update 200 under baseline rewards."""

    class runner(G1TwistBaselineAdapterCfgPPO.runner):
        experiment_name = "g1_motion_wm_anyadapter_baseline_pilot"
        max_iterations = 200
        init_at_random_ep_len = False

    class policy(G1TwistBaselineAdapterCfgPPO.policy):
        adapter_gain = 0.25

    class algorithm(G1TwistBaselineAdapterCfgPPO.algorithm):
        policy_learning_rate = 2e-6
        world_model_loss_coef = 0.0
        freeze_world_model = True
        adapter_reg_initial_coef = 2.0
        adapter_reg_coef = 2.0
        adapter_reg_anneal_iterations = 0
        adapter_tail_threshold = 0.12
        adapter_tail_coef = 4.0
        policy_anchor_checkpoint = _GUARDED_200
        policy_anchor_coef = 5.0


_BASELINE_RAW_150 = str(Path(__file__).resolve().parents[3] /
                        "logs/g1_motion_wm_anyadapter_baseline_pilot/"
                        "baseline_reward_raw_pilot200_20260924/model_150.pt")


class G1MotionWMAnyAdapterBaselineContinueCfgPPO(
        G1MotionWMAnyAdapterBaselineCfgPPO):
    """Conservative continuation from the full-motion screened raw update 150."""

    class runner(G1MotionWMAnyAdapterBaselineCfgPPO.runner):
        experiment_name = "g1_motion_wm_anyadapter_baseline_continue"

    class algorithm(G1MotionWMAnyAdapterBaselineCfgPPO.algorithm):
        policy_learning_rate = 1e-6
        policy_anchor_checkpoint = _BASELINE_RAW_150
        policy_anchor_coef = 10.0


class G1MotionWMAnyAdapterStableCfg(G1MotionWMAnyAdapterBaselineRawCfg):
    """Fixed training distribution and the deployment raw-reference path."""

    class motion(G1MotionWMAnyAdapterBaselineRawCfg.motion):
        motion_curriculum = False


class G1MotionWMAnyAdapterStableCfgPPO(G1MotionWMAnyAdapterBaselineCfgPPO):
    """Audited raw150 continuation with bounded actor updates."""

    class runner(G1MotionWMAnyAdapterBaselineCfgPPO.runner):
        experiment_name = "g1_motion_wm_anyadapter_stable"
        max_iterations = 100
        save_interval = 25
        resume = False
        init_at_random_ep_len = False

    class policy(G1MotionWMAnyAdapterBaselineCfgPPO.policy):
        fix_action_std = True

    class algorithm(G1MotionWMAnyAdapterBaselineCfgPPO.algorithm):
        policy_learning_rate = 1e-6
        critic_learning_rate = 1e-4
        kl_stop_threshold = 0.012
        critic_warmup_iterations = 20
        num_learning_epochs = 3
        clip_param = 0.1
        entropy_coef = 0.0
        policy_anchor_checkpoint = _BASELINE_RAW_150
        policy_anchor_coef = 10.0
        anchor_replay_size = 16384
        anchor_replay_batch_size = 256
        anchor_replay_fraction = 0.2


class G1MotionWMAnyAdapterStableFastCfgPPO(G1MotionWMAnyAdapterStableCfgPPO):
    """Controlled actor-learning-rate comparison; other stable settings match."""

    class runner(G1MotionWMAnyAdapterStableCfgPPO.runner):
        experiment_name = "g1_motion_wm_anyadapter_stable_fast"

    class algorithm(G1MotionWMAnyAdapterStableCfgPPO.algorithm):
        policy_learning_rate = 2e-6
