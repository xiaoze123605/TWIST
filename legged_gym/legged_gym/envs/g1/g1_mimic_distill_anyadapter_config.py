"""
G1 TWIST student + AnyAdapter configuration.

This config keeps the original TWIST student observation as the base actor
input and appends a separate AnyAdapter history after it:

    base_obs_dim = 1155
    anyadapter history = 20 * (51 selected state dims + 23 actions) = 1480
    final actor obs dim = 2635

The selected state deliberately excludes reference motion, TWIST's built-in
history block, and the action_history_buf slice already present in the base obs.
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicPrivCfgPPO


ANYADAPTER_STATE_INDICES = (
    list(range(31, 36)) +  # base_ang_vel + roll/pitch
    list(range(36, 59)) +  # dof_pos - default
    list(range(59, 82))    # dof_vel
)

DEFAULT_REF_DOF_POS = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2,
    0.0, -0.4, 0.0, 1.2,
]


class G1MimicStuAnyAdapterCfg(G1MimicStuRLCfg):
    class env(G1MimicStuRLCfg.env):
        use_anyadapter = True
        normalize_obs = False

        # Original TWIST student policy input before appending AnyAdapter history.
        base_obs_dim = 1155

        anyadapter_history_len = 20
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74

        # For reference/documentation only.  Do not assign this to
        # num_observations here; the mixin updates env.num_obs after BaseTask
        # has allocated the original TWIST observation buffer.
        anyadapter_added_obs_dim = 1480
        anyadapter_final_policy_obs_dim = 2635


class G1MimicStuAnyAdapterCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0721_twist_rlbcstu/traced/0721_twist_rlbcstu-23500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.25
        init_noise_std = 0.2
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.1
        adapter_reg_coef = 1e-3
        world_model_loss_type = "smooth_l1"


# ======================== V2 Conservative ========================

class G1MimicStuAnyAdapterV2Cfg(G1MimicStuRLCfg):
    """V2 conservative: smaller adapter delta, stronger regularization, lower wm coef.

    CRITICAL: normalize_obs is set to False because the JIT base actor contains its
    own internal Normalizer fitted during the original TWIST student training.  If
    the runner also normalizes observations the base actor receives double-normalized
    inputs, producing severely wrong actions.
    """

    class env(G1MimicStuRLCfg.env):
        use_anyadapter = True
        normalize_obs = False

        # Original TWIST student policy input before appending AnyAdapter history.
        base_obs_dim = 1155

        anyadapter_history_len = 20
        anyadapter_state_indices = ANYADAPTER_STATE_INDICES
        anyadapter_hist_state_dim = 51
        anyadapter_history_frame_dim = 74

        # For reference/documentation only.
        anyadapter_added_obs_dim = 1480
        anyadapter_final_policy_obs_dim = 2635

    class motion(G1MimicStuRLCfg.motion):
        # Slow curriculum: adapter sees easy motions for much longer before
        # progressing to max difficulty.  This prevents overfitting to extreme
        # dynamics and keeps corrections small for standing / simple gaits.
        motion_curriculum_gamma = 0.002


class G1MimicStuAnyAdapterV2CfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_v2"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0721_twist_rlbcstu/traced/0721_twist_rlbcstu-23500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.02
        init_noise_std = 0.05
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.1
        adapter_reg_coef = 2e-1
        world_model_loss_type = "smooth_l1"
        # Weight decay on adapter + history_encoder params to prevent large
        # weight norms that cause tanh saturation / action jitter.
        weight_decay = 1e-4


# ======================== Safe Stand-Preserving ========================

class G1MimicStuAnyAdapterSafeCfg(G1MimicStuAnyAdapterV2Cfg):
    class env(G1MimicStuAnyAdapterV2Cfg.env):
        use_anyadapter = True
        normalize_obs = False


class G1MimicStuAnyAdapterSafeCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "TwistAnyAdapterActorCritic"
        algorithm_class_name = "PPOAnyAdapter"
        runner_class_name = "OnPolicyRunnerMimic"
        experiment_name = "g1_twist_anyadapter_safe"
        run_name = ""

    class policy(G1MimicPrivCfgPPO.policy):
        # Must point to the frozen exported TWIST student JIT actor.
        base_actor_jit_path = "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0721_twist_rlbcstu/traced/0721_twist_rlbcstu-23500-jit.pt"

        base_obs_dim = 1155
        history_len = 20
        hist_state_dim = 51
        history_frame_dim = 74
        wm_target_indices = ANYADAPTER_STATE_INDICES
        default_ref_dof_pos = DEFAULT_REF_DOF_POS

        latent_dim = 32
        adapter_hidden_dims = [128, 128]
        world_model_hidden_dims = [256, 256]
        critic_hidden_dims = [512, 256, 128]
        action_delta_scale = 0.02
        adapter_gain = 1.0
        init_noise_std = 0.05
        freeze_base = True
        activation = "elu"
        use_conv_history = True

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        world_model_loss_coef = 0.05
        adapter_reg_coef = 0.05
        stand_anchor_coef = 2.0
        synthetic_stand_anchor_coef = 2.0
        synthetic_stand_root_height = 0.793
        stand_vel_threshold = 0.05
        stand_dof_threshold = 0.15
        world_model_loss_type = "smooth_l1"
        weight_decay = 1e-4
