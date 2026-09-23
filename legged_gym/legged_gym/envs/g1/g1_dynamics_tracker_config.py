"""Fresh-training configurations; no frozen base or continuation LR inheritance."""
from pathlib import Path

from legged_gym.envs.base.base_config import BaseConfig
from .g1_mimic_distill_config import G1MimicStuRLCfg


class G1DynamicsTrackerCfg(G1MimicStuRLCfg):
    class env(G1MimicStuRLCfg.env):
        num_envs = 256
        use_anyadapter = False
        normalize_obs = False
        dynamics_history_len = 79
        num_observations = 129 + 79 * 76
        num_privileged_obs = 135
        tar_obs_steps = [0]  # Streaming reference: no unavailable future frame.
        n_priv_mimic_obs = 58
        episode_length_s = 10
        randomize_start_yaw = False
        track_root = False

    class control(G1MimicStuRLCfg.control):
        action_mode = "reference"
        target_scales = [0.5, 0.3, 0.3, 0.5, 0.25, 0.2] * 2 + [0.3] * 3 + [0.5] * 8
        delay_max_steps = 0

    class domain_rand(G1MimicStuRLCfg.domain_rand):
        nominal_fraction = 1.0
        domain_rand_general = False
        randomize_gravity = False
        randomize_friction = False
        randomize_base_mass = False
        randomize_base_com = False
        push_robots = False
        push_end_effector = False
        randomize_motor = False
        action_delay = False  # new environment owns command delay explicitly
        randomize_mimic_obs = False

    class motion(G1MimicStuRLCfg.motion):
        motion_file = str(Path(__file__).resolve().parents[3] /
                          "motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml")
        motion_curriculum = False

    class rewards(G1MimicStuRLCfg.rewards):
        regularization_scale_curriculum = False
        class scales(G1MimicStuRLCfg.rewards.scales):
            # Reward scales are multiplied by dt; -50 gives -1 at a physical
            # failure and does not penalize normal clip completion/timeouts.
            termination = -50.0
            action_rate = -0.02  # actual sent target, rad
            feet_air_time = 0.0  # no universal gait duration for all motions


class G1DynamicsTrackerRobustCfg(G1DynamicsTrackerCfg):
    class control(G1DynamicsTrackerCfg.control):
        delay_max_steps = 1

    class domain_rand(G1DynamicsTrackerCfg.domain_rand):
        nominal_fraction = 0.3
        domain_rand_general = True
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        randomize_base_com = True
        added_com_range = [-0.02, 0.02]
        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.3


class G1DynamicsTrackerDirectCfg(G1DynamicsTrackerCfg):
    class control(G1DynamicsTrackerCfg.control):
        action_mode = "direct"


class G1DynamicsTrackerDirectRobustCfg(G1DynamicsTrackerRobustCfg):
    class control(G1DynamicsTrackerRobustCfg.control):
        action_mode = "direct"


class G1DynamicsTrackerHeightCfg(G1DynamicsTrackerCfg):
    class rewards(G1DynamicsTrackerCfg.rewards):
        class scales(G1DynamicsTrackerCfg.rewards.scales):
            root_height_tracking = 0.5


class G1DynamicsTrackerPreviewCfg(G1DynamicsTrackerCfg):
    class env(G1DynamicsTrackerCfg.env):
        # A control command computed at t is applied over [t, t+dt]. The old
        # TWIST policy uses this one-control-interval target as its first goal.
        tar_obs_steps = [1]


class G1DynamicsTrackerWideCfg(G1DynamicsTrackerCfg):
    class control(G1DynamicsTrackerCfg.control):
        target_scales = [0.6, 0.5, 0.8, 0.6, 0.8, 0.35] * 2 + [0.4, 0.3, 0.3] + [0.6] * 8


class G1DynamicsTrackerUniversalCfg(G1DynamicsTrackerWideCfg):
    """Nominal-physics training on the complete audited motion train split."""
    class env(G1DynamicsTrackerWideCfg.env):
        # The command selected at t is applied over [t, t+dt]. Matching the
        # original TWIST student/teacher at t+dt removes a one-step phase lag.
        tar_obs_steps = [1]

    class motion(G1DynamicsTrackerWideCfg.motion):
        motion_file = str(Path(__file__).resolve().parents[3] /
                          "motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml")
        # Reweight sampling toward clips with low completion while retaining a
        # non-zero probability for every clip. This is the curriculum used by
        # the original TWIST environment, now enabled for the universal tracker.
        motion_curriculum = True
        motion_curriculum_gamma = 0.01


class G1DynamicsTrackerUniversalRobustCfg(G1DynamicsTrackerRobustCfg):
    """Full-corpus system-identification/domain-randomization fine tuning."""
    class env(G1DynamicsTrackerRobustCfg.env):
        tar_obs_steps = [1]

    class control(G1DynamicsTrackerRobustCfg.control):
        target_scales = G1DynamicsTrackerWideCfg.control.target_scales

    class motion(G1DynamicsTrackerRobustCfg.motion):
        motion_file = str(Path(__file__).resolve().parents[3] /
                          "motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml")
        motion_curriculum = True
        motion_curriculum_gamma = 0.01


class G1DynamicsTrackerCfgPPO(BaseConfig):
    seed = 42
    class runner:
        runner_class_name = "DynamicsTrackerRunner"
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 100
        experiment_name = "g1_dynamics_tracker"
        run_name = "nominal"
        resume = False
        load_run = -1
        checkpoint = -1
        init_at_random_ep_len = False

    class policy:
        history_len = 79
        latent_dim = 128
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        world_model_hidden_dims = [256, 256]
        init_noise_std = 0.3
        use_dynamics_latent = False

    class algorithm:
        learning_rate = 3e-4
        world_model_learning_rate = 1e-4
        num_learning_epochs = 4
        num_mini_batches = 4
        gamma = 0.99
        lam = 0.95
        entropy_coef = 0.001
        value_loss_coef = 1.0
        max_grad_norm = 1.0
        schedule = "fixed"
        clip_param = 0.2
        world_model_sequence_length = 20
        world_model_num_epochs = 1
        world_model_batch_size = 64
        world_model_loss_coef = 1.0
        horizon_curriculum_updates = 200


class G1DynamicsTrackerRobustCfgPPO(G1DynamicsTrackerCfgPPO):
    class policy(G1DynamicsTrackerCfgPPO.policy):
        use_dynamics_latent = True
    class algorithm(G1DynamicsTrackerCfgPPO.algorithm):
        learning_rate = 1e-4
    class runner(G1DynamicsTrackerCfgPPO.runner):
        run_name = "robust"


class G1DynamicsTrackerAdaptiveCfgPPO(G1DynamicsTrackerCfgPPO):
    """Nominal physics with the already-pretrained history latent enabled."""
    class policy(G1DynamicsTrackerCfgPPO.policy):
        use_dynamics_latent = True
    class algorithm(G1DynamicsTrackerCfgPPO.algorithm):
        learning_rate = 1e-4
    class runner(G1DynamicsTrackerCfgPPO.runner):
        run_name = "nominal_adaptive"
