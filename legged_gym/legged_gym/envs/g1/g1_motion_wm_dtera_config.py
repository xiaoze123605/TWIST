from pathlib import Path

from .g1_mimic_distill_anyadapter_config import (
    G1MimicStuAnyAdapterDTERACfg, G1MimicStuAnyAdapterDTERACfgPPO,
)


class G1MotionWMDTERACfg(G1MimicStuAnyAdapterDTERACfg):
    class env(G1MimicStuAnyAdapterDTERACfg.env):
        num_envs = 64

    class motion_wm:
        checkpoint = str(Path(__file__).resolve().parents[3] /
                         'logs/motion_world_model/full_stable_v2/best.pt')
        # Episode-level mixture, chosen as a neutral first-pilot baseline.
        mode_probabilities = (1/3, 1/3, 1/3)
        corruption_preset = 'formal'
        artificial_corruption = True

    class domain_rand(G1MimicStuAnyAdapterDTERACfg.domain_rand):
        # New task applies exactly one corruption pipeline in its override.
        randomize_mimic_obs = False


class G1MotionWMDTERACfgPPO(G1MimicStuAnyAdapterDTERACfgPPO):
    class runner(G1MimicStuAnyAdapterDTERACfgPPO.runner):
        # RSI already samples motion time during reset. Changing only the
        # episode counter afterwards jumps the reference away from that pose.
        init_at_random_ep_len = False
        max_iterations = 100
        save_interval = 50
        experiment_name = 'g1_motion_wm_dtera'


class G1MotionWMDTERAV2Cfg(G1MotionWMDTERACfg):
    class rewards(G1MotionWMDTERACfg.rewards):
        class scales(G1MotionWMDTERACfg.rewards.scales):
            tracking_joint_dof = 0.8
            tracking_root_pose = 0.8


class G1MotionWMDTERAV2CfgPPO(G1MotionWMDTERACfgPPO):
    class runner(G1MotionWMDTERACfgPPO.runner):
        experiment_name = 'g1_motion_wm_dtera_v2'
        max_iterations = 1500
        resume = False

    class policy(G1MotionWMDTERACfgPPO.policy):
        use_independent_branch_gates = True
        tracking_demand_mode = 'smoothstep'
        tracking_demand_low = 0.30
        tracking_demand_high = 0.80
        dynamics_demand_low = 0.10
        dynamics_demand_high = 0.50
        # Gains alone set the conservative budget; do not halve it again
        # with the selective task's dynamics_gate_scale=0.5.
        dynamics_gate_scale = 1.0
        tracking_gate_scale = 1.0
        dynamics_branch_gain = 0.5
        tracking_branch_gain = 0.25
        adapter_gain = 1.0


class G1MotionWMDTERALegsCfg(G1MotionWMDTERAV2Cfg):
    class rewards(G1MotionWMDTERAV2Cfg.rewards):
        leg_position_sigma = 0.20  # radians; experimental, not fitted per motion
        leg_velocity_sigma = 2.0   # radians/second

        class scales(G1MotionWMDTERAV2Cfg.rewards.scales):
            tracking_leg_position = 0.4
            tracking_leg_velocity = 0.1


class G1MotionWMDTERALegsCfgPPO(G1MotionWMDTERAV2CfgPPO):
    class runner(G1MotionWMDTERAV2CfgPPO.runner):
        experiment_name = 'g1_motion_wm_dtera_legs'


class G1MotionWMDTERADeployV3Cfg(G1MotionWMDTERALegsCfg):
    class env(G1MotionWMDTERALegsCfg.env):
        # Deployment starts every supplied PKL at frame zero.  Combining RSI
        # with end-of-motion termination trained on only half a motion on
        # average and explains the observed episode-length regression.
        rand_reset = False

    class motion_wm(G1MotionWMDTERALegsCfg.motion_wm):
        # Exactly match deployment: the supplied PKL is sent directly to the
        # frozen reference GRU.  Synthetic corruption remains available only
        # to the legacy tasks above for reproducibility.
        mode_probabilities = (0.0, 0.0, 1.0)
        artificial_corruption = False

    class rewards(G1MotionWMDTERALegsCfg.rewards):
        class scales(G1MotionWMDTERALegsCfg.rewards.scales):
            tracking_joint_dof = 0.9
            tracking_root_pose = 0.9
            tracking_leg_position = 0.6
            tracking_leg_velocity = 0.15
            feet_slip = -0.2


class G1MotionWMDTERADeployV3CfgPPO(G1MotionWMDTERALegsCfgPPO):
    class runner(G1MotionWMDTERALegsCfgPPO.runner):
        experiment_name = 'g1_motion_wm_dtera_deploy_v3'
        max_iterations = 3000
        resume = False

    class policy(G1MotionWMDTERALegsCfgPPO.policy):
        # Old bound: .5*.03 + .25*.03 = .0225 action.  New bound:
        # 1.0*.06 + .5*.06 = .09 action (4x), still tanh-bounded.
        # A 64-env preflight showed that an equally weighted .08 tracking
        # branch saturated in 30 updates and suppressed the dynamics branch.
        dynamics_action_delta_scale = 0.06
        tracking_action_delta_scale = 0.06
        dynamics_branch_gain = 1.0
        tracking_branch_gain = 0.5
        tracking_history_policy_grad_scale = 0.25
        residual_warmup_iterations = 500

        # Open both gates earlier.  The old tracking gate averaged only 0.36
        # and drove the already-small residual toward zero late in training.
        tracking_demand_low = 0.15
        tracking_demand_high = 0.60
        dynamics_demand_low = 0.05
        dynamics_demand_high = 0.35

    class algorithm(G1MotionWMDTERALegsCfgPPO.algorithm):
        # Permit useful corrections while retaining explicit magnitude and
        # saturation penalties.  Auxiliary losses plateaued in the old run.
        adapter_reg_coef = 0.005
        adapter_reg_initial_coef = 0.10
        adapter_reg_anneal_iterations = 1000
        residual_saturation_reg_coef = 0.10
        world_model_loss_coef = 0.15
        error_prediction_loss_coef = 0.10


class G1MotionWMDTERADeployV4Cfg(G1MotionWMDTERADeployV3Cfg):
    """Deployment-aligned environment; only the DTERA policy changes."""


class G1MotionWMDTERADeployV4CfgPPO(G1MotionWMDTERADeployV3CfgPPO):
    class runner(G1MotionWMDTERADeployV3CfgPPO.runner):
        experiment_name = 'g1_motion_wm_dtera_deploy_v4'
        max_iterations = 3000
        resume = False

    class policy(G1MotionWMDTERADeployV3CfgPPO.policy):
        # V3-5000 was stable, but its dynamics gate was >.95 on 98.95% of a
        # held-out deployment motion. Its .09 residual erased WM joint/height
        # gains. V4 remains 2.44x larger than the old .0225 budget, but no
        # longer permits a nearly unconditional large correction.
        dynamics_action_delta_scale = 0.05
        tracking_action_delta_scale = 0.05
        dynamics_branch_gain = 0.75
        tracking_branch_gain = 0.35
        tracking_demand_low = 0.30
        tracking_demand_high = 0.90
        dynamics_demand_low = 0.20
        dynamics_demand_high = 0.90

        # Use transition feedback learned by the action-conditioned error
        # trend model. For each branch, evaluate 0/25/50/75/100 percent of its
        # residual and softly select the scale with the lowest relative
        # next-frame tracking error. This replaces the absolute 0.02
        # threshold, which was three orders above the observed improvements.
        use_predicted_improvement_gate = True
        predicted_improvement_mode = 'feedback_line_search'
        predicted_improvement_low = 0.0
        predicted_improvement_high = 0.02
        feedback_residual_scales = (0.0, 0.25, 0.50, 0.75, 1.0)
        feedback_temperature = 0.01
        predicted_improvement_gate_start_iteration = 500
        predicted_improvement_gate_ramp_iterations = 500

        # V3 trace attribution showed the largest harmful deltas in shoulders
        # and elbows. Keep full authority on legs, 75% on waist and 50% on
        # arms while the predictor is learned from scratch.
        residual_joint_scales = (
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.75, 0.75, 0.75,
            0.50, 0.50, 0.50, 0.50,
            0.50, 0.50, 0.50, 0.50,
        )

    class algorithm(G1MotionWMDTERADeployV3CfgPPO.algorithm):
        # The predictor now controls the action path, so train it more strongly.
        error_prediction_loss_coef = 0.20
        adapter_reg_coef = 0.01
        adapter_reg_initial_coef = 0.10
        adapter_reg_anneal_iterations = 1000
        residual_saturation_reg_coef = 0.15
