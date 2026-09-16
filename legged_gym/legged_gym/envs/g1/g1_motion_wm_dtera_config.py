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

    class domain_rand(G1MimicStuAnyAdapterDTERACfg.domain_rand):
        # New task applies exactly one corruption pipeline in its override.
        randomize_mimic_obs = False


class G1MotionWMDTERACfgPPO(G1MimicStuAnyAdapterDTERACfgPPO):
    class runner(G1MimicStuAnyAdapterDTERACfgPPO.runner):
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
