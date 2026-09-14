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
