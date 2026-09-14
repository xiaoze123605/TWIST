"""DTERA training with frozen Motion-WM student reference preprocessing."""
from pathlib import Path
import sys
from .g1_mimic_distill import G1MimicDistill


class G1MotionWMDTERA(G1MimicDistill):
    def _init_buffers(self):
        super()._init_buffers()
        root = str(Path(__file__).resolve().parents[4])
        if root not in sys.path:
            sys.path.insert(0, root)
        from motion_world_model.training_reference import TrainingReferencePipeline
        options = self.cfg.motion_wm
        self.motion_reference_pipeline = TrainingReferencePipeline(
            options.checkpoint, self.num_envs, device=self.device, seed=self.cfg.seed,
            mode_probabilities=options.mode_probabilities, preset=options.corruption_preset,
            control_dt=self.dt)
        print("[Motion-WM+DTERA] Frozen Motion GRU; 50 Hz; 3695-D dual history; clean reward targets")

    def _randomize_mimic_obs(self, mimic_obs):
        # Replace, rather than stack on top of, the old student corruption.
        # _get_mimic_obs's privileged reference and reward _ref_* tensors stay clean.
        return self.motion_reference_pipeline.process(mimic_obs)

    def reset_idx(self, env_ids, motion_ids=None):
        super().reset_idx(env_ids, motion_ids=motion_ids)
        if hasattr(self, "motion_reference_pipeline"):
            self.motion_reference_pipeline.reset(env_ids)
            self.obs_history_buf[env_ids] = 0
            self.privileged_obs_history_buf[env_ids] = 0
