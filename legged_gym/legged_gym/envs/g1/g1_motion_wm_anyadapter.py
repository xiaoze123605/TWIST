"""Motion-WM reference preprocessing plus paper-style dynamics AnyAdapter."""

from pathlib import Path
import sys

from .g1_mimic_distill import G1MimicDistill


class G1MotionWMAnyAdapter(G1MimicDistill):
    """Keeps clean targets while supplying clean/corrupt/WM processed references."""

    def _init_buffers(self):
        self._minimum_reference_remaining_time = 26 * self.dt
        super()._init_buffers()
        root = str(Path(__file__).resolve().parents[4])
        if root not in sys.path:
            sys.path.insert(0, root)
        from motion_world_model.training_reference import TrainingReferencePipeline
        options = self.cfg.motion_wm
        self.motion_reference_pipeline = TrainingReferencePipeline(
            options.checkpoint, self.num_envs, device=self.device, seed=self.cfg.seed,
            mode_probabilities=options.mode_probabilities,
            preset=options.corruption_preset, control_dt=self.dt,
            artificial_corruption=options.artificial_corruption,
        )
        print("[Motion-WM+AnyAdapter-OpenTrack] frozen reference WM; 7001-D "
              "dynamics-only history; clean reward targets")

    def _randomize_mimic_obs(self, mimic_obs):
        return self.motion_reference_pipeline.process(mimic_obs)

    def reset_idx(self, env_ids, motion_ids=None):
        super().reset_idx(env_ids, motion_ids=motion_ids)
        if hasattr(self, "motion_reference_pipeline"):
            self.motion_reference_pipeline.reset(env_ids)
            self.obs_history_buf[env_ids] = 0
            self.privileged_obs_history_buf[env_ids] = 0
