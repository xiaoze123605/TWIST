"""Motion-WM reference preprocessing plus paper-style dynamics AnyAdapter."""

from pathlib import Path
import sys

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .g1_mimic_distill import G1MimicDistill
from legged_gym.envs.base.legged_robot import euler_from_quaternion


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
            if getattr(self.cfg.motion_wm, "prefill_history", False):
                self._prefill_motion_wm_history(env_ids)

    def _prefill_motion_wm_history(self, env_ids):
        """Feed clean t-24..t reference frames into the pipeline at reset so
        wm-mode envs are GRU-ready from frame 0 (no frame-25 output jump)."""
        n = len(env_ids)
        offsets = torch.arange(-24, 1, device=self.device, dtype=torch.float) * self.dt
        times = self._get_motion_times(env_ids).unsqueeze(-1) + offsets
        times = times.clamp(min=0.0).flatten()
        ids = self._motion_ids[env_ids].unsqueeze(-1).expand(n, 25).flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = \
            self._motion_lib.calc_motion_frame(ids, times)
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(-1, 1); pitch = pitch.reshape(-1, 1); yaw = yaw.reshape(-1, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        frames = torch.cat((
            root_pos[..., 2:3],  # 1
            roll, pitch, yaw,  # 3
            root_vel,  # 3
            root_ang_vel[..., 2:3],  # 1
            dof_pos,  # num_dof
        ), dim=-1)  # (n*25, 31)
        self.motion_reference_pipeline.prefill(env_ids, frames.reshape(n, 25, 31))
