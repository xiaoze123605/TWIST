"""
Mixin for appending AnyAdapter history to TWIST observations.

Drop into:
    TWIST/legged_gym/legged_gym/envs/g1/anyadapter_history_mixin.py

Usage inside g1_mimic_distill.py:
    from .anyadapter_history_mixin import AnyAdapterHistoryMixin

    class G1MimicDistill(AnyAdapterHistoryMixin, ...):
        ...

Then call:
    self._init_anyadapter_history()   in _init_buffers or after obs_buf exists
    self.obs_buf = self._append_anyadapter_history(self.obs_buf, self.actions)
                                      at the end of compute_observations
    self._reset_anyadapter_history(env_ids) inside reset_idx

This mixin does not decide which observation entries are useful for dynamics;
you set cfg.env.anyadapter_state_indices.
"""

from __future__ import annotations

import torch


def _euler_from_quaternion(quaternion):
    """Local xyzw conversion keeps this lightweight mixin independently testable."""
    x, y, z, w = quaternion.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class AnyAdapterHistoryMixin:
    def _init_anyadapter_history(self):
        self.use_anyadapter = bool(getattr(self.cfg.env, "use_anyadapter", False))
        if not self.use_anyadapter:
            return
        self.anyadapter_history_len = int(getattr(self.cfg.env, "anyadapter_history_len", 20))
        state_indices = getattr(self.cfg.env, "anyadapter_state_indices", None)
        if state_indices is None:
            raise ValueError(
                "cfg.env.anyadapter_state_indices must be set before enabling AnyAdapter."
            )
        else:
            self.anyadapter_state_indices = torch.as_tensor(state_indices, device=self.device, dtype=torch.long)
        self.anyadapter_state_dim = int(self.anyadapter_state_indices.numel())
        self.anyadapter_frame_dim = self.anyadapter_state_dim + self.num_actions
        self.anyadapter_context_dim = int(
            getattr(self.cfg.env, "anyadapter_context_dim", 0)
        )
        self.anyadapter_fill_history_on_reset = bool(
            getattr(self.cfg.env, "anyadapter_fill_history_on_reset", False)
        )
        self.use_tracking_error_history = bool(
            getattr(self.cfg.env, "use_tracking_error_history", False)
        )
        self.tracking_error_history_len = int(
            getattr(self.cfg.env, "tracking_error_history_len", 0)
        )
        self.tracking_error_frame_dim = int(
            getattr(self.cfg.env, "tracking_error_frame_dim", 0)
        )
        if self.use_tracking_error_history:
            if self.tracking_error_history_len <= 0:
                raise ValueError("tracking_error_history_len must be positive")
            if self.tracking_error_frame_dim != 2 * self.num_actions + 7:
                raise ValueError(
                    "tracking_error_frame_dim must describe [q error, dq error, "
                    "root velocity/yaw error, roll/pitch error, phase lag]"
                )
        if self.anyadapter_context_dim not in (0, 2):
            raise ValueError(
                "anyadapter_context_dim currently supports 0 or 2 "
                "([sin(heading_error), 1-cos(heading_error)])."
            )
        expected_state_dim = getattr(self.cfg.env, "anyadapter_hist_state_dim", None)
        expected_frame_dim = getattr(self.cfg.env, "anyadapter_history_frame_dim", None)
        if expected_state_dim is not None and self.anyadapter_state_dim != int(expected_state_dim):
            raise ValueError(
                f"anyadapter_state_indices has dim {self.anyadapter_state_dim}, "
                f"expected {expected_state_dim}."
            )
        if expected_frame_dim is not None and self.anyadapter_frame_dim != int(expected_frame_dim):
            raise ValueError(
                f"AnyAdapter history frame dim is {self.anyadapter_frame_dim}, "
                f"expected {expected_frame_dim}."
            )
        self.anyadapter_history = torch.zeros(
            self.num_envs,
            self.anyadapter_history_len,
            self.anyadapter_frame_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.anyadapter_prev_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=torch.float32,
        )
        if self.use_tracking_error_history:
            self.tracking_error_history = torch.zeros(
                self.num_envs,
                self.tracking_error_history_len,
                self.tracking_error_frame_dim,
                device=self.device,
                dtype=torch.float32,
            )
        self.base_num_obs_before_anyadapter = self.num_obs
        self.num_obs = (
            self.base_num_obs_before_anyadapter
            + self.anyadapter_history_len * self.anyadapter_frame_dim
            + (
                self.tracking_error_history_len * self.tracking_error_frame_dim
                if self.use_tracking_error_history
                else 0
            )
            + self.anyadapter_context_dim
        )
        self.cfg.env.num_observations = self.num_obs
        if self.obs_buf.shape[1] != self.num_obs:
            self.obs_buf = torch.zeros(
                self.num_envs,
                self.num_obs,
                device=self.device,
                dtype=self.obs_buf.dtype,
            )

    def _reset_anyadapter_history(self, env_ids):
        if not getattr(self, "use_anyadapter", False):
            return
        self.anyadapter_history[env_ids] = 0.0
        self.anyadapter_prev_actions[env_ids] = 0.0
        if getattr(self, "use_tracking_error_history", False):
            self.tracking_error_history[env_ids] = 0.0

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _build_tracking_error_frame(self):
        """Build DTERA errors directly from named state/reference tensors."""
        q_error = self._ref_dof_pos - self.dof_pos
        dq_error = self._ref_dof_vel - self.dof_vel

        # Reference and actual velocities are both kept in the world frame.
        root_linear_error = self._ref_root_vel - self.root_states[:, 7:10]
        root_yaw_error = self._ref_root_ang_vel[:, 2:3] - self.root_states[:, 12:13]

        ref_roll, ref_pitch, _ = _euler_from_quaternion(self._ref_root_rot)
        roll_pitch_error = torch.stack(
            (
                self._wrap_to_pi(ref_roll - self.roll),
                self._wrap_to_pi(ref_pitch - self.pitch),
            ),
            dim=-1,
        )
        phase_lag = (
            torch.sum(q_error * self._ref_dof_vel, dim=-1, keepdim=True)
            / (torch.sum(self._ref_dof_vel.square(), dim=-1, keepdim=True) + 1e-6)
        ).clamp(-0.2, 0.2)
        frame = torch.cat(
            (
                q_error,
                dq_error,
                root_linear_error,
                root_yaw_error,
                roll_pitch_error,
                phase_lag,
            ),
            dim=-1,
        )
        if frame.shape[-1] != self.tracking_error_frame_dim:
            raise RuntimeError(
                f"tracking error frame has {frame.shape[-1]} dims, expected "
                f"{self.tracking_error_frame_dim}"
            )
        return frame

    def _anyadapter_context(self, base_obs: torch.Tensor) -> torch.Tensor:
        if self.anyadapter_context_dim == 0:
            return base_obs.new_zeros(self.num_envs, 0)
        ref_yaw = base_obs[:, 3]
        heading_error = torch.atan2(
            torch.sin(ref_yaw - self.yaw),
            torch.cos(ref_yaw - self.yaw),
        )
        return torch.stack(
            [torch.sin(heading_error), 1.0 - torch.cos(heading_error)],
            dim=-1,
        )

    def _append_anyadapter_history(self, base_obs: torch.Tensor, current_actions=None) -> torch.Tensor:
        """Append flattened history to base observations and update buffer.

        Call this after base_obs has been computed but before it is returned to
        the runner.  Each frame is [selected_state_from_base_obs, self.actions].
        """
        if not getattr(self, "use_anyadapter", False):
            return base_obs
        if current_actions is None:
            current_actions = getattr(self, "actions", self.anyadapter_prev_actions)
        dyn_state = base_obs.index_select(dim=1, index=self.anyadapter_state_indices)
        new_frame = torch.cat([dyn_state, current_actions.detach()], dim=-1)
        rolled_history = torch.roll(self.anyadapter_history, shifts=-1, dims=1)
        rolled_history[:, -1, :] = new_frame
        if self.anyadapter_fill_history_on_reset:
            reset_mask = (self.episode_length_buf <= 1).view(-1, 1, 1)
            reset_frame = torch.cat(
                [dyn_state, torch.zeros_like(current_actions)], dim=-1
            )
            initial_history = reset_frame.unsqueeze(1).expand(
                -1, self.anyadapter_history_len, -1
            )
            self.anyadapter_history = torch.where(
                reset_mask, initial_history, rolled_history
            )
        else:
            self.anyadapter_history = rolled_history
        self.anyadapter_prev_actions = current_actions.detach().clone()
        if self.anyadapter_fill_history_on_reset:
            reset_rows = (self.episode_length_buf <= 1).view(-1, 1)
            self.anyadapter_prev_actions = torch.where(
                reset_rows,
                torch.zeros_like(self.anyadapter_prev_actions),
                self.anyadapter_prev_actions,
            )
        parts = [base_obs, self.anyadapter_history.reshape(self.num_envs, -1)]
        if self.use_tracking_error_history:
            error_frame = self._build_tracking_error_frame()
            rolled_error_history = torch.roll(
                self.tracking_error_history, shifts=-1, dims=1
            )
            rolled_error_history[:, -1, :] = error_frame
            if self.anyadapter_fill_history_on_reset:
                reset_mask = (self.episode_length_buf <= 1).view(-1, 1, 1)
                repeated_error = error_frame.unsqueeze(1).expand(
                    -1, self.tracking_error_history_len, -1
                )
                self.tracking_error_history = torch.where(
                    reset_mask, repeated_error, rolled_error_history
                )
            else:
                self.tracking_error_history = rolled_error_history
            parts.append(self.tracking_error_history.reshape(self.num_envs, -1))
        parts.append(self._anyadapter_context(base_obs))
        return torch.cat(parts, dim=-1)
