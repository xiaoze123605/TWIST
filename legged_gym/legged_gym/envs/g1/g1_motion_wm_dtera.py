"""DTERA training with frozen Motion-WM student reference preprocessing."""
from pathlib import Path
import sys
import torch
from .g1_mimic_distill import G1MimicDistill


class G1MotionWMDTERA(G1MimicDistill):
    def _init_buffers(self):
        # The reference GRU needs 25 observations. Preserve one extra control
        # step so RSI cannot create episodes that terminate during warmup.
        self._minimum_reference_remaining_time = 26 * self.dt
        super()._init_buffers()
        root = str(Path(__file__).resolve().parents[4])
        if root not in sys.path:
            sys.path.insert(0, root)
        from motion_world_model.training_reference import TrainingReferencePipeline
        options = self.cfg.motion_wm
        self.motion_reference_pipeline = TrainingReferencePipeline(
            options.checkpoint, self.num_envs, device=self.device, seed=self.cfg.seed,
            mode_probabilities=options.mode_probabilities, preset=options.corruption_preset,
            control_dt=self.dt,
            artificial_corruption=getattr(options, 'artificial_corruption', True))
        print("[Motion-WM+DTERA] Frozen Motion GRU; 50 Hz; 3695-D dual history; clean reward targets; "
              f"artificial corruption={getattr(options, 'artificial_corruption', True)}; "
              f"RSI tail >= {self._minimum_reference_remaining_time:.2f}s")

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

    def check_termination(self):
        super().check_termination()
        # Preserve the reset cause before adding the per-motion boundary.  A
        # low aggregate episode length is otherwise ambiguous: short clips
        # ending normally and genuine falls are reported by the runner as the
        # same reset.  time_out_buf is excluded from catastrophic failures.
        catastrophic = self.reset_buf.bool() & ~self.time_out_buf.bool()
        # RSI starts part-way through a motion. End before its next reference
        # wraps to frame zero; do not train on a discontinuous motion splice.
        lengths = self._motion_lib.get_motion_length(self._motion_ids)
        boundary = self.episode_length_buf * self.dt + self._motion_time_offsets >= lengths - self.dt
        self.time_out_buf |= boundary & ~self.reset_buf
        self.reset_buf |= boundary
        self._last_catastrophic_termination = catastrophic
        self._last_motion_end_termination = boundary & ~catastrophic


class G1MotionWMDTERAV2(G1MotionWMDTERA):
    """V2 adds mode-specific, sample-weighted diagnostics against clean targets."""

    def _init_buffers(self):
        super()._init_buffers()
        # Steps, reward sum, joint squared error, completed episodes, length sum.
        self._mode_metrics = torch.zeros(3, 5, device=self.device)
        self._mode_episode_steps = torch.zeros(self.num_envs, device=self.device)
        print('[Motion-WM+DTERA v2] Independent gates; branch gains=0.5/0.25; '
              'clean joint/root-pose reward weights=0.8/0.8')

    def compute_reward(self):
        super().compute_reward()
        if not hasattr(self, '_mode_metrics'):
            return
        joint_mse = (self.dof_pos - self._ref_dof_pos).square().mean(dim=-1)
        self._mode_episode_steps += 1
        for index in range(3):
            mask = self.motion_reference_pipeline.mode == index
            self._mode_metrics[index, 0] += mask.sum()
            self._mode_metrics[index, 1] += self.rew_buf[mask].sum()
            self._mode_metrics[index, 2] += joint_mse[mask].sum()

    def reset_idx(self, env_ids, motion_ids=None):
        if hasattr(self, '_mode_metrics'):
            for index in range(3):
                ids = env_ids[self.motion_reference_pipeline.mode[env_ids] == index]
                lengths = self._mode_episode_steps[ids]
                self._mode_metrics[index, 3] += (lengths > 0).sum()
                self._mode_metrics[index, 4] += lengths.sum()
            self._mode_episode_steps[env_ids] = 0
        super().reset_idx(env_ids, motion_ids=motion_ids)

    def pop_training_metrics(self):
        values = self._mode_metrics.detach().cpu().tolist()
        self._mode_metrics.zero_()
        result = {}
        for name, (steps, reward, squared_error, episodes, length) in zip(
                self.motion_reference_pipeline.MODES, values):
            prefix = 'MotionReference/' + name + '/'
            result[prefix + 'steps'] = steps
            result[prefix + 'completed_episodes'] = episodes
            if steps:
                result[prefix + 'reward_per_step'] = reward / steps
                result[prefix + 'clean_joint_rmse'] = (squared_error / steps)**0.5
            if episodes:
                result[prefix + 'episode_length_steps'] = length / episodes
        return result


class G1MotionWMDTERALegs(G1MotionWMDTERAV2):
    """Separate bilateral leg objectives against the uncorrupted motion."""

    def _init_buffers(self):
        super()._init_buffers()
        groups = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
        names = [[side + '_' + group + '_joint' for side in ('left', 'right')]
                 for group in groups]
        missing = [name for pair in names for name in pair if name not in self.dof_names]
        if missing:
            raise ValueError('Missing lower-body joints: ' + str(missing))
        self._leg_ids = torch.tensor([[self.dof_names.index(n) for n in pair]
                                      for pair in names], device=self.device)

    def _reward_tracking_leg_position(self):
        error = self.dof_pos[:, self._leg_ids] - self._ref_dof_pos[:, self._leg_ids]
        # Reward each joint separately: one bad knee cannot saturate the
        # exponential for every other joint. Identical weights on both sides.
        sigma = self.cfg.rewards.leg_position_sigma
        return torch.exp(-error.square() / sigma**2).mean(dim=(1, 2))

    def _reward_tracking_leg_velocity(self):
        error = self.dof_vel[:, self._leg_ids] - self._ref_dof_vel[:, self._leg_ids]
        sigma = self.cfg.rewards.leg_velocity_sigma
        return torch.exp(-error.square() / sigma**2).mean(dim=(1, 2))


class G1MotionWMDTERADeployV3(G1MotionWMDTERALegs):
    """Deployment-aligned raw-PKL -> Motion-WM -> higher-budget DTERA task."""

    def _init_buffers(self):
        super()._init_buffers()
        # catastrophic resets, successful motion ends, environment steps
        self._termination_metrics = torch.zeros(3, device=self.device)
        print('[Motion-WM+DTERA deploy v3] raw PKL -> frozen Motion-WM; '
              'full-motion starts; residual bound=0.09 action')

    def check_termination(self):
        super().check_termination()
        self._termination_metrics[0] += self._last_catastrophic_termination.sum()
        self._termination_metrics[1] += self._last_motion_end_termination.sum()
        self._termination_metrics[2] += self.num_envs

    def pop_training_metrics(self):
        result = super().pop_training_metrics()
        catastrophic, motion_end, env_steps = self._termination_metrics.detach().cpu().tolist()
        reset_count = catastrophic + motion_end
        result['Termination/catastrophic_count'] = catastrophic
        result['Termination/motion_end_count'] = motion_end
        result['Termination/catastrophic_fraction_per_step'] = (
            catastrophic / env_steps if env_steps else 0.0)
        result['Termination/motion_end_fraction_per_step'] = (
            motion_end / env_steps if env_steps else 0.0)
        result['Termination/motion_end_share'] = (
            motion_end / reset_count if reset_count else 0.0)
        self._termination_metrics.zero_()
        return result


class G1MotionWMDTERADeployV4(G1MotionWMDTERADeployV3):
    """Harm-aware deployment task after the held-out V3-5000 audit."""

    def _init_buffers(self):
        super()._init_buffers()
        print('[Motion-WM+DTERA deploy v4] feedback line-search branch scaling; '
              'residual candidates=0/25/50/75/100%; '
              'bound=0.055 action before joint-group scaling')
