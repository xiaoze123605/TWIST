"""New task only: absolute PD commands, causal sensors and clean WM targets."""
import numpy as np
import torch
from torch.nn import functional as F
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from .g1_mimic_distill import G1MimicDistill
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
from rsl_rl.modules.dynamics_tracker import (
    STATE_DIM, CURRENT_DIM, HISTORY_FRAME_DIM, TargetTransform,
    append_history, reference_features,
)


class G1DynamicsTracker(G1MimicDistill):
    def __init__(self, cfg, *args, **kwargs):
        self.nominal_env_count = int(cfg.env.num_envs * cfg.domain_rand.nominal_fraction)
        super().__init__(cfg, *args, **kwargs)
        # Fresh training must not inherit TWIST student's 100000-update clock.
        self.total_env_steps_counter = 0
        self.global_counter = 0
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)
        self.compute_observations()

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        """Optionally randomize collection phases without changing evaluation RSI.

        DAgger collectors opt in by setting ``dagger_randomize_phase`` after
        environment construction. Normal training and fixed-phase evaluation
        continue through the inherited path unchanged.
        """
        if not getattr(self, "dagger_randomize_phase", False):
            return super()._reset_ref_motion(env_ids, motion_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(
                len(env_ids), motion_difficulty=self.motion_difficulty)
        lengths = self._motion_lib.get_motion_length(motion_ids)
        latest = (lengths - self._minimum_reference_remaining_time).clamp_min(0.)
        motion_times = torch.rand(len(env_ids), device=self.device) * latest
        had_ids = hasattr(self, "_eval_scenario_motion_ids")
        had_times = hasattr(self, "_eval_scenario_motion_times")
        old_ids = getattr(self, "_eval_scenario_motion_ids", None)
        old_times = getattr(self, "_eval_scenario_motion_times", None)
        fixed_ids = self._motion_ids.clone()
        fixed_times = self._motion_time_offsets.clone()
        fixed_ids[env_ids], fixed_times[env_ids] = motion_ids, motion_times
        self._eval_scenario_motion_ids, self._eval_scenario_motion_times = fixed_ids, fixed_times
        try:
            return super()._reset_ref_motion(env_ids, motion_ids)
        finally:
            if had_ids:
                self._eval_scenario_motion_ids = old_ids
            else:
                del self._eval_scenario_motion_ids
            if had_times:
                self._eval_scenario_motion_times = old_times
            else:
                del self._eval_scenario_motion_times

    def _init_buffers(self):
        self._minimum_reference_remaining_time = 22 * self.dt
        super()._init_buffers()
        self.motor_strength[:, :self.nominal_env_count] = 1.
        self.dynamics_history = torch.zeros(self.num_envs, self.cfg.env.dynamics_history_len,
                                            HISTORY_FRAME_DIM, device=self.device)
        self.previous_reference = torch.zeros(self.num_envs, 31, device=self.device)
        self.reference_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sent_target = self.dof_pos.clone()
        self.previous_target = self.dof_pos.clone()
        self.applied_target = self.dof_pos.clone()
        self.target_queue = self.dof_pos[:, None].repeat(1, self.cfg.control.delay_max_steps + 1, 1)
        self.command_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_transform = TargetTransform(
            self.dof_pos_limits[:, 0].cpu(), self.dof_pos_limits[:, 1].cpu(),
            self.cfg.control.target_scales, self.cfg.control.action_mode).to(self.device)
        self.sensor_state = torch.zeros(self.num_envs, STATE_DIM, device=self.device)
        self.current_reference = torch.zeros(self.num_envs, 31, device=self.device)
        self.current_task = torch.zeros(self.num_envs, 54, device=self.device)
        self.terminal_state = torch.zeros_like(self.sensor_state)
        self.terminal_critic = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)

    def _process_rigid_shape_props(self, props, env_id):
        if env_id == 0:
            self.nominal_shape_frictions = [p.friction for p in props]
        props = super()._process_rigid_shape_props(props, env_id)
        if env_id < self.nominal_env_count:
            for prop, friction in zip(props, self.nominal_shape_frictions):
                prop.friction = friction
            if self.cfg.domain_rand.randomize_friction:
                self.friction_coeffs[env_id] = self.nominal_shape_frictions[0]
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id < self.nominal_env_count:
            return props, np.zeros(4)
        return super()._process_rigid_body_props(props, env_id)

    def _push_robots(self):
        # Preserve nominal environments and add an impulse to current velocity.
        count = self.num_envs - self.nominal_env_count
        delta = (2. * torch.rand(count, 2, device=self.device) - 1.) * self.cfg.domain_rand.max_push_vel_xy
        self.root_states[self.nominal_env_count:, 7:9] += delta
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def clean_state(self):
        quat = self.root_states[:, 3:7]
        gyro = quat_rotate_inverse(quat, self.root_states[:, 10:13]) * 0.25
        gravity = F.normalize(quat_rotate_inverse(quat, self.gravity_vec), dim=-1)
        return torch.cat((gyro, gravity, self.dof_pos, self.dof_vel * 0.05), dim=-1)

    def _relative_reference(self, reference):
        reference = reference.clone()
        _, _, yaw = euler_from_quaternion(self.root_states[:, 3:7])
        reference[:, 3] = torch.atan2(torch.sin(reference[:, 3] - yaw), torch.cos(reference[:, 3] - yaw))
        return reference

    def _critic_observation(self, state, task):
        velocity = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.).float()
        return torch.cat((state, task, self.sent_target, velocity,
                          self.root_states[:, 2:3], contact), dim=-1)

    def compute_observations(self):
        clean = self.clean_state()
        sensor = clean.clone()
        if self.cfg.noise.add_noise:
            scales = sensor.new_tensor([0.025] * 3 + [0.02] * 3 + [0.01] * 23 + [0.005] * 23)
            sensor += (2. * torch.rand_like(sensor) - 1.) * scales
            sensor[:, 3:6] = F.normalize(sensor[:, 3:6], dim=-1)
        # Includes real ankle dq; nothing is sliced from the old masked obs.
        self.sensor_state.copy_(sensor)
        _, reference = self._get_mimic_obs()
        reference = self._relative_reference(self._randomize_mimic_obs(reference))
        self.current_task = reference_features(reference, self.previous_reference,
                                               self.reference_initialized, self.dt)
        self.current_reference.copy_(reference)
        self.previous_reference.copy_(reference)
        self.reference_initialized.fill_(True)
        current = torch.cat((sensor, self.current_task, self.sent_target), dim=-1)
        self.obs_buf = torch.cat((current, self.dynamics_history.flatten(1)), dim=-1)
        self.privileged_obs_buf = self._critic_observation(clean, self.current_task)

    def step(self, raw_sample):
        if raw_sample.shape != (self.num_envs, self.num_actions) or not torch.isfinite(raw_sample).all():
            raise ValueError("invalid pre-tanh policy sample")
        before = self.clean_state().clone()
        command = self.target_transform(raw_sample, self.current_reference[:, 8:31])
        self.previous_target.copy_(self.sent_target)
        self.sent_target = command.detach().clone()
        # Online and model rollouts share exactly this (s_t, sent_target_t) contract.
        self.dynamics_history = append_history(self.dynamics_history, self.sensor_state, self.sent_target)
        self.target_queue = torch.cat((self.target_queue[:, 1:], self.sent_target[:, None]), dim=1)
        ids = torch.arange(self.num_envs, device=self.device)
        self.applied_target = self.target_queue[ids, self.target_queue.shape[1] - 1 - self.command_delay]
        self.actions = raw_sample.detach().clone()  # legacy reward buffers only
        self.global_counter += 1
        self.total_env_steps_counter += 1
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # Capture sent command before reset_idx can replace its per-env rows.
        sent, applied = self.sent_target.clone(), self.applied_target.clone()
        self.post_physics_step()
        self.extras.update(dynamics_state=before, dynamics_next_state=self.terminal_state.clone(),
                           sent_target=sent, applied_target=applied,
                           terminal_critic_observation=self.terminal_critic.clone(),
                           time_outs=self.time_out_buf.clone())
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        kp, kd = self.p_gains, self.d_gains
        if self.cfg.domain_rand.randomize_motor:
            kp, kd = kp * self.motor_strength[0], kd * self.motor_strength[1]
        torque = kp * (self.applied_target - self.dof_pos) - kd * self.dof_vel
        return torch.maximum(torch.minimum(torque, self.torque_limits), -self.torque_limits)

    def _reward_action_rate(self):
        # Regularize actual sent PD targets, not unbounded/pre-tanh samples.
        return (self.sent_target - self.previous_target).square().sum(-1)

    def _reward_root_height_tracking(self):
        error = self.root_states[:, 2] - self._ref_root_pos[:, 2]
        return torch.exp(-100. * error.square())

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # Pushes may alter root velocity after the parent cached proprioception.
        self.base_lin_vel[:] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13])

    def check_termination(self):
        contact = torch.linalg.vector_norm(self.contact_forces[:, self.termination_contact_indices], dim=-1)
        contact_failure = torch.any(contact > 1., dim=-1)
        height_error = (self.root_states[:, 2] - self._ref_root_pos[:, 2]).abs()
        height_failure = height_error > self.cfg.rewards.root_height_diff_threshold
        tilt_failure = (self.roll.abs() > self.cfg.rewards.termination_roll) | (self.pitch.abs() > self.cfg.rewards.termination_pitch)
        speed_failure = torch.linalg.vector_norm(self.root_states[:, 7:10], dim=-1) > 5.
        failure = contact_failure | height_failure | tilt_failure | speed_failure
        pose_error = torch.zeros_like(height_error)
        pose_failure = torch.zeros_like(failure)
        if self._pose_termination:
            actual = self.rigid_body_states[:, self._key_body_ids, :3] - self.root_states[:, None, :3]
            target = self._ref_body_pos[:, self._key_body_ids] - self._ref_root_pos[:, None, :]
            if not self.global_obs:
                actual = convert_to_local_root_body_pos(self.root_states[:, 3:7], actual)
                target = convert_to_local_root_body_pos(self._ref_root_rot, target)
            pose_error = (target - actual).square().sum(-1).amax(-1).sqrt()
            pose_failure = pose_error > self._pose_termination_dist
            failure |= pose_failure
        motion_end = self._get_motion_times() >= self._motion_lib.get_motion_length(self._motion_ids)
        timeout = self.episode_length_buf >= self.max_episode_length
        # Finite reference clips are terminal, not an infinite-horizon timeout.
        self.time_out_buf = timeout & ~failure & ~motion_end
        self.reset_buf = failure | timeout | motion_end
        self._physical_failure = failure
        self.terminal_state.copy_(self.clean_state())
        _, next_reference = self._get_mimic_obs()
        task = reference_features(self._relative_reference(next_reference), self.previous_reference,
                                  self.reference_initialized, self.dt)
        self.terminal_critic.copy_(self._critic_observation(self.terminal_state, task))
        self.extras['tracking_joint_error'] = (self.dof_pos - self._ref_dof_pos).detach().clone()
        self.extras['tracking_joint_velocity_error'] = (self.dof_vel - self._ref_dof_vel).detach().clone()
        self.extras['sent_reference_offset'] = (self.sent_target - self._ref_dof_pos).detach().clone()
        self.extras['torque_limit_ratio'] = (self.torques.abs() / self.torque_limits).detach().clone()
        self.extras['tracking_height_error'] = (self.root_states[:,2] - self._ref_root_pos[:,2]).detach().clone()
        self.extras['torque_saturation_fraction'] = (self.torques.abs() >= self.torque_limits*.99).float().mean(-1)
        self.extras['physical_failure'] = failure.clone()
        self.extras['motion_completed'] = motion_end & ~failure
        self.extras['failure_contact'] = contact_failure
        self.extras['failure_height'] = height_failure
        self.extras['failure_tilt'] = tilt_failure
        self.extras['failure_speed'] = speed_failure
        self.extras['failure_pose'] = pose_failure
        self.extras['root_height_abs_error'] = height_error
        self.extras['max_keybody_error'] = pose_error
        self.extras['base_roll_pitch'] = torch.stack((self.roll, self.pitch), dim=-1).detach().clone()

    def _reward_termination(self):
        return self._physical_failure.float()

    def reset_idx(self, env_ids, motion_ids=None):
        """RSI without advancing EVERY simulator environment an extra substep."""
        if len(env_ids) == 0:
            return
        self.extras["episode"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["episode"]["metric_" + key] = sums[env_ids].mean()
            sums[env_ids] = 0.
        if self.cfg.motion.motion_curriculum:
            self._update_motion_difficulty(env_ids)
            self.mean_motion_difficulty = self.motion_difficulty.mean()
        self._reset_ref_motion(env_ids, motion_ids)
        self._reset_dofs(env_ids, self._ref_dof_pos, self._ref_dof_vel * 0.8)
        self._reset_root_states(env_ids, root_vel=self._ref_root_vel * 0.8,
                                root_quat=self._ref_root_rot, root_pos=self._ref_root_pos,
                                root_ang_vel=self._ref_root_ang_vel * 0.8)
        for name in ("last_actions", "last_dof_vel", "last_torques", "feet_air_time",
                     "obs_history_buf", "privileged_obs_history_buf", "contact_buf",
                     "action_history_buf", "feet_land_time", "deviate_tracking_frames",
                     "deviate_vel_tracking_frames", "contact_forces", "last_contacts"):
            getattr(self, name)[env_ids] = 0
        self._reset_buffers_extra(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True
        _, _, yaw = euler_from_quaternion(self.root_states[:, 3:7])
        self.init_yaw[env_ids] = yaw[env_ids]
        self.dynamics_history[env_ids] = 0
        self.reference_initialized[env_ids] = False
        self.previous_reference[env_ids] = 0
        for target in (self.sent_target, self.previous_target, self.applied_target):
            target[env_ids] = self.dof_pos[env_ids]
        self.target_queue[env_ids] = self.dof_pos[env_ids, None]
        self.command_delay[env_ids] = torch.randint(self.cfg.control.delay_max_steps + 1,
                                                   (len(env_ids),), device=self.device)
        self.command_delay[:self.nominal_env_count] = 0
        self.mimic_obs_delay_buf[env_ids] = 0
        self.mimic_obs_lpf_buf[env_ids] = 0

    def deployment_spec(self):
        return dict(version=1, policy_kind="dynamics_tracker", joint_names=list(self.dof_names),
                    history_len=self.cfg.env.dynamics_history_len, state_dim=STATE_DIM,
                    current_dim=CURRENT_DIM, history_frame_dim=HISTORY_FRAME_DIM,
                    observation_dim=self.num_obs, action_mode=self.cfg.control.action_mode,
                    target_scales=list(self.cfg.control.target_scales), control_dt=self.dt,
                    initial_target=self.default_dof_pos[0].cpu().tolist(),
                    joint_lower=self.target_transform.lower.cpu().tolist(),
                    joint_upper=self.target_transform.upper.cpu().tolist(),
                    kp=self.p_gains.cpu().tolist(), kd=self.d_gains.cpu().tolist(),
                    torque_limits=self.torque_limits.cpu().tolist(),
                    state_order="gyro*.25,gravity_unit,absolute_q,dq*.05",
                    history_order="oldest_to_newest:state_t,sent_target_t,valid",
                    reference_order="height,roll,pitch,relative_yaw,local_vxyz,local_wz,q_ref",
                    reference_velocity="backward_difference_clamp20_scaled.05",
                    reference_time_offset_steps=int(self.cfg.env.tar_obs_steps[0]),
                    padding="zeros_with_invalid_mask", action_filter="none")
