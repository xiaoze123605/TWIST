"""Deployment contract without Isaac Gym or privileged observations.

Propose a target, send it through the external controller, then commit the
target actually sent. Unobserved actuator delay stays inside the plant.
"""
import json

import torch
from torch import nn
from .dynamics_tracker import TargetTransform, append_history, reference_features


def measured_state_and_reference(q, dq, gyro_body, quaternion_wxyz, reference):
    """Batched body IMU + joint sensors; reference heading is world-frame yaw.

    Reference linear/angular velocities are expressed in the reference body's
    local frame, as in the motion library. No actual base velocity is needed.
    """
    if (q.ndim != 2 or q.shape[1] != 23 or dq.shape != q.shape or
            gyro_body.shape != (q.shape[0], 3) or quaternion_wxyz.shape != (q.shape[0], 4) or
            reference.shape != (q.shape[0], 31)):
        raise ValueError('invalid measured state dimensions')
    if not all(torch.isfinite(v).all() for v in (q, dq, gyro_body, quaternion_wxyz, reference)):
        raise ValueError('non-finite sensor or reference')
    norms = quaternion_wxyz.norm(dim=-1, keepdim=True)
    if torch.any(norms < 1e-6):
        raise ValueError('invalid IMU quaternion')
    w, x, y, z = (quaternion_wxyz / norms).unbind(-1)
    gravity = torch.stack((2*(w*y-x*z), -2*(w*x+y*z), 2*(x*x+y*y)-1), -1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    relative = reference.clone()
    relative[:, 3] = torch.atan2(torch.sin(reference[:,3]-yaw), torch.cos(reference[:,3]-yaw))
    return torch.cat((gyro_body*.25, gravity, q, dq*.05), -1), relative


class DeploymentPolicy(nn.Module):
    def __init__(self, actor_critic, spec):
        super().__init__()
        self.encoder = actor_critic.history_encoder
        self.actor = actor_critic.actor
        self.history_len = int(spec['history_len'])
        self.use_latent = bool(spec['use_dynamics_latent'])
        self.transform = TargetTransform(spec['joint_lower'], spec['joint_upper'],
                                         spec['target_scales'], spec['action_mode'])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[1] != 129 + self.history_len * 76:
            raise ValueError('wrong deployment observation shape')
        current = obs[:, :129]
        latent = self.encoder(obs[:, 129:].reshape(-1, self.history_len, 76))
        if not self.use_latent:
            latent = torch.zeros_like(latent)
        state, task = current[:, :52], current[:, 52:106]
        errors = torch.cat((task[:, 8:31] - state[:, 6:29], task[:, 31:] - state[:, 29:52]), -1)
        raw = self.actor(torch.cat((current, errors, latent), -1))
        return self.transform(raw, task[:, 8:31])


class DynamicsTrackerRuntime:
    def __init__(self, path, joint_names, device='cpu'):
        extra = {'deployment.json': ''}
        self.policy = torch.jit.load(str(path), map_location=device, _extra_files=extra).eval()
        self.spec = json.loads(extra['deployment.json'])
        if self.spec['version'] != 1 or self.spec['policy_kind'] != 'dynamics_tracker':
            raise ValueError('unsupported deployment contract')
        if list(joint_names) != self.spec['joint_names']:
            raise ValueError('joint order mismatch')
        self.device = device
        self.lower = torch.tensor(self.spec['joint_lower'], device=device)
        self.upper = torch.tensor(self.spec['joint_upper'], device=device)
        self.ready = False
        self.pending = None

    def _tensor(self, value, width):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(1, -1)
        if value.shape != (1, width) or not torch.isfinite(value).all():
            raise ValueError('invalid runtime input')
        return value

    def reset(self, initial_target):
        self.previous_target = self._tensor(initial_target, 23).clone()
        self.history = torch.zeros(1, self.spec['history_len'], 76, device=self.device)
        self.previous_reference = torch.zeros(1, 31, device=self.device)
        self.initialized = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.pending = None
        self.ready = True

    @torch.no_grad()
    def propose(self, sensor_state, reference):
        """reference already uses relative yaw and the declared local velocity frame."""
        if not self.ready or self.pending is not None:
            raise RuntimeError('reset first; commit each proposal before the next sample')
        state, reference = self._tensor(sensor_state, 52), self._tensor(reference, 31)
        task = reference_features(reference, self.previous_reference, self.initialized,
                                  self.spec['control_dt'])
        obs = torch.cat((state, task, self.previous_target, self.history.flatten(1)), -1)
        target = self.policy(obs)
        if not torch.isfinite(target).all():
            raise FloatingPointError('non-finite deployment target')
        self.pending = (state.clone(), reference.clone())
        return target[0].clone()

    @torch.no_grad()
    def commit(self, sent_target):
        if self.pending is None:
            raise RuntimeError('no pending proposal')
        sent = self._tensor(sent_target, 23)
        if torch.any(sent < self.lower - 1e-6) or torch.any(sent > self.upper + 1e-6):
            raise ValueError('sent target outside exported joint limits')
        state, reference = self.pending
        self.history = append_history(self.history, state, sent)
        self.previous_target = sent.clone()
        self.previous_reference = reference
        self.initialized.fill_(True)
        self.pending = None
