"""Versioned physical dynamics path, independent of the legacy TWIST contract.

State: local angular velocity, roll/pitch, absolute joint q, true joint dq.
Action: final joint PD target in radians, after delay, clipping and projection.
Never feed these states to a legacy 7001/7003-D actor or overwrite PPO actions.
"""
import torch
from torch import nn

STATE_DIM = 51
ACTION_DIM = 23
HISTORY_LEN = 79
CONTRACT = 'physical_pd_target_v1'


def physical_state(angular_velocity, roll_pitch, joint_position, joint_velocity):
    parts = [angular_velocity, roll_pitch, joint_position, joint_velocity]
    expected = [3, 2, 23, 23]
    if any(x.shape[-1] != n for x, n in zip(parts, expected)):
        raise ValueError('physical state requires dimensions 3+2+23+23')
    state = torch.cat(parts, -1)
    if not torch.isfinite(state).all():
        raise ValueError('physical state must be finite')
    return state


def validate_transition(packet):
    for key, size in [('state',51),('next_state',51),('commanded_action',23),
                      ('executed_action',23),('executed_pd_target',23)]:
        value = packet[key]
        if value.ndim != 2 or value.shape[-1] != size or not torch.isfinite(value).all():
            raise ValueError('invalid physical transition field '+key)
    batches = {packet[k].shape[0] for k in ['state','next_state','commanded_action',
                                           'executed_action','executed_pd_target','done']}
    if len(batches) != 1 or packet['done'].ndim != 1:
        raise ValueError('physical transition batches do not align')
    if packet.get('contract') != CONTRACT:
        raise ValueError('physical transition contract mismatch')


class PhysicalWorldModel(nn.Module):
    """Normalized residual forward model with its own, trainable history encoder.

Predict q and endpoint dq independently. Endpoint velocity is not interval
average velocity, so do not force q_next=q+dt*dq_next in a contact transition.
"""
    def __init__(self, state_mean, state_scale, target_mean, target_scale, latent_dim=32):
        super().__init__()
        for name, value, size in [('state_mean',state_mean,51),('state_scale',state_scale,51),
                                  ('target_mean',target_mean,23),('target_scale',target_scale,23)]:
            value = torch.as_tensor(value,dtype=torch.float32).clone()
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('invalid normalizer '+name)
            if name.endswith('scale') and bool((value <= 0).any()):
                raise ValueError('normalizer scales must be positive')
            self.register_buffer(name,value)
        self.history_encoder = nn.Sequential(nn.Conv1d(74,64,9,stride=5),nn.SiLU(),
                                              nn.Conv1d(64,64,6,stride=3),nn.SiLU(),
                                              nn.Flatten(),nn.Linear(256,latent_dim),nn.SiLU())
        self.transition = nn.Sequential(nn.Linear(51+23+latent_dim,128),nn.SiLU(),
                                        nn.Linear(128,128),nn.SiLU(),nn.Linear(128,51))
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def normalize_state(self, state):
        return (state-self.state_mean)/self.state_scale

    def normalize_target(self, target):
        return (target-self.target_mean)/self.target_scale

    def forward(self, state, executed_pd_target, history):
        if state.shape[-1] != 51 or executed_pd_target.shape[-1] != 23 or history.shape[1:] != (79,74):
            raise ValueError('expected physical state51, PD target23 and history79x74')
        normalized_history=torch.cat((self.normalize_state(history[:,:,:51]),
                                       self.normalize_target(history[:,:,51:])), -1)
        latent=self.history_encoder(normalized_history.transpose(1,2))
        current=self.normalize_state(state)
        delta=self.transition(torch.cat((current,self.normalize_target(executed_pd_target),latent),-1))
        return state+delta*self.state_scale

    def loss(self, prediction, target):
        residual=(prediction-target)/self.state_scale
        residual=torch.cat((residual[:,:3],
            torch.atan2(torch.sin(prediction[:,3:5]-target[:,3:5]),
                        torch.cos(prediction[:,3:5]-target[:,3:5]))/self.state_scale[3:5],residual[:,5:]),-1)
        return residual.abs().mean()


def fit_normalizers(states, pd_targets):
    """Fit on training trajectories only; physical floors bound constant axes."""
    floors=states.new_tensor([.05]*3+[.01]*2+[.03]*23+[.3]*23)
    return dict(state_mean=states.mean(0),state_scale=torch.maximum(states.std(0),floors),
                target_mean=pd_targets.mean(0),target_scale=pd_targets.std(0).clamp_min(.03))
