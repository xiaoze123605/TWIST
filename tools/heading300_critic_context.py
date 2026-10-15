"""Training-only heading context; deployment actor and its weights are unchanged."""
import torch
from torch import nn
from heading23500_actor import TwistHeadingSupportActorCritic


class TwistHeadingCriticContextActorCritic(TwistHeadingSupportActorCritic):
    def __init__(self, *args, num_critic_observations=1320, **kwargs):
        if num_critic_observations != 1320:
            raise ValueError('critic context requires 1318 original plus two heading channels')
        super().__init__(*args, num_critic_observations=1318, **kwargs)
        old = self.critic[0]
        # Preserve the original constructor RNG consumption for matched experiments.
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(1320, old.out_features)
        with torch.no_grad():
            expanded.weight[:, :1318].copy_(old.weight)
            expanded.weight[:, 1318:].zero_()
            expanded.bias.copy_(old.bias)
        self.critic[0] = expanded

    def load_state_dict(self, state_dict, strict=True):
        state = dict(state_dict)
        weight = state['critic.0.weight']
        if weight.shape[1] == 1318:
            state['critic.0.weight'] = torch.cat((weight, weight.new_zeros(weight.shape[0], 2)), dim=1)
        elif weight.shape[1] != 1320:
            raise ValueError('unsupported critic input width')
        return super().load_state_dict(state, strict=strict)


def append_heading_context(privileged, observations):
    if privileged.shape[-1] != 1318 or observations.shape[-1] != 7003:
        raise ValueError('unexpected original observation layout')
    return torch.cat((privileged, observations[:, -2:]), dim=-1)
