"""TWIST AnyAdapter with an observable-demand gate on adapter influence."""

import torch

from .actor_critic_twist_anyadapter_opentrack import TwistAnyAdapterOpenTrackActorCritic


def compute_adapter_gate(base_obs, contract, default_dof_pos):
    reference = base_obs[:, :31]
    relative_q = base_obs[:, 36:59]
    joint_error = (reference[:, 8:31] -
                   (relative_q + default_dof_pos)).abs().mean(dim=-1)
    speed = reference[:, 4:7].norm(dim=-1)
    yaw_rate = 0.4 * reference[:, 7].abs()
    demand = torch.maximum(torch.maximum(speed, yaw_rate), 2.0 * joint_error)
    low, high = contract.unbind()
    fraction = ((demand - low) / (high - low)).clamp(0, 1)
    return (fraction * fraction * (3 - 2 * fraction)).unsqueeze(-1)


class TwistBaselineGuardedActorCritic(TwistAnyAdapterOpenTrackActorCritic):
    """Leave low-demand behavior to TWIST; enable layer adapters when needed.

    The gate reads only deployment-observable reference and joint sensors from
    the original 1155-D TWIST input. It never reads privileged critic state.
    A buffer records the contract so the exporter can select this class from
    the checkpoint without relying on filenames or command-line memory.
    """

    def __init__(self, *args, demand_low=0.18, demand_high=0.55, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= demand_low < demand_high:
            raise ValueError('invalid adapter demand gate thresholds')
        self.register_buffer('demand_gate_contract',
                             torch.tensor([demand_low, demand_high], dtype=torch.float32))
        self.register_buffer('default_dof_pos', torch.tensor([
            -0.2, 0, 0, 0.4, -0.2, 0,
            -0.2, 0, 0, 0.4, -0.2, 0,
            0, 0, 0,
            0, 0.4, 0, 1.2,
            0, -0.4, 0, 1.2,
        ], dtype=torch.float32))

    def adapter_demand(self, base_obs):
        reference = base_obs[:, :31]
        relative_q = base_obs[:, 36:59]
        joint_error = (reference[:, 8:31] -
                       (relative_q + self.default_dof_pos)).abs().mean(dim=-1)
        speed = reference[:, 4:7].norm(dim=-1)
        yaw_rate = 0.4 * reference[:, 7].abs()
        return torch.maximum(torch.maximum(speed, yaw_rate), 2.0 * joint_error)

    def adapter_gate(self, base_obs):
        return compute_adapter_gate(base_obs, self.demand_gate_contract,
                                    self.default_dof_pos)

    def actor_mean(self, observations):
        base_obs, history = self.split_obs(observations)
        z = self.encode_history_for_policy(history)
        adapted = self.layerwise_actor(base_obs, z)
        baseline = self.layerwise_actor.base_forward(base_obs)
        return baseline + self.adapter_gate(base_obs) * (adapted - baseline)
