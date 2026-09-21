"""Paper-style AnyAdapter PPO: dynamics-WM first, then detached-latent PPO."""

import torch

from .ppo_any2track import PPOAny2Track


class PPOAnyAdapterOpenTrack(PPOAny2Track):
    """Clean OpenTrack baseline without DTERA or residual regularizers.

    PPOAny2Track already supplies the required 20-step autoregressive storage
    traversal and optimizer separation.  This specialization fixes the world
    model objective to L1 and rejects every legacy regularization knob rather
    than silently carrying it into this experiment.
    """

    def __init__(self, *args, **kwargs):
        forbidden = (
            "stand_anchor_coef", "synthetic_stand_anchor_coef", "adapter_reg_coef",
            "adapter_bias_reg_coef", "residual_saturation_reg_coef",
            "tracking_error_loss_coef", "risk_loss_coef", "confidence_loss_coef",
            "error_prediction_loss_coef",
        )
        enabled = {
            key: value for key, value in kwargs.items()
            if key in forbidden and float(value) != 0.0
        }
        if enabled:
            raise ValueError(
                "AnyAdapter-OpenTrack baseline forbids auxiliary/residual losses: "
                f"{enabled}"
            )
        kwargs["stand_anchor_coef"] = 0.0
        kwargs["synthetic_stand_anchor_coef"] = 0.0
        kwargs["adapter_reg_coef"] = 0.0
        kwargs["adapter_bias_reg_coef"] = 0.0
        kwargs["world_model_loss_type"] = "l1"
        super().__init__(*args, **kwargs)

    def _autoregressive_world_model_loss(
        self, obs_sequence, action_sequence, next_obs_sequence,
        done_sequence, available_sequence,
    ):
        """20-step autoregressive L1 forward-dynamics objective from OpenTrack."""
        actor_critic = self.actor_critic
        _, history = actor_critic.split_obs(obs_sequence[:, 0])
        state = actor_critic.selected_state(obs_sequence[:, 0])
        component_sums = {
            name: state.new_zeros(()) for name in actor_critic.wm_component_splits
        }
        valid_total = state.new_zeros(())
        for step in range(obs_sequence.shape[1]):
            prediction = actor_critic.predict_world_model(
                obs_sequence[:, step], action_sequence[:, step], state=state,
                history=history,
            )
            target = actor_critic.selected_state(next_obs_sequence[:, step])
            done = done_sequence[:, step].bool().reshape(-1, 1)
            valid = available_sequence[:, step].float().reshape(-1, 1) * (~done).float()
            valid_total += valid.sum()
            for name, (start, end) in actor_critic.wm_component_splits.items():
                per_sample = (prediction[:, start:end] - target[:, start:end]).abs().mean(
                    dim=-1, keepdim=True
                )
                component_sums[name] += (per_sample * valid).sum()
            predicted_frame = torch.cat([prediction, action_sequence[:, step]], dim=-1)
            predicted_history = torch.cat([history[:, 1:], predicted_frame.unsqueeze(1)], dim=1)
            _, reset_history = actor_critic.split_obs(next_obs_sequence[:, step])
            history = torch.where(done.unsqueeze(-1), reset_history, predicted_history)
            state = torch.where(done, target, prediction)
        denominator = valid_total.clamp_min(1.0)
        components = {name: value / denominator for name, value in component_sums.items()}
        total = sum(self.world_model_component_weights[name] * value for name, value in components.items())
        return total, components, int(valid_total.detach().item())
