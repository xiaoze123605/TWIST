"""Causal dynamics supervision for the frozen TWIST adapter experiment."""

import torch
import torch.nn.functional as F

from .ppo_any2track import PPOAny2Track


class PPOTwistBaselineAdapter(PPOAny2Track):
    def __init__(self, *args, adapter_tail_threshold=0.0,
                 adapter_tail_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_tail_threshold = float(adapter_tail_threshold)
        self.adapter_tail_coef = float(adapter_tail_coef)
        if self.adapter_tail_threshold < 0 or self.adapter_tail_coef < 0:
            raise ValueError("adapter tail threshold and coefficient must be non-negative")

    def _adapter_regularization(self, delta):
        base_loss = delta.square().mean()
        if self.adapter_tail_coef == 0.0:
            return base_loss, delta.new_zeros(())
        tail_loss = (delta.abs() - self.adapter_tail_threshold).clamp_min(0).square().mean()
        return base_loss + self.adapter_tail_coef * tail_loss, tail_loss

    def _autoregressive_world_model_loss(
        self, obs_sequence, action_sequence, next_obs_sequence,
        done_sequence, available_sequence,
    ):
        actor = self.actor_critic
        _, history = actor.split_obs(obs_sequence[:, 0])
        state = actor.selected_state(obs_sequence[:, 0])
        sums = {name: state.new_zeros(()) for name in actor.wm_component_splits}
        valid_total = state.new_zeros(())

        for step in range(obs_sequence.shape[1]):
            # The next observation contains the environment's committed
            # (state_t, sent_action_t) history frame. Under delay/clipping this
            # differs from the PPO sample in action_sequence.
            _, next_history = actor.split_obs(next_obs_sequence[:, step])
            sent_action = next_history[:, -1, -actor.num_actions:]
            prediction = actor.predict_world_model(
                obs_sequence[:, step], sent_action, state=state, history=history,
            )
            target = actor.selected_state(next_obs_sequence[:, step])
            done = done_sequence[:, step].bool().reshape(-1, 1)
            valid = available_sequence[:, step].float().reshape(-1, 1) * (~done).float()
            valid_total = valid_total + valid.sum()
            for name, (start, end) in actor.wm_component_splits.items():
                if self.world_model_loss_type == "mse":
                    per_dim = (prediction[:, start:end] - target[:, start:end]).square()
                else:
                    per_dim = F.smooth_l1_loss(
                        prediction[:, start:end], target[:, start:end], reduction="none",
                    )
                sums[name] = sums[name] + (per_dim.mean(-1, keepdim=True) * valid).sum()

            # Online history records the state before the command is sent.
            frame = torch.cat((state, sent_action), dim=-1)
            predicted_history = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)
            history = torch.where(done.unsqueeze(-1), next_history, predicted_history)
            state = torch.where(done, target, prediction)

        denominator = valid_total.clamp_min(1.0)
        components = {name: value / denominator for name, value in sums.items()}
        total = sum(self.world_model_component_weights[name] * value
                    for name, value in components.items())
        return total, components, int(valid_total.detach().item())
