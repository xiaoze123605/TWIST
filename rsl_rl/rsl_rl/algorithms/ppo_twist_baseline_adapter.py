"""Causal WM supervision and guarded refinement of the frozen TWIST actor."""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from .ppo_any2track import PPOAny2Track


class PPOTwistBaselineAdapter(PPOAny2Track):
    def __init__(self, *args, adapter_tail_threshold=0.0,
                 adapter_tail_coef=0.0, freeze_world_model=False,
                 policy_anchor_checkpoint=None, policy_anchor_coef=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_tail_threshold = float(adapter_tail_threshold)
        self.adapter_tail_coef = float(adapter_tail_coef)
        self.freeze_world_model = bool(freeze_world_model)
        self.policy_anchor_coef = float(policy_anchor_coef)
        if self.adapter_tail_threshold < 0 or self.adapter_tail_coef < 0:
            raise ValueError("adapter tail threshold and coefficient must be non-negative")
        if self.policy_anchor_coef < 0:
            raise ValueError("policy anchor coefficient must be non-negative")
        if self.freeze_world_model:
            if self.world_model_loss_coef != 0.0:
                raise ValueError("frozen world model requires world_model_loss_coef=0")
            self.actor_critic.history_encoder.requires_grad_(False)
            self.actor_critic.world_model.requires_grad_(False)
        self.anchor_policy = None
        if self.policy_anchor_coef > 0:
            if not policy_anchor_checkpoint:
                raise ValueError("policy anchor coefficient requires a checkpoint")
            checkpoint = Path(policy_anchor_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            source = torch.load(checkpoint, map_location="cpu")
            self.anchor_policy = deepcopy(self.actor_critic)
            self.anchor_policy.load_state_dict(source["model_state_dict"], strict=True)
            self.anchor_policy.eval().requires_grad_(False)

    def _policy_anchor_penalty(self, observations, policy_mean):
        if self.anchor_policy is None:
            return super()._policy_anchor_penalty(observations, policy_mean)
        with torch.no_grad():
            anchor_mean = self.anchor_policy.actor_mean(observations)
        difference = policy_mean - anchor_mean
        return (self.policy_anchor_coef * difference.square().mean(),
                difference.abs().mean())

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
                elif self.world_model_loss_type == "l1":
                    per_dim = (prediction[:, start:end] - target[:, start:end]).abs()
                elif self.world_model_loss_type == "smooth_l1":
                    per_dim = F.smooth_l1_loss(
                        prediction[:, start:end], target[:, start:end], reduction="none",
                    )
                else:
                    raise ValueError(f"Unsupported world_model_loss_type: {self.world_model_loss_type}")
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
