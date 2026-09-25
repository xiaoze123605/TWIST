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
                 anchor_replay_size=0, anchor_replay_batch_size=0,
                 anchor_replay_fraction=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_tail_threshold = float(adapter_tail_threshold)
        self.adapter_tail_coef = float(adapter_tail_coef)
        self.freeze_world_model = bool(freeze_world_model)
        self.policy_anchor_coef = float(policy_anchor_coef)
        self.anchor_replay_size = int(anchor_replay_size)
        self.anchor_replay_batch_size = int(anchor_replay_batch_size)
        self.anchor_replay_fraction = float(anchor_replay_fraction)
        if (self.anchor_replay_size < 0 or self.anchor_replay_batch_size < 0 or
                not 0.0 <= self.anchor_replay_fraction <= 1.0):
            raise ValueError("invalid anchor replay configuration")
        if self.anchor_replay_fraction and (not self.anchor_replay_size or
                                            not self.anchor_replay_batch_size or
                                            not self.freeze_world_model):
            raise ValueError("anchor replay requires a frozen encoder and nonempty bank")
        self.anchor_replay_count = 0
        actor = self.actor_critic
        self.anchor_replay_base = None
        self.anchor_replay_latent = None
        self.anchor_replay_actions = None
        if self.anchor_replay_size:
            self.anchor_replay_base = torch.empty(
                self.anchor_replay_size, actor.base_obs_dim, dtype=torch.float16)
            self.anchor_replay_latent = torch.empty(
                self.anchor_replay_size, actor.latent_dim, dtype=torch.float16)
            self.anchor_replay_actions = torch.empty(
                self.anchor_replay_size, actor.num_actions, dtype=torch.float16)
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
        penalty = difference.square().mean()
        if self.anchor_replay_count and self.anchor_replay_fraction:
            indices = torch.randint(
                self.anchor_replay_count, (self.anchor_replay_batch_size,))
            base = self.anchor_replay_base[indices].to(
                device=policy_mean.device, dtype=policy_mean.dtype)
            latent = self.anchor_replay_latent[indices].to(
                device=policy_mean.device, dtype=policy_mean.dtype)
            target = self.anchor_replay_actions[indices].to(
                device=policy_mean.device, dtype=policy_mean.dtype)
            replay_mean = self.actor_critic.layerwise_actor(base, latent)
            replay_penalty = (replay_mean - target).square().mean()
            penalty = ((1.0 - self.anchor_replay_fraction) * penalty +
                       self.anchor_replay_fraction * replay_penalty)
        return self.policy_anchor_coef * penalty, difference.abs().mean()

    def _collect_anchor_replay(self):
        if not self.anchor_replay_size or self.anchor_replay_count >= self.anchor_replay_size:
            return
        observations = self.storage.observations.flatten(0, 1)
        take = min(512, self.anchor_replay_size - self.anchor_replay_count)
        indices = torch.randperm(observations.shape[0], device=observations.device)[:take]
        with torch.no_grad():
            base, history = self.actor_critic.split_obs(observations[indices])
            latent = self.actor_critic.encode_history_for_policy(history)
            teacher = self.anchor_policy.layerwise_actor(base, latent)
            start, end = self.anchor_replay_count, self.anchor_replay_count + take
            self.anchor_replay_base[start:end] = base.to(
                device="cpu", dtype=torch.float16)
            self.anchor_replay_latent[start:end] = latent.to(
                device="cpu", dtype=torch.float16)
            self.anchor_replay_actions[start:end] = teacher.to(
                device="cpu", dtype=torch.float16)
            self.anchor_replay_count = end

    def on_save_checkpoint(self):
        if not self.anchor_replay_size:
            return {}
        count = self.anchor_replay_count
        return dict(anchor_replay_count=count,
                    anchor_replay_base=self.anchor_replay_base[:count].clone(),
                    anchor_replay_latent=self.anchor_replay_latent[:count].clone(),
                    anchor_replay_actions=self.anchor_replay_actions[:count].clone())

    def on_load_checkpoint(self, checkpoint):
        if not self.anchor_replay_size or "anchor_replay_count" not in checkpoint:
            return
        count = int(checkpoint["anchor_replay_count"])
        if not 0 <= count <= self.anchor_replay_size:
            raise ValueError("invalid anchor replay count in checkpoint")
        for name in ("base", "latent", "actions"):
            source = checkpoint[f"anchor_replay_{name}"]
            destination = getattr(self, f"anchor_replay_{name}")
            if source.shape != destination[:count].shape:
                raise ValueError(f"invalid anchor replay {name} shape")
            destination[:count] = source.cpu()
        self.anchor_replay_count = count

    def update(self):
        self._collect_anchor_replay()
        result = super().update()
        self.anyadapter_metrics["anchor_replay_count"] = self.anchor_replay_count
        return result

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
