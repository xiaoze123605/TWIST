"""PPO parameter ownership and auxiliary updates for DTERA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from .ppo_anyadapter import PPOAnyAdapter


class PPODTERA(PPOAnyAdapter):
    def __init__(
        self,
        *args,
        risk_horizon: int = 10,
        risk_pos_weight: float = 5.0,
        risk_learning_rate: Optional[float] = None,
        world_model_bootstrap: bool = True,
        world_model_bootstrap_probability: float = 0.8,
        **kwargs,
    ):
        kwargs["joint_encoder_optimization"] = True
        super().__init__(*args, **kwargs)
        if not getattr(self.actor_critic, "is_dtera", False):
            raise TypeError("PPODTERA requires TwistDTERAActorCritic")
        self.risk_horizon = int(risk_horizon)
        self.risk_pos_weight = float(risk_pos_weight)
        self.world_model_bootstrap = bool(world_model_bootstrap)
        self.world_model_bootstrap_probability = float(
            world_model_bootstrap_probability
        )

        adapter_critic_params = [
            p
            for module in (self.actor_critic.adapter, self.actor_critic.critic)
            for p in module.parameters()
            if p.requires_grad
        ]
        std_params = []
        if self.actor_critic.std.requires_grad:
            std_params.append(self.actor_critic.std)
        tracking_aux_params = [
            p
            for module in (
                self.actor_critic.tracking_error_history_encoder,
                self.actor_critic.error_trend_predictor,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        self.ppo_params = adapter_critic_params + std_params + tracking_aux_params
        self.history_encoder_params = [
            p for p in self.actor_critic.history_encoder.parameters() if p.requires_grad
        ]
        self.world_model_params = [
            p for p in self.actor_critic.world_model.parameters() if p.requires_grad
        ]
        self.wm_params = self.history_encoder_params + self.world_model_params
        self.risk_params = [
            p for p in self.actor_critic.risk_predictor.parameters() if p.requires_grad
        ]

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": adapter_critic_params, "weight_decay": self.weight_decay},
                {"params": std_params, "weight_decay": 0.0},
                {"params": tracking_aux_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        self.wm_optimizer = torch.optim.Adam(
            self.wm_params, lr=self.learning_rate, weight_decay=0.0
        )
        self.risk_optimizer = torch.optim.Adam(
            self.risk_params,
            lr=self.learning_rate if risk_learning_rate is None else risk_learning_rate,
            weight_decay=0.0,
        )
        self.optimizer = self.ppo_optimizer
        self._assert_optimizer_ownership()

    @staticmethod
    def _optimizer_ids(optimizer):
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def _assert_optimizer_ownership(self):
        ppo_ids = self._optimizer_ids(self.ppo_optimizer)
        wm_ids = self._optimizer_ids(self.wm_optimizer)
        risk_ids = self._optimizer_ids(self.risk_optimizer)
        if ppo_ids & wm_ids or ppo_ids & risk_ids or wm_ids & risk_ids:
            raise AssertionError("a DTERA parameter is owned by more than one optimizer")
        trainable_ids = {
            id(parameter)
            for parameter in self.actor_critic.parameters()
            if parameter.requires_grad
        }
        if trainable_ids != ppo_ids | wm_ids | risk_ids:
            missing = len(trainable_ids - (ppo_ids | wm_ids | risk_ids))
            extra = len((ppo_ids | wm_ids | risk_ids) - trainable_ids)
            raise AssertionError(f"DTERA optimizer ownership mismatch: missing={missing}, extra={extra}")

        dyn_ids = {id(p) for p in self.actor_critic.history_encoder.parameters()}
        tracking_ids = {
            id(p) for p in self.actor_critic.tracking_error_history_encoder.parameters()
        }
        risk_predictor_ids = {id(p) for p in self.actor_critic.risk_predictor.parameters()}
        diagnostics = {
            "dynamics_encoder_in_ppo": bool(dyn_ids & ppo_ids),
            "dynamics_encoder_in_wm": dyn_ids <= wm_ids,
            "tracking_encoder_in_ppo": tracking_ids <= ppo_ids,
            "tracking_encoder_in_wm": bool(tracking_ids & wm_ids),
            "risk_predictor_only_in_risk": (
                risk_predictor_ids <= risk_ids
                and not bool(risk_predictor_ids & (ppo_ids | wm_ids))
            ),
        }
        self.optimizer_ownership = diagnostics
        for key, value in diagnostics.items():
            print(f"[DTERA] {key}={value}")
        expected = {
            "dynamics_encoder_in_ppo": False,
            "dynamics_encoder_in_wm": True,
            "tracking_encoder_in_ppo": True,
            "tracking_encoder_in_wm": False,
            "risk_predictor_only_in_risk": True,
        }
        if diagnostics != expected:
            raise AssertionError(f"unexpected DTERA optimizer ownership: {diagnostics}")

    def process_env_step(self, rewards, dones, infos):
        timeouts = infos.get("time_outs", None)
        self.transition.timeouts = (
            torch.zeros_like(dones) if timeouts is None else timeouts.detach()
        )
        return super().process_env_step(rewards, dones, infos)

    def _world_model_loss_from_batch(
        self,
        obs_batch,
        actions_batch,
        next_obs_batch,
        dones_batch,
        available_batch,
    ):
        indices = self.actor_critic.wm_target_indices.to(next_obs_batch.device)
        target = next_obs_batch.index_select(1, indices)
        members = self.actor_critic.predict_world_model_members(obs_batch, actions_batch)
        self.actor_critic.update_uncertainty_ema(members)
        valid = (1.0 - dones_batch.float()).reshape(1, -1, 1)
        valid = valid * available_batch.float().reshape(1, -1, 1)
        if self.world_model_bootstrap:
            bootstrap = (
                torch.rand(
                    members.shape[0], members.shape[1], 1,
                    device=members.device,
                )
                < self.world_model_bootstrap_probability
            ).float()
            # Every member must retain at least one valid sample.
            empty = (bootstrap * valid).sum(dim=1, keepdim=True) == 0
            bootstrap = torch.where(empty, torch.ones_like(bootstrap), bootstrap)
            valid = valid * bootstrap

        total = members.new_zeros(())
        component_info = {}
        for name, (start, end) in self.actor_critic.wm_component_splits.items():
            pred = members[:, :, start:end]
            truth = target[None, :, start:end]
            if self.world_model_loss_type == "mse":
                per_sample = (pred - truth).square().mean(dim=-1, keepdim=True)
            else:
                per_sample = F.smooth_l1_loss(
                    pred, truth.expand_as(pred), reduction="none"
                ).mean(dim=-1, keepdim=True)
            member_counts = valid.sum(dim=1).clamp_min(1.0)
            member_losses = (per_sample * valid).sum(dim=1) / member_counts
            component_loss = member_losses.mean()
            component_info[f"wm_{name}_loss"] = float(component_loss.detach().cpu())
            total = total + float(
                self.actor_critic.wm_component_weights.get(name, 1.0)
            ) * component_loss
        return total, component_info

    def _future_risk_targets(self):
        dones = self.storage.dones.bool()
        failures = dones & ~self.storage.timeouts.bool()
        targets = torch.zeros_like(dones, dtype=torch.float32)
        alive = torch.ones_like(dones, dtype=torch.bool)
        horizon = min(self.risk_horizon, self.storage.num_transitions_per_env)
        for offset in range(horizon):
            if offset == 0:
                shifted_done = dones
                shifted_failure = failures
            else:
                shifted_done = torch.zeros_like(dones)
                shifted_failure = torch.zeros_like(failures)
                shifted_done[:-offset] = dones[offset:]
                shifted_failure[:-offset] = failures[offset:]
            targets = torch.maximum(targets, (alive & shifted_failure).float())
            alive = alive & ~shifted_done
        return targets

    def _train_risk_predictor(self, risk_targets=None):
        if risk_targets is None:
            risk_targets = self._future_risk_targets()
        targets = risk_targets.flatten(0, 1).reshape(-1)
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        batch_size = 4096
        order = torch.randperm(targets.numel(), device=targets.device)
        total_loss = 0.0
        batches = 0
        pos_weight = targets.new_tensor(self.risk_pos_weight)
        for start in range(0, targets.numel(), batch_size):
            ids = order[start : start + batch_size]
            logits = self.actor_critic.risk_logits(observations[ids], actions[ids])
            loss = F.binary_cross_entropy_with_logits(
                logits, targets[ids], pos_weight=pos_weight
            )
            self.risk_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.risk_params, self.max_grad_norm)
            self.risk_optimizer.step()
            total_loss += loss.item()
            batches += 1
        return {
            "risk_loss": total_loss / max(batches, 1),
            "risk_positive_ratio": float(targets.mean().cpu()),
        }

    def update(self):
        # Freeze labels before PPO clears rollout metadata, but delay the risk
        # step until after PPO has recomputed log probabilities with exactly
        # the same risk/gate parameters that generated this rollout.
        risk_targets = self._future_risk_targets()
        result = super().update()
        risk_metrics = self._train_risk_predictor(risk_targets)
        self.anyadapter_metrics.update(risk_metrics)
        return result
