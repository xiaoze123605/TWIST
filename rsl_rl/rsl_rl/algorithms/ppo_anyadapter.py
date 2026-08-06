"""
PPO with AnyAdapter auxiliary losses.

Drop into:
    TWIST/rsl_rl/rsl_rl/algorithms/ppo_anyadapter.py

This class is intentionally close to TWIST/rsl_rl/rsl_rl/algorithms/ppo.py, but
adds:
    - world_model_loss_coef * actor_critic.world_model_loss(obs_batch)
    - adapter_reg_coef * actor_critic.adapter_regularization_loss(obs_batch)

It assumes the environment has already appended history to actor observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO


class PPOAnyAdapter(PPO):
    def __init__(
        self,
        *args,
        world_model_loss_coef: float = 0.1,
        adapter_reg_coef: float = 1e-3,
        adapter_bias_reg_coef: float = 0.0,
        stand_anchor_coef: float = 1.0,
        synthetic_stand_anchor_coef: float = 0.0,
        synthetic_stand_root_height: float = 0.793,
        stand_vel_threshold: float = 0.05,
        stand_dof_threshold: float = 0.15,
        world_model_loss_type: str = "smooth_l1",
        weight_decay: float = 0.0,
        joint_encoder_optimization: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.world_model_loss_coef = float(world_model_loss_coef)
        self.adapter_reg_coef = float(adapter_reg_coef)
        self.adapter_bias_reg_coef = float(adapter_bias_reg_coef)
        self.stand_anchor_coef = float(stand_anchor_coef)
        self.synthetic_stand_anchor_coef = float(synthetic_stand_anchor_coef)
        self.synthetic_stand_root_height = float(synthetic_stand_root_height)
        self.stand_vel_threshold = float(stand_vel_threshold)
        self.stand_dof_threshold = float(stand_dof_threshold)
        self.world_model_loss_type = world_model_loss_type
        self.weight_decay = float(weight_decay)
        self.joint_encoder_optimization = bool(joint_encoder_optimization)
        self.skip_dagger_update = True
        self.requires_next_observations = True

        self.ppo_params = []
        self.ppo_params += [p for p in self.actor_critic.adapter.parameters() if p.requires_grad]
        self.ppo_params += [p for p in self.actor_critic.critic.parameters() if p.requires_grad]
        if getattr(self.actor_critic, "std", None) is not None and self.actor_critic.std.requires_grad:
            self.ppo_params.append(self.actor_critic.std)

        self.wm_params = []
        self.wm_params += [p for p in self.actor_critic.history_encoder.parameters() if p.requires_grad]
        self.wm_params += [p for p in self.actor_critic.world_model.parameters() if p.requires_grad]

        if self.joint_encoder_optimization:
            self.ppo_params += self.wm_params

        self.ppo_optimizer = torch.optim.Adam(
            self.ppo_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if not self.joint_encoder_optimization:
            self.wm_optimizer = torch.optim.Adam(self.wm_params, lr=self.learning_rate)
        self.optimizer = self.ppo_optimizer
        self.anyadapter_metrics = {}
        self._wm_target_warning_printed = False

    def _has_world_model_target(self) -> bool:
        has_next_obs = hasattr(self.storage, "next_observations") and self.storage.next_observations is not None
        has_available_mask = (
            hasattr(self.storage, "next_observations_available")
            and bool(torch.any(self.storage.next_observations_available).item())
        )
        return has_next_obs and has_available_mask

    def _warn_missing_world_model_target_once(self) -> None:
        if self._wm_target_warning_printed:
            return
        has_next_obs = hasattr(self.storage, "next_observations") and self.storage.next_observations is not None
        has_target_indices = getattr(self.actor_critic, "wm_target_indices", None) is not None
        has_predict_world_model = hasattr(self.actor_critic, "predict_world_model")
        print(
            "[AnyAdapter] world model target missing, wm loss skipped "
            f"(rollout storage has next_obs: {has_next_obs}; "
            f"wm_target_indices set: {has_target_indices}; "
            "PPOAnyAdapter active: True; "
            f"actor_critic.predict_world_model available: {has_predict_world_model}; "
            "predict_world_model called: False)"
        )
        self._wm_target_warning_printed = True

    def process_env_step(self, rewards, dones, infos):
        next_obs = infos.get("next_observations", None)
        if next_obs is not None:
            self.transition.next_observations = next_obs.detach()
        return super().process_env_step(rewards, dones, infos)

    def _world_model_loss_from_batch(self, obs_batch, actions_batch, next_obs_batch, dones_batch, available_batch):
        wm_target_indices = getattr(self.actor_critic, "wm_target_indices", None)
        if wm_target_indices is None:
            target_next_state = next_obs_batch[:, : self.actor_critic.hist_state_dim]
        else:
            idx = wm_target_indices.to(next_obs_batch.device)
            target_next_state = next_obs_batch.index_select(dim=1, index=idx)

        pred_next_state = self.actor_critic.predict_world_model(obs_batch, actions_batch)

        valid_mask = (1.0 - dones_batch.float()).reshape(-1, 1)
        if available_batch is not None:
            valid_mask = valid_mask * available_batch.float().reshape(-1, 1)
        valid_count = valid_mask.sum().clamp_min(1.0)

        splits = getattr(self.actor_critic, "wm_component_splits", None)
        weights = getattr(self.actor_critic, "wm_component_weights", None)
        if splits is not None and weights is not None:
            total_loss = pred_next_state.new_zeros(())
            for name, (start, end) in splits.items():
                w = float(weights.get(name, 1.0))
                pred_slice = pred_next_state[:, start:end]
                target_slice = target_next_state[:, start:end]
                if self.world_model_loss_type == "mse":
                    comp_loss_per_dim = (pred_slice - target_slice) ** 2
                else:
                    comp_loss_per_dim = F.smooth_l1_loss(pred_slice, target_slice, reduction="none")
                comp_loss_per_sample = comp_loss_per_dim.mean(dim=-1, keepdim=True)
                total_loss = total_loss + w * ((comp_loss_per_sample * valid_mask).sum() / valid_count)
            return total_loss
        else:
            if self.world_model_loss_type == "mse":
                loss_per_dim = (pred_next_state - target_next_state) ** 2
            else:
                loss_per_dim = F.smooth_l1_loss(pred_next_state, target_next_state, reduction="none")
            loss_per_sample = loss_per_dim.mean(dim=-1, keepdim=True)
            return (loss_per_sample * valid_mask).sum() / valid_count

    def _stand_anchor_loss_from_batch(self, obs_batch):
        base_obs, _ = self.actor_critic.split_obs(obs_batch)
        if base_obs.shape[1] < 31:
            zero = obs_batch.new_tensor(0.0)
            return zero, zero

        root_vel = base_obs[:, 4:7]
        yaw_ang_vel = base_obs[:, 7]
        ref_dof_pos = base_obs[:, 8:31]
        default_ref = self.actor_critic.default_ref_dof_pos.to(base_obs.device).view(1, -1)

        stand_mask = (
            (torch.norm(root_vel, dim=-1) < self.stand_vel_threshold)
            & (torch.abs(yaw_ang_vel) < self.stand_vel_threshold)
            & (torch.mean(torch.abs(ref_dof_pos - default_ref), dim=-1) < self.stand_dof_threshold)
        )
        stand_ratio = stand_mask.float().mean()
        if not torch.any(stand_mask):
            return obs_batch.new_tensor(0.0), stand_ratio

        delta = self.actor_critic.action_delta(obs_batch[stand_mask], detach_history=True)
        return (delta ** 2).mean(), stand_ratio

    def _synthetic_stand_anchor_loss_from_batch(self, obs_batch):
        stand_obs = obs_batch.clone()
        base_obs, _ = self.actor_critic.split_obs(stand_obs)
        default_ref = self.actor_critic.default_ref_dof_pos.to(base_obs.device)
        base_obs[:, 0] = self.synthetic_stand_root_height
        base_obs[:, 1:4] = 0.0
        base_obs[:, 4:7] = 0.0
        base_obs[:, 7] = 0.0
        base_obs[:, 8:31] = default_ref.view(1, -1)
        delta = self.actor_critic.action_delta(stand_obs, detach_history=True)
        return (delta ** 2).mean()

    @staticmethod
    def _grad_norm(parameters):
        sq_sum = None
        for p in parameters:
            if p.grad is None:
                continue
            norm_sq = torch.sum(p.grad.detach() ** 2)
            sq_sum = norm_sq if sq_sum is None else sq_sum + norm_sq
        if sq_sum is None:
            return 0.0
        return float(torch.sqrt(sq_sum).detach().cpu())

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_wm_loss = 0.0
        mean_adapter_reg_loss = 0.0
        mean_adapter_bias_reg_loss = 0.0
        mean_adapter_delta_l2 = 0.0
        mean_stand_anchor_loss = 0.0
        mean_synthetic_stand_anchor_loss = 0.0
        mean_stand_sample_ratio = 0.0
        mean_history_encoder_ppo_grad_norm = 0.0
        mean_history_encoder_wm_grad_norm = 0.0
        mean_adapter_grad_norm = 0.0
        mean_max_abs_delta_action = 0.0
        mean_mean_abs_delta_action = 0.0
        wm_loss_skipped = False

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator_anyadapter(self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches

        for sample in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                next_obs_batch,
                dones_batch,
                next_obs_available_batch,
            ) = sample

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            wm_loss = obs_batch.new_tensor(0.0)
            adapter_reg_loss = obs_batch.new_tensor(0.0)
            adapter_bias_reg_loss = obs_batch.new_tensor(0.0)
            stand_anchor_loss = obs_batch.new_tensor(0.0)
            synthetic_stand_anchor_loss = obs_batch.new_tensor(0.0)
            stand_sample_ratio = obs_batch.new_tensor(0.0)
            if hasattr(self.actor_critic, "adapter_regularization_loss") and self.adapter_reg_coef > 0.0:
                adapter_reg_loss, adapter_info = self.actor_critic.adapter_regularization_loss(obs_batch)
                loss = loss + self.adapter_reg_coef * adapter_reg_loss
                mean_adapter_delta_l2 += adapter_info.get("adapter_delta_l2", adapter_reg_loss.item())
                mean_max_abs_delta_action += adapter_info.get("max_abs_delta_action", 0.0)
                mean_mean_abs_delta_action += adapter_info.get("mean_abs_delta_action", 0.0)
            if (
                hasattr(self.actor_critic, "adapter_bias_regularization_loss")
                and self.adapter_bias_reg_coef > 0.0
            ):
                adapter_bias_reg_loss = self.actor_critic.adapter_bias_regularization_loss(obs_batch)
                loss = loss + self.adapter_bias_reg_coef * adapter_bias_reg_loss
            if self.stand_anchor_coef > 0.0:
                stand_anchor_loss, stand_sample_ratio = self._stand_anchor_loss_from_batch(obs_batch)
                loss = loss + self.stand_anchor_coef * stand_anchor_loss
            if self.synthetic_stand_anchor_coef > 0.0:
                synthetic_stand_anchor_loss = self._synthetic_stand_anchor_loss_from_batch(obs_batch)
                loss = loss + self.synthetic_stand_anchor_coef * synthetic_stand_anchor_loss

            wm_target_available = self._has_world_model_target()
            if self.joint_encoder_optimization and self.world_model_loss_coef > 0.0:
                if hasattr(self.actor_critic, "predict_world_model") and wm_target_available:
                    wm_loss = self._world_model_loss_from_batch(
                        obs_batch,
                        actions_batch,
                        next_obs_batch,
                        dones_batch,
                        next_obs_available_batch,
                    )
                    loss = loss + self.world_model_loss_coef * wm_loss
                else:
                    wm_loss_skipped = True
                    self._warn_missing_world_model_target_once()

            self.ppo_optimizer.zero_grad()
            if hasattr(self, "wm_optimizer"):
                self.wm_optimizer.zero_grad()
            loss.backward()
            history_encoder_ppo_grad_norm = self._grad_norm(self.actor_critic.history_encoder.parameters())
            adapter_grad_norm = self._grad_norm(self.actor_critic.adapter.parameters())
            nn.utils.clip_grad_norm_(self.ppo_params, self.max_grad_norm)
            self.ppo_optimizer.step()

            if (
                not self.joint_encoder_optimization
                and hasattr(self.actor_critic, "predict_world_model")
                and self.world_model_loss_coef > 0.0
            ):
                if wm_target_available:
                    self.wm_optimizer.zero_grad()
                    wm_loss = self._world_model_loss_from_batch(
                        obs_batch,
                        actions_batch,
                        next_obs_batch,
                        dones_batch,
                        next_obs_available_batch,
                    )
                    (self.world_model_loss_coef * wm_loss).backward()
                    history_encoder_wm_grad_norm = self._grad_norm(self.actor_critic.history_encoder.parameters())
                    nn.utils.clip_grad_norm_(self.wm_params, self.max_grad_norm)
                    self.wm_optimizer.step()
                else:
                    history_encoder_wm_grad_norm = 0.0
                    wm_loss_skipped = True
                    self._warn_missing_world_model_target_once()
            elif self.joint_encoder_optimization:
                history_encoder_wm_grad_norm = history_encoder_ppo_grad_norm
            else:
                history_encoder_wm_grad_norm = 0.0

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_wm_loss += wm_loss.item()
            mean_adapter_reg_loss += adapter_reg_loss.item()
            mean_adapter_bias_reg_loss += adapter_bias_reg_loss.item()
            mean_stand_anchor_loss += stand_anchor_loss.item()
            mean_synthetic_stand_anchor_loss += synthetic_stand_anchor_loss.item()
            mean_stand_sample_ratio += stand_sample_ratio.item()
            mean_history_encoder_ppo_grad_norm += history_encoder_ppo_grad_norm
            mean_history_encoder_wm_grad_norm += history_encoder_wm_grad_norm
            mean_adapter_grad_norm += adapter_grad_norm

        if self.fix_std:
            std_stage = min(max((self.counter - self.std_schedule[2]), 0) / self.std_schedule[3], 1)
            std_coef = std_stage * (self.std_schedule[1] - self.std_schedule[0]) + self.std_schedule[0]
            if hasattr(self.actor_critic, "update_std"):
                self.actor_critic.update_std(std_coef)

        self.storage.clear()
        self.anyadapter_metrics = {
            "world_model_loss": mean_wm_loss / num_updates,
            "world_model_loss_skipped": float(wm_loss_skipped),
            "adapter_delta_l2": mean_adapter_delta_l2 / num_updates,
            "adapter_reg_loss": mean_adapter_reg_loss / num_updates,
            "adapter_bias_reg_loss": mean_adapter_bias_reg_loss / num_updates,
            "stand_anchor_loss": mean_stand_anchor_loss / num_updates,
            "synthetic_stand_anchor_loss": mean_synthetic_stand_anchor_loss / num_updates,
            "stand_sample_ratio": mean_stand_sample_ratio / num_updates,
            "history_encoder_ppo_grad_norm": mean_history_encoder_ppo_grad_norm / num_updates,
            "history_encoder_wm_grad_norm": mean_history_encoder_wm_grad_norm / num_updates,
            "adapter_grad_norm": mean_adapter_grad_norm / num_updates,
            "max_abs_delta_action": mean_max_abs_delta_action / num_updates,
            "mean_abs_delta_action": mean_mean_abs_delta_action / num_updates,
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
        }
        self.update_counter()
        return (
            mean_value_loss / num_updates,
            mean_surrogate_loss / num_updates,
            0.0,
            mean_wm_loss / num_updates,
            mean_adapter_reg_loss / num_updates,
            0.0,
        )

    def update_dagger(self):
        """AnyAdapter does not use TWIST/RMA latent DAgger updates.

        OnPolicyRunnerMimic calls update_dagger() for every algorithm whose
        class name is not exactly "PPO".  The original implementation assumes
        actor_critic.actor exposes infer_priv_latent/infer_hist_latent, which
        is specific to the RMA actor.  AnyAdapter trains its history encoder
        through PPO plus the world-model auxiliary loss instead.
        """
        return 0.0
