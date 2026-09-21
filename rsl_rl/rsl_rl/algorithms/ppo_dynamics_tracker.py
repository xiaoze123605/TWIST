"""PPO on the complete actor; separate multi-step dynamics representation loss."""
import torch
from torch import nn

from .ppo import PPO
from rsl_rl.modules.dynamics_tracker import STATE_DIM, NUM_JOINTS, append_history


class PPODynamicsTracker(PPO):
    skip_dagger_update = True

    def __init__(self, *args, world_model_learning_rate=1e-4,
                 world_model_sequence_length=20, world_model_num_epochs=1,
                 world_model_loss_coef=1.0, horizon_curriculum_updates=200,
                 world_model_batch_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = int(world_model_sequence_length)
        self.wm_epochs = int(world_model_num_epochs)
        self.wm_coef = float(world_model_loss_coef)
        self.horizon_curriculum_updates = int(horizon_curriculum_updates)
        self.wm_batch_size = int(world_model_batch_size)
        if self.horizon < 1 or self.wm_epochs < 1 or self.wm_batch_size < 1 or self.wm_coef < 0:
            raise ValueError("invalid dynamics training parameters")
        ac = self.actor_critic
        self.ppo_params = list(ac.actor.parameters()) + list(ac.critic.parameters()) + [ac.log_std]
        self.wm_params = list(ac.history_encoder.parameters()) + list(ac.world_model.parameters())
        assert not ({id(p) for p in self.ppo_params} & {id(p) for p in self.wm_params})
        self.optimizer = torch.optim.Adam(self.ppo_params, lr=self.learning_rate)
        self.wm_optimizer = torch.optim.Adam(self.wm_params, lr=world_model_learning_rate)
        self.metrics = {}

    def init_storage(self, num_envs, num_transitions_per_env, *args):
        if self.horizon > num_transitions_per_env:
            raise ValueError("world-model horizon exceeds rollout length")
        super().init_storage(num_envs, num_transitions_per_env, *args)
        shape = (num_transitions_per_env, num_envs)
        self.dyn_state = torch.zeros(*shape, STATE_DIM, device=self.device)
        self.dyn_next = torch.zeros_like(self.dyn_state)
        self.dyn_command = torch.zeros(*shape, NUM_JOINTS, device=self.device)

    def process_env_step(self, rewards, dones, infos):
        step = self.storage.step
        # Required, explicit simulator targets. Never slice masked/noisy actor obs.
        for key, dest in (("dynamics_state", self.dyn_state),
                          ("dynamics_next_state", self.dyn_next),
                          ("sent_target", self.dyn_command)):
            value = infos[key]
            if value.shape != dest[step].shape or not torch.isfinite(value).all():
                raise ValueError("invalid dynamics transition: " + key)
            dest[step].copy_(value)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        timeouts = infos["time_outs"].bool()
        if torch.any(timeouts):
            # Bootstrap the state BEFORE reset, not the value of state_t.
            with torch.no_grad():
                terminal_value = self.actor_critic.evaluate(infos["terminal_critic_observation"]).squeeze(-1)
            self.transition.rewards += self.gamma * terminal_value * timeouts
        self.transition.timeouts = timeouts
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        return rewards

    def sequence_loss(self, history, state, commands, targets, dones):
        """Train terminal transitions, then mask the remainder of that episode.

        h_(k+1) receives (s_k, command_k), NOT (s_(k+1), command_k).
        No reset state or future measured state is fed into the prediction.
        """
        ids = torch.arange(state.shape[0], device=state.device)
        loss_sum, count = state.new_zeros(()), state.new_zeros(())
        for k in range(commands.shape[1]):
            if ids.numel() == 0:
                break
            latent = self.actor_critic.history_encoder(history)
            predicted = self.actor_critic.world_model(state, commands[ids, k], latent)
            error = (predicted - targets[ids, k]).abs()
            gravity_loss = 1. - (predicted[:, 3:6] * targets[ids, k, 3:6]).sum(-1).clamp(-1., 1.)
            # Equal group weights in declared sensor units; logged/ablatable,
            # deliberately independent of old 51-D adapter loss weights.
            per_sample = (error[:, :3].mean(-1) + gravity_loss +
                          error[:, 6:29].mean(-1) + error[:, 29:].mean(-1))
            loss_sum = loss_sum + per_sample.sum()
            count = count + ids.numel()
            history = append_history(history, state, commands[ids, k])
            alive = ~dones[ids, k].bool()
            history, state, ids = history[alive], predicted[alive], ids[alive]
        return loss_sum / count.clamp_min(1.), count

    def update_world_model(self):
        if self.wm_coef == 0:
            return 0., 0., 0
        horizon = self.horizon
        if self.horizon_curriculum_updates > 0:
            fraction = min(1., self.counter / self.horizon_curriculum_updates)
            horizon = 1 + int((self.horizon - 1) * fraction)
        total, valid, updates = 0., 0., 0
        T, E = self.dyn_state.shape[:2]
        for _ in range(self.wm_epochs):
            order = torch.randperm(E, device=self.device)
            for start in range(0, E, self.wm_batch_size):
                ids = order[start:start + self.wm_batch_size]
                begins = torch.randint(T - horizon + 1, (ids.numel(),), device=self.device)
                times = begins[:, None] + torch.arange(horizon, device=self.device)[None]
                _, history = self.actor_critic.split_obs(self.storage.observations[begins, ids])
                loss, count = self.sequence_loss(
                    history, self.dyn_state[begins, ids], self.dyn_command[times, ids[:, None]],
                    self.dyn_next[times, ids[:, None]], self.storage.dones[times, ids[:, None], 0])
                self.wm_optimizer.zero_grad(set_to_none=True)
                (self.wm_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.wm_params, self.max_grad_norm)
                self.wm_optimizer.step()
                total += float(loss.detach())
                valid += float(count.detach())
                updates += 1
        return total / max(updates, 1), valid, horizon

    def update(self):
        # Keep the rollout encoder fixed through every PPO epoch; update its
        # representation only AFTER PPO, before collecting the next rollout.
        self.wm_optimizer.zero_grad(set_to_none=True)
        values = super().update()  # clear() resets cursors, not stored samples
        wm_loss, valid, horizon = self.update_world_model()
        with torch.no_grad():
            self.actor_critic.log_std.clamp_(-3.5065579, -0.3566749)
        self.counter += 1
        self.metrics = {"value_loss": values[0], "policy_loss": values[1],
                        "world_model_loss": wm_loss, "world_model_valid_steps": valid,
                        "world_model_horizon": horizon,
                        "mean_action_std": float(self.actor_critic.std.mean())}
        return values
