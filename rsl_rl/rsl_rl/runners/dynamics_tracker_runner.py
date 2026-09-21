"""Isolated runner with explicit checkpoint/config/deployment contracts."""
import json
import math
from pathlib import Path

import torch

from rsl_rl.modules.dynamics_tracker import DynamicsTrackerActorCritic
from rsl_rl.algorithms.ppo_dynamics_tracker import PPODynamicsTracker


class DynamicsTrackerRunner:
    def __init__(self, env, train_cfg, log_dir=None, init_wandb=False, device="cpu", **kwargs):
        self.env, self.cfg, self.device = env, train_cfg, device
        if str(env.device) != str(device):
            raise ValueError("new tracker requires simulation and learner on the same device")
        self.log_dir = Path(log_dir) if log_dir else None
        self.policy_cfg, self.alg_cfg = train_cfg["policy"], train_cfg["algorithm"]
        self.spec = env.deployment_spec()
        self.spec["use_dynamics_latent"] = self.policy_cfg["use_dynamics_latent"]
        actor = DynamicsTrackerActorCritic(env.num_obs, env.num_privileged_obs,
                                           env.num_actions, **self.policy_cfg).to(device)
        if self.spec['action_mode'] == 'direct':
            lower = torch.tensor(self.spec['joint_lower'], device=device)
            upper = torch.tensor(self.spec['joint_upper'], device=device)
            initial = torch.tensor(self.spec['initial_target'], device=device)
            normalized = (2. * (initial - lower) / (upper - lower) - 1.).clamp(-.99, .99)
            with torch.no_grad():
                actor.actor[-1].bias.copy_(torch.atanh(normalized))
        self.alg = PPODynamicsTracker(env, actor, device=device, **self.alg_cfg)
        self.num_steps_per_env = train_cfg["runner"]["num_steps_per_env"]
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [env.num_obs],
                              [env.num_privileged_obs], [env.num_actions])
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("RSI sets reference phase; do not randomize episode counters")
        obs, critic = self.env.get_observations(), self.env.get_privileged_observations()
        self.alg.actor_critic.train()
        for _ in range(num_learning_iterations):
            reward_sum, dones_count = 0., 0
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    action = self.alg.act(obs, critic, {})
                    obs, critic, reward, done, info = self.env.step(action)
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("non-finite rollout")
                    self.alg.process_env_step(reward, done, info)
                    reward_sum += float(reward.mean())
                    dones_count += int(done.sum())
                self.alg.compute_returns(critic)
            self.alg.update()
            self.current_learning_iteration += 1
            metrics = dict(self.alg.metrics, iteration=self.current_learning_iteration,
                           mean_reward=reward_sum / self.num_steps_per_env,
                           terminations=dones_count)
            if not all(math.isfinite(float(v)) for v in metrics.values()):
                raise FloatingPointError("non-finite learning metrics")
            print(json.dumps(metrics, sort_keys=True))
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / "train_metrics.jsonl").open("a") as output:
                    output.write(json.dumps(metrics) + "\n")
                if self.current_learning_iteration % self.cfg["runner"]["save_interval"] == 0:
                    self.save(self.log_dir / ("model_%d.pt" % self.current_learning_iteration))
        if self.log_dir:
            self.save(self.log_dir / ("model_%d.pt" % self.current_learning_iteration))

    def save(self, path):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
                        optimizer_state_dict=self.alg.optimizer.state_dict(),
                        wm_optimizer_state_dict=self.alg.wm_optimizer.state_dict(),
                        iter=self.current_learning_iteration, train_cfg=self.cfg,
                        deployment_spec=self.spec,
                        critic_dim=self.env.num_privileged_obs), path)

    def load(self, path, load_optimizer=True, warm_start=False):
        checkpoint = torch.load(path, map_location=self.device)
        previous = dict(checkpoint["deployment_spec"])
        current = dict(self.spec)
        if warm_start:
            previous.pop("use_dynamics_latent", None)
            current.pop("use_dynamics_latent", None)
        if previous != current:
            raise ValueError("checkpoint deployment contract differs from the current task")
        if not warm_start and checkpoint["train_cfg"]["policy"] != self.policy_cfg:
            raise ValueError("resume policy configuration differs")
        if not warm_start and checkpoint['train_cfg']['algorithm'] != self.alg_cfg:
            raise ValueError('resume algorithm configuration differs; use warm start for a new experiment')
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if (warm_start and self.policy_cfg['use_dynamics_latent'] and
                not checkpoint['train_cfg']['policy']['use_dynamics_latent']):
            # Stage A saw zero latent inputs: these columns were never trained.
            # Start stage B with EXACTLY the same deterministic action function.
            with torch.no_grad():
                self.alg.actor_critic.actor[0].weight[:, -self.alg.actor_critic.latent_dim:] = 0
        if not warm_start:
            self.current_learning_iteration = int(checkpoint["iter"])
            self.alg.counter = self.current_learning_iteration
            if load_optimizer:
                self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.alg.wm_optimizer.load_state_dict(checkpoint["wm_optimizer_state_dict"])
        # Warm start deliberately keeps new optimizers and curriculum clock.
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        return self.alg.actor_critic.act_inference
