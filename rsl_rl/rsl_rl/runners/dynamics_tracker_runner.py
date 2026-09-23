"""Isolated runner with explicit checkpoint/config/deployment contracts."""
import json
import math
import hashlib
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
        self.extra_checkpoint_metadata = {}
        self.motion_coverage = torch.zeros(
            self.env._motion_lib.num_motions(), dtype=torch.bool, device=self.device)
        self.motion_manifest_sha256 = hashlib.sha256('\n'.join(
            str(Path(path).resolve()) for path in self.env._motion_lib._motion_files
        ).encode()).hexdigest()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("RSI sets reference phase; do not randomize episode counters")
        obs, critic = self.env.get_observations(), self.env.get_privileged_observations()
        self.alg.actor_critic.train()
        for _ in range(num_learning_iterations):
            reward_sum, dones_count = 0., 0
            physical_failures = motion_completions = timeouts = 0
            iteration_motion_ids = []
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    iteration_motion_ids.append(self.env._motion_ids.clone())
                    self.motion_coverage[self.env._motion_ids] = True
                    action = self.alg.act(obs, critic, {})
                    obs, critic, reward, done, info = self.env.step(action)
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("non-finite rollout")
                    self.alg.process_env_step(reward, done, info)
                    reward_sum += float(reward.mean())
                    dones_count += int(done.sum())
                    physical_failures += int(info['physical_failure'].sum())
                    motion_completions += int(info['motion_completed'].sum())
                    timeouts += int(info['time_outs'].sum())
                self.alg.compute_returns(critic)
            self.alg.update()
            self.current_learning_iteration += 1
            metrics = dict(self.alg.metrics, iteration=self.current_learning_iteration,
                           mean_reward=reward_sum / self.num_steps_per_env,
                           terminations=dones_count,
                           physical_failures=physical_failures,
                           motion_completions=motion_completions,
                           timeouts=timeouts,
                           motions_seen_iteration=int(torch.unique(
                               torch.cat(iteration_motion_ids)).numel()),
                           motions_seen_total=int(self.motion_coverage.sum()),
                           motion_count=int(self.motion_coverage.numel()),
                           mean_motion_difficulty=float(self.env.motion_difficulty.mean()))
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
                        critic_dim=self.env.num_privileged_obs,
                        environment_state=dict(
                            motion_manifest_sha256=self.motion_manifest_sha256,
                            motion_difficulty=self.env.motion_difficulty.detach().cpu(),
                            motion_coverage=self.motion_coverage.detach().cpu()),
                        metadata=dict(self.extra_checkpoint_metadata)), path)

    def load(self, path, load_optimizer=True, warm_start=False):
        checkpoint = torch.load(path, map_location=self.device)
        previous = dict(checkpoint["deployment_spec"])
        current = dict(self.spec)
        rescale_output = False
        if warm_start:
            previous.pop("use_dynamics_latent", None)
            current.pop("use_dynamics_latent", None)
            # A warm-start experiment may change whether the supplied goal is
            # the current sample or the command interval endpoint.
            previous.pop('reference_time_offset_steps', None)
            current.pop('reference_time_offset_steps', None)
            if previous.get('action_mode') == current.get('action_mode') == 'reference':
                rescale_output = previous.get('target_scales') != current.get('target_scales')
                previous.pop('target_scales', None)
                current.pop('target_scales', None)
        if previous != current:
            raise ValueError("checkpoint deployment contract differs from the current task")
        if not warm_start and checkpoint["train_cfg"]["policy"] != self.policy_cfg:
            raise ValueError("resume policy configuration differs")
        if not warm_start and checkpoint['train_cfg']['algorithm'] != self.alg_cfg:
            raise ValueError('resume algorithm configuration differs; use warm start for a new experiment')
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if rescale_output:
            old = torch.tensor(checkpoint['deployment_spec']['target_scales'], device=self.device)
            new = torch.tensor(self.spec['target_scales'], device=self.device)
            ratio = old / new
            with torch.no_grad():
                self.alg.actor_critic.actor[-1].weight.mul_(ratio[:, None])
                self.alg.actor_critic.actor[-1].bias.mul_(ratio)
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
            environment_state = checkpoint.get('environment_state', {})
            saved_manifest = environment_state.get('motion_manifest_sha256')
            if saved_manifest is not None and saved_manifest != self.motion_manifest_sha256:
                raise ValueError('resume motion manifest/order differs')
            difficulty = environment_state.get('motion_difficulty')
            coverage = environment_state.get('motion_coverage')
            if difficulty is not None:
                if difficulty.shape != self.env.motion_difficulty.shape:
                    raise ValueError('resume motion curriculum shape differs')
                self.env.motion_difficulty.copy_(difficulty.to(self.device))
                self.env.mean_motion_difficulty = self.env.motion_difficulty.mean()
            if coverage is not None:
                if coverage.shape != self.motion_coverage.shape:
                    raise ValueError('resume motion coverage shape differs')
                self.motion_coverage.copy_(coverage.to(self.device))
        # Warm start deliberately keeps new optimizers and curriculum clock.
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        return self.alg.actor_critic.act_inference
