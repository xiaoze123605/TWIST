import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import reference
from .geometry import finite, increments
from .networks import WorldModel, Critic, policy, DeploymentPolicy
from .replay import Replay


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def export_policy(actor, observations, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    actor.eval()
    traced = torch.jit.trace(DeploymentPolicy(actor), observations[:2])
    with torch.no_grad():
        maximum = float((traced(observations)-actor(observations)).abs().max())
    if not math.isfinite(maximum) or maximum > 1e-5:
        raise RuntimeError('JIT与Python控制动作不一致')
    temporary = output/'policy_jit.tmp.pt'
    traced.save(str(temporary))
    temporary.replace(output/'policy_jit.pt')
    save_json(output/'export.json', dict(metadata, observation_dimension=1155, action_dimension=23,
                                        compatible_input_dimensions=[1155, 7001, 7003],
                                        reference_offset_steps=0, jit_max_error=maximum,
                                        qualification='尚未完成最终策略验收'))
    return maximum


class Learner:
    def __init__(self, cfg, env, task, library):
        self.cfg, self.env, self.task, self.library = cfg, env, task, library
        self.output = Path(cfg.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.actor = policy().to(cfg.device)
        self.world = WorldModel(cfg.wm_width, cfg.amp).to(cfg.device)
        self.critic = Critic().to(cfg.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(.9, .95))
        self.opt_world = torch.optim.Adam(self.world.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.replay = Replay(cfg.buffer_steps, cfg.num_envs, cfg.device)
        self.iteration, self.model_updates, self.policy_updates, self.critic_updates = 0, 0, 0, 0
        self.elapsed = 0.
        self.collection_accumulator = 0.
        self.best_score = None
        self.degradation_count = 0
        save_json(self.output/'config.json', cfg.to_dict())

    @torch.no_grad()
    def collect(self, steps):
        for _ in range(steps):
            request = self.actor(self.env.obs)
            if self.cfg.action_noise:
                request = request+self.cfg.action_noise*torch.randn_like(request)
            finite('真实动作', request)
            row, _, _ = self.env.step(request)
            self.replay.add(row)

    def checked_step(self, module, optimizer, actor=False):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError('可训练网络没有收到梯度')
        norm = nn.utils.clip_grad_norm_(module.parameters(), 1.)
        finite('参数梯度', norm)
        if actor:
            nn.utils.clip_grad_value_(module.parameters(), .1)
        optimizer.step()
        return float(norm)

    def world_update(self):
        batch = self.replay.windows(self.cfg.wm_batch, self.cfg.wm_window)
        valid = batch['mask'].bool()
        self.world.update_norm(batch['state'][valid], batch['next_state'][valid], batch['executed'][valid], batch['context'][valid])
        self.opt_world.zero_grad(set_to_none=True)
        total = 0.
        denominator = batch['mask'].sum().clamp_min(1)
        for start in range(0, self.cfg.wm_batch, self.cfg.micro_batch):
            stop = min(start+self.cfg.micro_batch, self.cfg.wm_batch)
            state = batch['state'][start:stop, 0]
            losses = []
            for t in range(self.cfg.wm_window):
                state = self.world(state, batch['executed'][start:stop, t], batch['context'][start:stop, t])
                diff = increments(state, batch['next_state'][start:stop, t])/self.world.output_norm.std()
                losses.append(diff.square().mean(-1)*batch['mask'][start:stop, t])
            loss = torch.stack(losses, 1).sum()/denominator
            finite('世界模型损失', loss)
            loss.backward()
            total += float(loss.detach())
        norm = self.checked_step(self.world, self.opt_world)
        self.model_updates += 1
        return dict(world_loss=total, world_grad=norm)

    def imagined(self, starts, actor=None, horizon=None):
        actor = self.actor if actor is None else actor
        horizon = self.cfg.horizon if horizon is None else horizon
        state, obs = starts['state'], starts['full_obs']
        history, processor = starts['next_history'], starts['processor']
        previous_request, previous_action = starts['previous_request'], starts['previous_action']
        context = starts['context']
        ids, frames = starts['motion'].squeeze(-1), starts['frame'].squeeze(-1)
        feature = self.task.critic_features(state, self.library.state(ids, frames), obs, context, previous_action)
        features, states, references, actions, previous_actions = [feature], [], [], [], []
        for _ in range(horizon):
            request = actor(obs)
            executed = self.task.action(request, previous_request, context)
            state = self.world(state, executed, context)
            frames = frames+1
            ref = self.library.state(ids, frames)
            states.append(state)
            references.append(ref)
            actions.append(executed)
            previous_actions.append(previous_action)
            obs, history, processor = self.task.observe(state, request, reference(ref), history, processor)
            previous_request, previous_action = request, executed
            features.append(self.task.critic_features(state, ref, obs, context, previous_action))
        flat_state = torch.stack(states).flatten(0, 1)
        flat_reference = torch.stack(references).flatten(0, 1)
        rewards = self.task.reward(flat_state, flat_reference, torch.stack(actions).flatten(0, 1),
                                   torch.stack(previous_actions).flatten(0, 1)).reshape(horizon, len(state))
        dones = self.task.termination(flat_state, flat_reference, soft=True).reshape(horizon, len(state))
        return torch.stack(features), rewards, dones

    def policy_update(self):
        self.world.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt_actor.zero_grad(set_to_none=True)
        critic_features, rewards_all, dones_all = [], [], []
        total = 0.
        for start in range(0, self.cfg.pi_batch, self.cfg.micro_batch):
            n = min(self.cfg.micro_batch, self.cfg.pi_batch-start)
            starts = self.replay.starts(n)
            features, rewards, dones = self.imagined(starts)
            alive = torch.cumprod(1-dones, 0)
            continuation = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
            discounts = self.cfg.gamma**torch.arange(self.cfg.horizon, device=self.cfg.device)[:, None]
            objective = (discounts*continuation*rewards).sum(0)
            objective = objective+self.cfg.gamma**self.cfg.horizon*alive[-1]*self.critic.value(features[-1])
            loss = -objective.sum()/self.cfg.pi_batch
            finite('策略损失', loss)
            loss.backward()
            total += float(loss.detach())
            critic_features.append(features.detach())
            rewards_all.append(rewards.detach())
            dones_all.append(dones.detach())
        grad = self.checked_step(self.actor, self.opt_actor, actor=True)
        self.policy_updates += 1
        self.world.requires_grad_(True)
        self.critic.requires_grad_(True)
        value_loss = self.critic_update(torch.cat(critic_features, 1), torch.cat(rewards_all, 1), torch.cat(dones_all, 1))
        return dict(policy_loss=total, policy_grad=grad, critic_loss=value_loss)

    def critic_update(self, features, rewards, dones):
        self.critic.norm.update(features)
        with torch.no_grad():
            values = self.critic.value(features)
            target = values[-1]
            targets = []
            for t in range(self.cfg.horizon-1, -1, -1):
                target = rewards[t]+self.cfg.gamma*(1-dones[t])*((1-self.cfg.lam)*values[t+1]+self.cfg.lam*target)
                targets.append(target)
            targets = self.critic.targets(torch.stack(list(reversed(targets))).flatten())
        x = features[:-1].flatten(0, 1)
        loss_value = 0.
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(len(x), device=x.device)
            for indices in permutation.chunk(self.cfg.critic_minibatches):
                self.opt_critic.zero_grad(set_to_none=True)
                for chunk in indices.split(max(self.cfg.micro_batch, 128)):
                    loss = (self.critic(x[chunk])-targets[chunk]).square().sum()/len(indices)
                    finite('价值网络损失', loss)
                    loss.backward()
                    loss_value += float(loss.detach())
                self.checked_step(self.critic, self.opt_critic)
                self.critic_updates += 1
        return loss_value/(self.cfg.critic_epochs*self.cfg.critic_minibatches)

    def checkpoint(self, path=None):
        payload = dict(version=1, config=self.cfg.to_dict(), actor=self.actor.state_dict(), world=self.world.state_dict(),
                       critic=self.critic.state_dict(), opt_actor=self.opt_actor.state_dict(), opt_world=self.opt_world.state_dict(),
                       opt_critic=self.opt_critic.state_dict(), iteration=self.iteration, elapsed=self.elapsed,
                       physical_controls=self.env.physical_controls, model_updates=self.model_updates,
                       policy_updates=self.policy_updates, critic_updates=self.critic_updates,
                       collection_accumulator=self.collection_accumulator, best_score=self.best_score,
                       degradation_count=self.degradation_count, sampling_counts=self.library.bin_counts,
                       sampling_cost=self.library.bin_cost, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()),
                       dataset_names=[(r['name'], r['role'], r['frames']) for r in self.library.rows],
                       export_observations=self.env.obs[:16].detach().clone(),
                       resume_behavior='恢复网络和优化器；真实环境重置并重新采样缓存，继续记账')
        path = Path(path or self.output/'training_checkpoint.pt')
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def restore(self, path):
        payload = torch.load(path, map_location=self.cfg.device, weights_only=False)
        expected = [(r['name'], r['role'], r['frames']) for r in self.library.rows]
        if payload['dataset_names'] != expected:
            raise ValueError('恢复训练使用的动作划分与原检查点不同')
        for name in ('wm_width', 'wm_window', 'horizon', 'gamma', 'lam', 'observation_noise', 'domain_randomization'):
            if payload['config'][name] != getattr(self.cfg, name):
                raise ValueError('恢复训练不能改变'+name)
        for name in ('actor', 'world', 'critic', 'opt_actor', 'opt_world', 'opt_critic'):
            getattr(self, name).load_state_dict(payload[name])
        for name in ('iteration', 'elapsed', 'model_updates', 'policy_updates', 'critic_updates',
                     'collection_accumulator', 'best_score', 'degradation_count'):
            setattr(self, name, payload[name])
        self.env.physical_controls = payload['physical_controls']
        self.library.bin_counts.copy_(payload['sampling_counts'])
        self.library.bin_cost.copy_(payload['sampling_cost'])
        random.setstate(payload['rng']['python'])
        np.random.set_state(payload['rng']['numpy'])
        torch.set_rng_state(payload['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in payload['rng']['cuda']])

    def export(self):
        maximum = export_policy(self.actor, self.env.obs[:16].detach(), self.output,
                                dict(iteration=self.iteration, policy_updates=self.policy_updates))
        torch.save(dict(world=self.world.state_dict(), config=self.cfg.to_dict()), self.output/'world_model_checkpoint.pt')
        self.actor.train()
        return maximum

    def run(self, evaluate=None):
        started = time.monotonic()
        elapsed_before = self.elapsed
        self.collect(max(0, self.cfg.warmup_steps-self.replay.size))
        for iteration in range(self.iteration, self.cfg.iterations):
            tick = time.monotonic()
            lr = self.cfg.lr*(.03+.97*.5*(1+math.cos(math.pi*min(iteration/self.cfg.lr_decay_iters, 1))))
            for optimizer in (self.opt_actor, self.opt_world, self.opt_critic):
                for group in optimizer.param_groups:
                    group['lr'] = lr
            self.collection_accumulator += 2.5
            steps = int(self.collection_accumulator)
            self.collection_accumulator -= steps
            self.collect(steps)
            metrics = {}
            for _ in range(self.cfg.wm_batches):
                metrics.update(self.world_update())
            if iteration >= self.cfg.wm_only_iters:
                metrics.update(self.policy_update())
            self.iteration = iteration+1
            self.elapsed = elapsed_before+time.monotonic()-started
            row = dict(iteration=self.iteration, seconds=self.elapsed, physical_controls=self.env.physical_controls,
                       model_updates=self.model_updates, policy_updates=self.policy_updates, critic_updates=self.critic_updates,
                       iteration_seconds=time.monotonic()-tick, lr=lr, cuda_allocated=torch.cuda.memory_allocated(),
                       cuda_reserved=torch.cuda.memory_reserved(), **metrics)
            with (self.output/'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            save_json(self.output/'progress.json', row)
            if self.iteration % self.cfg.evaluate_every == 0 and evaluate is not None:
                result = evaluate(self)
                score = result['stability_cost']
                if self.best_score is None or score < self.best_score['stability_cost']:
                    self.best_score = result
                    self.degradation_count = 0
                    self.checkpoint(self.output/'best_checkpoint.pt')
                elif self.iteration >= 12000 and score > 1.1*self.best_score['stability_cost'] and result['fall_fraction'] > self.best_score['fall_fraction']:
                    self.degradation_count += 1
                else:
                    self.degradation_count = 0
                save_json(self.output/'evaluation_summary.json', result)
            if self.iteration % self.cfg.checkpoint_every == 0:
                self.checkpoint(self.output/('checkpoint_%06d.pt' % self.iteration))
                self.checkpoint()
                self.export()
            if self.degradation_count >= 3 or self.elapsed >= self.cfg.max_gpu_hours*3600:
                break
        self.elapsed = elapsed_before+time.monotonic()-started
        self.checkpoint()
        self.export()
        return dict(iteration=self.iteration, physical_controls=self.env.physical_controls,
                    model_updates=self.model_updates, policy_updates=self.policy_updates, seconds=self.elapsed)
