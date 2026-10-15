"""Original-distribution matched control; at most50 updates with read-only diagnostics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'legged_gym'), str(ROOT/'rsl_rl'), str(ROOT/'pose')]
import isaacgym  # must precede torch
import torch
import wandb
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import task_registry
from legged_gym.envs.g1.g1_motion_wm_anyadapter import G1MotionWMAnyAdapter
from legged_gym.gym_utils import get_args, class_to_dict
import rsl_rl.runners.on_policy_runner_mimic as runner_module
from train_4900_improved import PPO4900Improved
from recovery1250_core import StratifiedReplay, anchor_error
from heading300_teacher import complete_teacher_action
from heading300_training_diagnostics import critic_before,critic_after,gradient_groups,append_gradients
from heading300_frozen_head_diagnostics import objective_gradients

DEFAULT_SOURCE = ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'


def resume_training_signature(training):
    # Checkpoint cadence and the run budget do not change learning semantics.
    # Keep all policy, optimizer, rollout and reward settings strict.
    signature = json.loads(json.dumps(training))
    for key in ('save_interval', 'max_iterations'):
        signature.get('runner', {}).pop(key, None)
    return signature


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CoordinatedEnv(G1MotionWMAnyAdapter):
    def _in_place_reference_mask(self):
        # A stationary root does not imply stationary legs (kicks, steps,
        # squats). Apply the inherited stillness costs only to still legs.
        return super()._in_place_reference_mask() & (self._ref_dof_vel[:, :12].abs().amax(-1) < .2)

    def _reward_support_slip(self):
        # Only actual contact feet: swinging legs remain unrestricted.
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        speed2 = self.rigid_body_states[:, self.feet_indices, 7:9].square().sum(-1)
        turn = self._ref_root_ang_vel[:, 2].abs().clamp(max=1.)
        return (speed2.clamp(max=4.) * contact).sum(-1) * (1. + turn)

    def _reward_reference_tilt_excess(self):
        # Follow the motion's intended lean; do not demand an upright torso.
        target = quat_rotate_inverse(self._ref_root_rot, self.gravity_vec)
        error = (self.projected_gravity - target).norm(dim=-1)
        return (error - .15).clamp(min=0., max=1.).square()

    def _reward_turn_velocity_error(self):
        target_v = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        target_w = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        v = (self.base_lin_vel[:, :2] - target_v[:, :2]).norm(dim=-1)
        w = (self.base_ang_vel[:, 2] - target_w[:, 2]).abs()
        # Smooth bounded error avoids exp-reward saturation at larger errors.
        turn = target_w[:, 2].abs().clamp(max=1.)
        return (v.clamp(max=2.).square() + .25*w.clamp(max=2.).square()) * (1. + turn)


class PPOHeading300FrozenHead(PPO4900Improved):
    def __init__(self, *args, source_checkpoint, fixed_anchor_scale=.087,
                 replay_capacity_per_group=4096, replay_insert_per_group=128,
                 tail_anchor_coef=.01, **kwargs):
        super().__init__(*args, source_checkpoint=source_checkpoint, **kwargs)
        if fixed_anchor_scale <= 0 or tail_anchor_coef < 0:
            raise ValueError('invalid anchor settings')
        self.source_hash = sha(source_checkpoint)
        self.fixed_anchor_scale = float(fixed_anchor_scale)
        self.tail_anchor_coef = float(tail_anchor_coef)
        self.replay = StratifiedReplay(replay_capacity_per_group, replay_insert_per_group)
        self.rollout_modes = []
        self.rollout_sampling = []
        self._diag_timeouts = []
        self.environment = None
        self.actor_critic.adapter[5].requires_grad_(False)
        self.initial_heading={k:v.detach().clone() for k,v in self.actor_critic.adapter[5].state_dict().items()}

    def _transition_weights(self, obs_steps):
        return super()._transition_weights(obs_steps[..., :7001])

    def act(self, *args, **kwargs):
        self.rollout_sampling.append(sampling_coverage(self.environment))
        self.rollout_modes.append(self.environment.motion_reference_pipeline.mode.detach().clone())
        return super().act(*args, **kwargs)

    def process_env_step(self,rewards,dones,infos):
        self._diag_timeouts.append(infos.get("time_outs",torch.zeros_like(dones)).detach().clone())
        return super().process_env_step(rewards,dones,infos)

    def compute_returns(self,last_critic_obs):
        with torch.no_grad():
            self._diag_last_critic_obs=last_critic_obs.detach().clone()
            self._diag_last_values=self.actor_critic.evaluate(last_critic_obs).detach()
        return super().compute_returns(last_critic_obs)

    def update(self):
        critic_before(self)
        gradient_records=[];clip_norms=[]
        actor = self.actor_critic
        obs_steps = self.storage.observations
        flat = obs_steps.flatten(0, 1)
        if len(self.rollout_modes) != obs_steps.shape[0]:
            raise RuntimeError('reference modes do not align with rollout')
        modes = torch.stack(self.rollout_modes).reshape(-1)
        self.rollout_modes.clear()
        # Keep a unit floor for steady motion instead of weakening it by
        # dividing transition weights by a minibatch mean.
        weights = self._transition_weights(obs_steps).clamp(1., 3.)
        speed = flat[:, 4:6].norm(dim=-1)
        turn = flat[:, 7].abs()
        phase = torch.full_like(modes, 2)
        phase[(speed < .12) & (turn < .12)] = 0
        phase[(turn > .6) | (weights > 1.5)] = 1
        strata = (modes == 2).long()*3 + phase
        with torch.no_grad():
            # Bound encoder activation memory for the 4096 x 24 rollout.
            base = flat[:, :1155]
            latent_chunks, target_chunks = [], []
            for chunk in flat.split(4096):
                chunk_base, hist = actor.split_obs(chunk)
                chunk_z = actor.history_encoder(hist)
                latent_chunks.append(chunk_z)
                target_chunks.append(complete_teacher_action(self.source, chunk_base, chunk_z, chunk[:, -2:]))
            z = torch.cat(latent_chunks)
            targets = torch.cat(target_chunks)
            replay_inputs = torch.cat((base, z, flat[:, -2:]), -1)
            self.replay.add(replay_inputs, targets, weights, strata)
        self.ppo_optimizer.param_groups[0]['lr'] = self._actor_lr()
        coef = self._anchor_coef()
        totals = dict(value=0., surrogate=0., anchor=0., tail=0., mae=0.)
        steps = skipped = 0
        max_kl = 0.
        for idx, sample in self._mini_batch_generator():
            obs, critic, actions, old_values, advantages, returns, old_logp, old_mu, old_std, _, _ = sample
            mu = actor.layerwise_actor(base[idx], z[idx]) + actor.heading_delta(z[idx], flat[idx, -2:])
            with torch.no_grad():
                kl = (torch.log(actor.std/old_std) + (old_std.square()+(old_mu-mu).square())/(2*actor.std.square())-.5).sum(-1).mean()
                max_kl = max(max_kl, float(kl))
            if kl > .012:
                skipped += 1
                continue
            logp = torch.distributions.Normal(mu, actor.std).log_prob(actions).sum(-1)
            ratio = (logp-old_logp.squeeze(-1)).exp()
            adv = advantages.squeeze(-1)
            surrogate = torch.maximum(-adv*ratio, -adv*ratio.clamp(.9, 1.1)).mean()
            value = actor.evaluate(critic)
            clipped = old_values + (value-old_values).clamp(-.2, .2)
            value_loss = torch.maximum((value-returns).square(), (clipped-returns).square()).mean()
            errors = anchor_error(mu, targets[idx], self.fixed_anchor_scale, weights[idx])
            memory, teacher, memory_w = self.replay.sample()
            memory_mu = actor.layerwise_actor(memory[:, :1155], memory[:, 1155:1283]) + actor.heading_delta(memory[:, 1155:1283], memory[:, -2:])
            memory_errors = anchor_error(memory_mu, teacher, self.fixed_anchor_scale, memory_w)
            baseline_loss = mu.new_zeros(())
            if self.baseline_anchor_ratio > 0:
                with torch.no_grad():
                    base_teacher = actor.layerwise_actor.base_forward(memory[:, :1155])
                baseline_loss = anchor_error(memory_mu, base_teacher, self.fixed_anchor_scale, memory_w).mean()

            anchor = errors.mean() + memory_errors.mean()
            # Soft tail cost, not a hard rejection of nonmonotonic progress.
            tail = errors.topk(max(1, math.ceil(.1*len(errors)))).values.mean()
            loss = surrogate + self.value_loss_coef*value_loss + coef*anchor + self.tail_anchor_coef*tail + coef*self.baseline_anchor_ratio*baseline_loss
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            if steps==0 and self.count>=self.critic_warmup_updates:
                objective_gradients(self,surrogate,coef*anchor+self.tail_anchor_coef*tail)
            self.optimizer.zero_grad()
            loss.backward()
            if self.count < self.critic_warmup_updates:
                for parameter in actor.adapter.parameters():
                    parameter.grad = None
            gradient_records.append(gradient_groups(actor))
            clip_norms.append(float(torch.nn.utils.clip_grad_norm_(self.ppo_params, .5)))
            self.optimizer.step()
            for key, val in [('value', value_loss), ('surrogate', surrogate), ('anchor', anchor), ('tail', tail), ('mae', (mu-targets[idx]).abs().mean())]:
                totals[key] += float(val.detach())
            steps += 1
        if not steps:
            raise RuntimeError('all PPO minibatches skipped')
        with torch.no_grad():
            for key, value in actor.adapter.state_dict().items():
                self.ema[key].mul_(self.ema_decay).add_(value, alpha=1-self.ema_decay)
        if not all(torch.equal(v,self.actor_critic.adapter[5].state_dict()[k]) for k,v in self.initial_heading.items()):
            raise RuntimeError("frozen heading parameters changed")
        critic_after(self)
        append_gradients(self,gradient_records,clip_norms)
        self.storage.clear()
        self.count += 1
        # Fixed exploration by default. Optional annealing does not alter
        # the fixed physical scale of the anchor objective.
        if self.std_anneal_start >= 0 and self.count >= self.std_anneal_start:
            with torch.no_grad():
                actor.std.mul_(.999).clamp_(min=self.std_floor)
        if not all(torch.equal(actor.state_dict()[k], v) for k, v in self.initial_frozen.items()):
            raise RuntimeError('frozen source parameters changed')
        self.anyadapter_metrics = dict(source_action_mae=totals['mae']/steps,
            fixed_scale_anchor_loss=totals['anchor']/steps, tail_anchor_loss=totals['tail']/steps,
            max_rollout_kl=max_kl, skipped_minibatches=skipped, std_mean=float(actor.std.mean()),
            fixed_anchor_scale=self.fixed_anchor_scale, anchor_coef_effective=coef,
            baseline_anchor_ratio=self.baseline_anchor_ratio,
            actor_lr=self._actor_lr(), world_model_loss=0.,
            reference_stance_mean=float(self.environment.ref_stance_confidence.mean()),
            double_stance_ratio=float((self.environment.ref_stance_confidence.min(-1).values>.5).float().mean()),
            stance_in_place_ratio=float(self.environment._in_place_reference_mask().float().mean()),
            **{'replay_group_'+str(g): n for g, n in self.replay.counts().items()})
        self.anyadapter_metrics.update({'sampling_'+key:sum(r[key] for r in self.rollout_sampling)/len(self.rollout_sampling) for key in self.rollout_sampling[0]})
        self.rollout_sampling.clear()
        if hasattr(self, 'audit_path'):
            with self.audit_path.open('a') as f:
                f.write(json.dumps(dict(update=self.count, **self.anyadapter_metrics))+'\n')
        self.update_counter()
        return totals['value']/steps, totals['surrogate']/steps, 0., 0., totals['anchor']/steps, 0.

    def recovery_state(self):
        return dict(version=4, frozen_heading=True, complete_heading_teacher=True, sampling_sha256=self.sampling_sha256, heading_context_dim=2, source_sha256=self.source_hash, count=self.count,
                    fixed_anchor_scale=self.fixed_anchor_scale,
                    adapter_gain=self.actor_critic.layerwise_actor.adapter_gain,
                    baseline_anchor_ratio=self.baseline_anchor_ratio,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    ema={k: v.cpu() for k, v in self.ema.items()}, replay=self.replay.state_dict())

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('recovery1250')
        if not state or not state.get('frozen_heading') or not state.get('complete_heading_teacher') or state.get('sampling_sha256')!=self.sampling_sha256:
            raise ValueError('sampling/complete-teacher checkpoint mismatch')
        if state is None or state.get('heading_context_dim') != 2 or state['source_sha256'] != self.source_hash:
            raise ValueError('resume requires a coordinated checkpoint with the same23500 reference')
        if state.get('adapter_gain') != self.actor_critic.layerwise_actor.adapter_gain or state.get('baseline_anchor_ratio') != self.baseline_anchor_ratio:
            raise ValueError('resume gain or baseline teacher ratio differs')
        if state['fixed_anchor_scale'] != self.fixed_anchor_scale:
            raise ValueError('resume anchor scale differs')
        if state['count'] != checkpoint['iter']:
            raise ValueError('checkpoint counters differ')
        self.count = int(state['count'])
        self.ema = {k: v.to(self.device) for k, v in state['ema'].items()}
        self.replay.load_state_dict(state['replay'], self.device)
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])


def sampling_coverage(env):
    h=env._ref_root_pos[:,2]-env.episode_init_origin[:,2]
    squat=(h<.7)&(env._ref_dof_pos[:,[3,9]].mean(-1)>.8)
    confidence=getattr(env,'ref_stance_confidence',torch.zeros((env.num_envs,2),device=env.device))
    support=confidence.min(-1).values>.5
    actual=env.contact_forces[:,env.feet_indices,2].min(-1).values>5.
    return {k:float(v.float().mean()) for k,v in dict(squat=squat,squat_stance=squat&support,squat_actual_stance=squat&support&actual,walking=(h>.74)&(env._ref_root_vel[:,:2].norm(dim=-1)>.15),turning=env._ref_root_ang_vel[:,2].abs()>.5).items()}

def verify_initialization(alg,source,env,out):
    stored=torch.load(source,map_location=env.device)['model_state_dict']
    assert all(torch.equal(alg.actor_critic.state_dict()[k],v) for k,v in stored.items())
    obs=env.get_observations()
    assert obs.shape[-1]==7003
    critic=env.get_privileged_observations()
    assert critic.shape[-1]==1318
    with torch.no_grad():
        base,hist=alg.actor_critic.split_obs(obs);z=alg.actor_critic.history_encoder(hist)
        actual=alg.actor_critic.actor_mean(obs)
        target=complete_teacher_action(alg.source,base,z,obs[:,-2:])
        error=float((actual-target).abs().max())
        omitted_heading_error=float((target-alg.source.layerwise_actor(base,z)).abs().max())
    assert error==0. and actual.shape[-1]==23
    assert alg.actor_critic.layerwise_actor.adapter_gain==.5
    assert all(not p.requires_grad for p in alg.source.parameters())
    (out/'initialization.json').write_text(json.dumps(dict(input_dim=7003,critic_input_dim=1318,output_dim=23,adapter_gain=alg.actor_critic.layerwise_actor.adapter_gain,teacher_max_abs_error=error,omitted_heading_max_abs_error=omitted_heading_error,teacher_frozen=True,source_state_exact=True),indent=2))

def run_sampling_audit(actor,env,steps,out):
    records=[];obs=env.get_observations()
    with torch.no_grad():
        for step in range(steps):
            obs=env.step(actor.actor_mean(obs))[0]
            records.append(sampling_coverage(env))
            if step%256==0:print('Sampling audit',step,flush=True)
    stats={k:sum(x[k] for x in records)/len(records) for k in records[0]}
    stats.update(frames=steps*env.num_envs,steps=steps,optimizer_updates=0)
    (out/'sampling_audit.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))

def guarded_training(runner,alg,total,interval,out):
    from sim_auto_eval import AutoEvalHook
    hook=AutoEvalHook(enabled=True,interval=10,output_root=out/'evaluations',mode='standard',cases=['mocap1__77'])
    while alg.count<total:
        target=next(x for x in (30,40,50) if x>alg.count)
        runner.learn(num_learning_iterations=target-alg.count,init_at_random_ep_len=False)
        checkpoint=out/f'model_{alg.count}.pt'
        runner.save(checkpoint)
        event=hook.on_checkpoint_saved(checkpoint,alg.count)
        if event['status']!='started':
            (out/'STOPPED.json').write_text(json.dumps(dict(update=alg.count,reason='required evaluation did not start',event=event),indent=2));return
        print('Training paused for CPU evaluation',event['output'],flush=True)
        result=None
        while hook.worker is not None:
            try:hook.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:continue
            result=hook.poll()
        if not result or result['status']!='finished':
            (out/'STOPPED.json').write_text(json.dumps(dict(reason='evaluation failed',event=result),indent=2));return
        report=json.loads((Path(event['output'])/'summary.json').read_text())
        comparison=report['comparisons']['checkpoint']
        if not comparison['gates'].get('yaw_within_5_percent') or not comparison['gates'].get('no_new_falls'):
            (out/'STOPPED.json').write_text(json.dumps(dict(update=alg.count,reason='yaw>5% regression or new fall',evaluation=event['output'],comparison=comparison),indent=2));return
        print('Evaluation guard passed; continue to',min(target+interval,total),flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-checkpoint', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--actor-rate', type=float, default=5e-7)
    parser.add_argument('--actor-rate-end', type=float, default=2e-7)
    parser.add_argument('--anchor-coef', type=float, default=.04)
    parser.add_argument('--anchor-coef-start', type=float, default=.08)
    parser.add_argument('--fixed-anchor-scale', type=float, default=.087)
    parser.add_argument('--tail-anchor-coef', type=float, default=.01)
    parser.add_argument('--critic-warmup-updates', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--std-anneal-start', type=int, default=-1)
    parser.add_argument('--baseline-anchor-ratio', type=float, default=0.)
    parser.add_argument('--adapter-gain', type=float, default=.5)
    parser.add_argument('--heading-rate', type=float, default=5e-6)
    parser.add_argument('--sampling-config',type=Path,default=ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--audit-steps',type=int,default=1024)
    parser.add_argument('--auto-eval',action='store_true')
    parser.add_argument('--eval-interval',type=int,default=10)
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.seed is None:
        args.seed = 42
    if args.resume or args.resumeid:
        raise ValueError('use --resume-from only for checkpoints from this script')
    if args.max_iterations < 1 or custom.save_every < 1:
        raise ValueError('iterations and save interval must be positive')
    if custom.eval_interval!=10 or (custom.auto_eval and args.max_iterations!=50):
        raise ValueError('auto-eval budget must be a positive multiple of the evaluation interval')
    if not 0. <= custom.baseline_anchor_ratio <= .5 or not 0. < custom.adapter_gain <= 1.:
        raise ValueError('invalid baseline ratio or adapter gain')
    if custom.heading_rate <= 0:
        raise ValueError('heading learning rate must be positive')
    if custom.adapter_gain!=.5 or custom.baseline_anchor_ratio!=0.:raise ValueError('sampling-only branch requires gain0.5, no new baseline teacher')
    if args.max_iterations>50:raise ValueError('original control capped at50 updates')
    original_sampling=ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml'
    if custom.sampling_config.resolve()!=original_sampling.resolve():raise ValueError('control requires original distribution')
    source = custom.source_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cfg, train = task_registry.get_cfgs('g1_motion_wm_anyadapter')
    args.task = 'g1_heading300_frozen_head'
    args.headless = True
    args.no_wandb = True
    args.fix_action_std = True
    cfg.seed = train.seed = args.seed
    cfg.env.num_envs = args.num_envs
    cfg.motion.motion_file = str(custom.sampling_config.resolve())
    cfg.motion.motion_curriculum = False
    cfg.motion_wm.mode_probabilities = (.5, 0., .5)
    cfg.motion_wm.artificial_corruption = False
    cfg.motion_wm.prefill_history = False  # match deployed refiner warmup
    cfg.rewards.scales.support_slip = -.25
    cfg.rewards.scales.reference_tilt_excess = -1.
    cfg.rewards.scales.turn_velocity_error = -.15
    cfg.control.projection_margin = .02
    cfg.control.projection_knee_min = 0.
    cfg.control.target_projection = True
    cfg.rewards.scales.reference_stance_slip = -.5
    cfg.rewards.scales.hip_yaw_tracking = -.3
    cfg.rewards.scales.squat_knee_tracking = -.3
    cfg.rewards.scales.target_projection = -1.
    cfg.rewards.scales.in_place_feet_motion = 0.
    cfg.env.anyadapter_context_dim = 2
    cfg.env.anyadapter_added_obs_dim = 79*74+2
    cfg.env.anyadapter_final_policy_obs_dim = 7003
    cfg.rewards.scales.heading_tracking = -.5
    train.runner.policy_class_name = 'TwistHeadingSupportActorCritic'
    train.algorithm.heading_rate = custom.heading_rate
    train.policy.fix_action_std = True
    train.runner.algorithm_class_name = 'PPOHeading300FrozenHead'
    train.runner.save_interval = custom.save_every
    train.runner.constant_save_interval = True
    train.runner.resume = False
    train.algorithm.source_checkpoint = str(source)
    for key in ('actor_rate', 'actor_rate_end', 'anchor_coef', 'anchor_coef_start',
                'fixed_anchor_scale', 'tail_anchor_coef', 'critic_warmup_updates', 'std_anneal_start'):
        setattr(train.algorithm, key, getattr(custom, key))
    train.algorithm.lr_decay_steps = 2000
    train.algorithm.anchor_decay_steps = 500
    train.algorithm.transition_anchor_scale = 2.
    train.algorithm.transition_anchor_dev_max = .4
    train.algorithm.transition_anchor_turn_max = 1.5
    train.algorithm.ema_decay = .99
    train.algorithm.num_learning_epochs = 3
    train.algorithm.num_mini_batches = 16
    train.algorithm.world_model_loss_coef = 0.
    train.algorithm.entropy_coef = 0.
    out = ROOT/'legged_gym/logs/recovery_heading300_yaw_controls'/args.exptid
    out.mkdir(parents=True, exist_ok=True)
    if custom.resume_from is None and list(out.glob('model_*.pt')):
        raise ValueError('run already has checkpoints; choose another exptid or resume')
    manifest = dict(source=str(source), source_sha256=sha(source),
                    custom={k: str(v) if isinstance(v, Path) else v for k, v in vars(custom).items()},
                    environment=class_to_dict(cfg), training=class_to_dict(train),
                    seed=args.seed, code={str(p.relative_to(ROOT)): sha(p) for p in
                    [Path(__file__).resolve(), ROOT/'tools/heading300_teacher.py', ROOT/'tools/heading300_training_diagnostics.py', ROOT/'tools/heading300_frozen_head_diagnostics.py', ROOT/'tools/support23500_env.py', ROOT/'tools/heading23500_env.py', ROOT/'tools/heading23500_actor.py', ROOT/'deploy_real/joint_target_projection.py', ROOT/'tools/recovery1250_core.py', ROOT/'tools/train_4900_improved.py', ROOT/'tools/train_4900_rawguard.py']})
    if custom.resume_from:
        previous = json.loads((out/'manifest.json').read_text())
        for key in ('source_sha256', 'environment', 'training', 'code'):
            before, after = previous[key], manifest[key]
            if key == 'training':
                before, after = resume_training_signature(before), resume_training_signature(after)
            if json.dumps(before, sort_keys=True, default=str) != json.dumps(after, sort_keys=True, default=str):
                raise ValueError('resume configuration changed: '+key)
    else:
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    # Compare all original environment settings except explicitly declared sampling/resources.
    original=json.loads((DEFAULT_SOURCE.parent/'manifest.json').read_text())
    current=json.loads(json.dumps(class_to_dict(cfg),default=str))
    before=json.loads(json.dumps(original['environment']))
    before['motion']['motion_file']=current['motion']['motion_file']
    before['env']['num_envs']=current['env']['num_envs']
    before['seed']=current['seed']
    if before!=current:raise ValueError('non-sampling environment configuration changed')
    runner_module.PPOHeading300FrozenHead = PPOHeading300FrozenHead
    from heading23500_actor import TwistHeadingSupportActorCritic
    from heading23500_env import Heading23500Env
    runner_module.TwistHeadingSupportActorCritic = TwistHeadingSupportActorCritic
    task_registry.register(args.task, Heading23500Env, cfg, train)
    wandb.init(mode='disabled')
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train, log_root=str(out))
    alg = runner.alg
    head_params = list(alg.actor_critic.adapter[5].parameters())
    head_ids = {id(p) for p in head_params}
    alg.optimizer.param_groups[0]['params'] = [p for p in alg.optimizer.param_groups[0]['params'] if id(p) not in head_ids]
    alg.optimizer.add_param_group(dict(params=head_params, lr=custom.heading_rate))
    alg.environment = env
    alg.baseline_anchor_ratio = custom.baseline_anchor_ratio
    alg.actor_critic.layerwise_actor.adapter_gain = custom.adapter_gain
    alg.source.layerwise_actor.adapter_gain = custom.adapter_gain
    alg.audit_path = out/'anchor_metrics.jsonl'
    alg.sampling_sha256=sha(custom.sampling_config)
    if custom.resume_from:
        runner.load(str(custom.resume_from.resolve()))
    else:
        # Fresh optimizer, EMA and reservoir intentionally start at zero new
        # updates. Never load bank_state_latest.pt from the source directory.
        runner.current_learning_iteration = 0
    def save(path, infos=None, iteration=None):
        state = dict(model_state_dict=alg.actor_critic.state_dict(),
                     optimizer_state_dict=alg.optimizer.state_dict(),
                     ppo_optimizer_state_dict=alg.ppo_optimizer.state_dict(),
                     iter=alg.count, infos=infos, recovery1250=alg.recovery_state())
        # Live weights and optimizer belong together. EMA is stored separately
        # and can be evaluated/exported without corrupting resume semantics.
        tmp = str(path)+'.tmp'
        torch.save(state, tmp)
        os.replace(tmp, path)
    runner.save = save
    if custom.resume_from is None:
        runner.save(out/'model_0.pt')
        verify_initialization(alg,source,env,out)
    if custom.audit_only:
        run_sampling_audit(alg.actor_critic,env,custom.audit_steps,out)
        return
    stop = {'requested': False}
    def request_stop(*unused):
        stop['requested'] = True
        print('Stop requested; saving at the next completed update.', flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    original_log = runner.log
    def log_completed(locs, *a, **kw):
        original_log(locs, *a, **kw)
        if stop['requested']:
            runner.save(out/f'model_{alg.count}.pt')
            raise SystemExit(0)
    runner.log = log_completed
    remaining_updates = args.max_iterations - alg.count
    if remaining_updates <= 0:
        print('Requested additional-update budget already complete.')
        return
    try:
        if custom.auto_eval:
            guarded_training(runner,alg,args.max_iterations,custom.eval_interval,out)
        else:
            runner.learn(num_learning_iterations=remaining_updates, init_at_random_ep_len=False)
    except Exception as exc:
        (out/'STOPPED.json').write_text(json.dumps(dict(update=alg.count,reason='training or evaluation anomaly',error=repr(exc)),indent=2))
        raise


if __name__ == '__main__':
    main()
