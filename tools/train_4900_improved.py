"""Improved continuation of the 4900 source-anchored pilot, fresh or resumed.

Keeps the pilot's frozen-source guards (TWIST backbone / history encoder /
world model verified unchanged after every update) and its PPO mechanics
(anchor + replay bank, KL skip at 0.012, clip 0.1, critic warmup), and adds:

1. Anchor schedule: linear from --anchor-coef-start to --anchor-coef over
   --anchor-decay-steps updates (strong early anchor, proven-good 0.1 later).
2. Adapter EMA (--ema-decay) with snapshot-on-save: every checkpoint carries
   the exponentially-smoothed adapter, reducing per-case gate variance.
3. Replay bank FIFO refresh (never stale) plus sidecar persistence (*.bank.pt)
   so --resume-from restores bank and EMA state.
4. Actor LR cosine schedule toward --actor-rate-end over --lr-decay-steps.
5. Std anneal after --std-anneal-start updates toward --std-floor, narrowing
   the train/eval gap (eval runs the deterministic mean).
6. Motion-WM history prefill at reset (cfg.motion_wm.prefill_history) so
   wm-mode envs are GRU-ready from frame 0 instead of jumping at frame 25.
7. Transition-aware anchor: the source-anchor (and replay) loss is multiplied
   by 1+scale*max((|v-EMA|/dev_max)^2,(|yaw_rate|/turn_max)^2) where EMA is a
   ~1s reference-speed filter walked along each rollout trajectory (episode
   resets detected from the prefilled identical 11-frame history restart the
   EMA): ~1 during steady motion (room for tracking gains) and up to 1+scale
   while the reference decelerates/accelerates/turns hard (adapter pinned to
   the source exactly where deviations caused the gate tail regressions).
   Defaults dev_max=.4, turn_max=1.5 calibrated on the training set: steady
   frames (|v-EMA|<.1 m/s) stay at w~1, decel-to-stop peaks (dev~.55) and
   spins (|yaw|>1 rad/s) saturate w=1+scale. Weights are renormalized by
   their batch mean each update so the average anchor pressure equals the
   base schedule (uniform-pilot level, which produced the mean gains) while
   transitions keep up to (1+scale)/mean_w relative pinning.
"""
import argparse
import hashlib
import json
import math
import signal
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose')]
import isaacgym
import torch
import wandb
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args, class_to_dict
import rsl_rl.runners.on_policy_runner_mimic as runner_module
from train_4900_rawguard import PPO4900Guarded, SOURCE
PILOT_SCRIPT=ROOT/'tools/train_4900_rawguard.py'

class PPO4900Improved(PPO4900Guarded):
    def __init__(self,*args,anchor_coef=.1,anchor_coef_start=.3,anchor_decay_steps=500,
                 actor_rate=2e-6,actor_rate_end=2e-7,lr_decay_steps=20000,
                 ema_decay=.99,bank_capacity=16384,std_anneal_start=2000,std_floor=.03,
                 exploration_std=.1,last_layer_only=False,critic_warmup_updates=10,
                 anchor_dof_weights=None,transition_anchor_scale=9.,
                 transition_anchor_dev_max=.4,transition_anchor_turn_max=1.5,**kwargs):
        super().__init__(*args,actor_rate=actor_rate,anchor_coef=anchor_coef,
                         exploration_std=exploration_std,last_layer_only=last_layer_only,
                         critic_warmup_updates=critic_warmup_updates,**kwargs)
        self.anchor_coef_start=float(anchor_coef_start);self.anchor_decay_steps=int(anchor_decay_steps)
        self.actor_rate_end=float(actor_rate_end);self.lr_decay_steps=int(lr_decay_steps)
        self.ema_decay=float(ema_decay);self.bank_capacity=int(bank_capacity)
        self.std_anneal_start=int(std_anneal_start);self.std_floor=float(std_floor)
        self.bank_ptr=0;self.bank_size=0
        if anchor_dof_weights is None:
            self.anchor_dof_weights=torch.ones(23,device=self.device)
        else:
            weights=torch.tensor([float(x) for x in anchor_dof_weights],device=self.device)
            if weights.shape!=(23,):raise ValueError(f'anchor_dof_weights must have 23 entries, got {tuple(weights.shape)}')
            self.anchor_dof_weights=weights
        self.ema={k:v.detach().clone() for k,v in self.actor_critic.adapter.state_dict().items()}
        self.transition_anchor_scale=float(transition_anchor_scale)
        self.transition_anchor_dev_max=float(transition_anchor_dev_max)
        self.transition_anchor_turn_max=float(transition_anchor_turn_max)
        self._ref_speed_ema=None
        self.bank_w=None

    def _transition_weights(self, obs_steps):
        """Per-sample anchor multiplier, one per rollout transition.

        obs_steps is the rollout storage observations (num_steps, num_envs,
        7001).  A ~1s EMA of the reference root speed is walked along each
        env's trajectory in temporal order: gait-frequency speed oscillation
        is attenuated by the EMA while sustained transitions (decelerate /
        accelerate / turn) stand out as |v - EMA|.  Episode resets are
        detected by the prefilled 11-frame history (frames identical) and
        restart the EMA.  Weights are returned flattened in step-major order
        to match the storage mini-batch indexing.
        """
        steps,envs=obs_steps.shape[0],obs_steps.shape[1]
        if self.transition_anchor_scale<=0:
            return torch.ones(steps*envs,device=obs_steps.device)
        assert obs_steps.shape[-1]==7001,f'expected 7001-dim policy obs, got {obs_steps.shape[-1]}'
        v=obs_steps[:,:,4:7].norm(dim=-1)          # (steps, envs) reference root speed
        turn=obs_steps[:,:,7].abs()                # (steps, envs) reference yaw rate
        fresh=(obs_steps[:,:,:105]-obs_steps[:,:,1050:1155]).abs().amax(dim=-1)<1e-7
        if self._ref_speed_ema is None:
            self._ref_speed_ema=v[0].detach().clone()
        ema=self._ref_speed_ema
        alpha=.02  # EMA time constant ~ dt/alpha = 1s at 50 Hz
        ws=[];devs=[]
        for s in range(steps):
            ema=torch.where(fresh[s],v[s],ema)
            ema=ema+alpha*(v[s]-ema)
            dev=(v[s]-ema).abs()
            devs.append(dev.detach())
            score=torch.maximum((dev/self.transition_anchor_dev_max).square(),(turn[s]/self.transition_anchor_turn_max).square()).clamp(max=1.)
            ws.append(1.+self.transition_anchor_scale*score)
        self._ref_speed_ema=ema.detach()
        flat_v=v.reshape(-1);flat_turn=turn.reshape(-1)
        flat_dev=torch.stack(devs,dim=0).reshape(-1)
        self._trans_debug=dict(v_p50=float(flat_v.median()),v_p90=float(flat_v.quantile(.9)),
                               turn_p50=float(flat_turn.median()),turn_p90=float(flat_turn.quantile(.9)),
                               dev_p50=float(flat_dev.median()),dev_p90=float(flat_dev.quantile(.9)))
        return torch.stack(ws,dim=0).reshape(-1)

    def _mini_batch_generator(self):
        """Replica of RolloutStorage.mini_batch_generator that also yields the
        flat batch indices so per-sample transition weights can be aligned."""
        st=self.storage
        n=st.num_envs*st.num_transitions_per_env
        mini=n//self.num_mini_batches
        indices=torch.randperm(self.num_mini_batches*mini,device=self.device)
        obs_f=st.observations.flatten(0,1)
        crit_f=(st.privileged_observations if st.privileged_observations is not None else st.observations).flatten(0,1)
        acts_f=st.actions.flatten(0,1);vals_f=st.values.flatten(0,1)
        rets_f=st.returns.flatten(0,1);advs_f=st.advantages.flatten(0,1)
        logps_f=st.actions_log_prob.flatten(0,1);mus_f=st.mu.flatten(0,1);sigs_f=st.sigma.flatten(0,1)
        for _epoch in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                idx=indices[i*mini:(i+1)*mini]
                yield idx,(obs_f[idx],crit_f[idx],acts_f[idx],vals_f[idx],advs_f[idx],rets_f[idx],
                           logps_f[idx],mus_f[idx],sigs_f[idx],(None,None),None)

    def _anchor_coef(self):
        if self.anchor_decay_steps<=0 or self.count>=self.anchor_decay_steps:return self.anchor_coef
        return self.anchor_coef_start+(self.anchor_coef-self.anchor_coef_start)*self.count/self.anchor_decay_steps

    def _actor_lr(self):
        if self.lr_decay_steps<=0 or self.count>=self.lr_decay_steps:return self.actor_rate_end
        return self.actor_rate_end+.5*(self.actor_rate-self.actor_rate_end)*(1+math.cos(math.pi*self.count/self.lr_decay_steps))

    def update(self):
        actor=self.actor_critic;total_value=total_surrogate=total_anchor=total_drift=0.;steps=0;skipped=0;max_kl=0.
        anchor_coef_t=self._anchor_coef()
        self.ppo_optimizer.param_groups[0]['lr']=self._actor_lr()
        w_flat=self._transition_weights(self.storage.observations)
        # Renormalize by the batch mean so the AVERAGE anchor pressure equals
        # the base schedule (same level as the plain uniform-anchor pilot that
        # produced the mean gains) while transitions keep their relative
        # up-to-(scale/mean) pinning. Without this the pin dominates and the
        # adapter never deviates from source (observed: no mean gains at @50).
        trans_w_mean=float(w_flat.mean());trans_frac=float((w_flat>2.).float().mean())
        w_flat=w_flat/(trans_w_mean+1e-8)
        generator=self._mini_batch_generator()
        for idx,sample in generator:
            obs,critic_obs,actions,old_values,advantages,returns,old_logp,old_mu,old_std,_,_=sample
            trans_w=w_flat[idx]
            with torch.no_grad():
                base,hist=actor.split_obs(obs);z=actor.history_encoder(hist)
                source_mu=self.source.layerwise_actor(base,z)
                if self.bank is None:
                    self.bank=torch.zeros(self.bank_capacity,1155+128,device=self.device)
                    self.bank_targets=torch.zeros(self.bank_capacity,23,device=self.device)
                    self.bank_w=torch.ones(self.bank_capacity,device=self.device)
                bank_idx=torch.randperm(len(base),device=self.device)[:128]
                values=torch.cat((base[bank_idx],z[bank_idx]),-1)
                targets=source_mu[bank_idx]
                n_new=min(len(bank_idx),self.bank_capacity)
                self.bank[self.bank_ptr:self.bank_ptr+n_new]=values[:n_new]
                self.bank_targets[self.bank_ptr:self.bank_ptr+n_new]=targets[:n_new]
                self.bank_w[self.bank_ptr:self.bank_ptr+n_new]=trans_w[bank_idx][:n_new]
                self.bank_ptr=(self.bank_ptr+n_new)%self.bank_capacity
                self.bank_size=min(self.bank_capacity,self.bank_size+n_new)
            mu=actor.layerwise_actor(base,z)
            distribution=torch.distributions.Normal(mu,actor.std)
            logp=distribution.log_prob(actions).sum(-1)
            with torch.no_grad():
                kl=(torch.log(actor.std/old_std)+(old_std.square()+(old_mu-mu).square())/(2*actor.std.square())-.5).sum(-1).mean()
                max_kl=max(max_kl,float(kl))
            if kl>.012: skipped+=1;continue
            ratio=(logp-old_logp.squeeze(-1)).exp();adv=advantages.squeeze(-1)
            surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(.9,1.1)).mean()
            value=actor.evaluate(critic_obs)
            value_loss=(value-returns).square().mean()
            anchor=((((mu-source_mu)/actor.std).square()*self.anchor_dof_weights).sum(-1)*trans_w).mean()*.5
            indices=torch.randint(self.bank_size,(min(1024,self.bank_size),),device=self.device)
            bank=self.bank[indices];bank_mu=actor.layerwise_actor(bank[:,:1155],bank[:,1155:])
            bank_w=self.bank_w[indices]
            replay=((((bank_mu-self.bank_targets[indices])/actor.std).square()*self.anchor_dof_weights).sum(-1)*bank_w).mean()*.5
            loss=surrogate+self.value_loss_coef*value_loss+anchor_coef_t*(anchor+replay)
            self.optimizer.zero_grad();loss.backward()
            if self.count < self.critic_warmup_updates:
                for parameter in actor.adapter.parameters():parameter.grad=None
            torch.nn.utils.clip_grad_norm_(self.ppo_params,.5);self.optimizer.step()
            total_value+=float(value_loss.detach());total_surrogate+=float(surrogate.detach());total_anchor+=float(anchor.detach());total_drift+=float((mu-source_mu).abs().mean().detach());steps+=1
        if self.ema_decay>0:
            with torch.no_grad():
                for k,v in actor.adapter.state_dict().items():
                    self.ema[k].mul_(self.ema_decay).add_(v.detach(),alpha=1-self.ema_decay)
        self.storage.clear();self.count+=1
        if self.std_anneal_start>=0 and self.count>=self.std_anneal_start:
            with torch.no_grad():actor.std.mul_(.999).clamp_(min=self.std_floor)
        if not all(torch.equal(actor.state_dict()[k],v) for k,v in self.initial_frozen.items()):raise RuntimeError('Frozen source parameters changed')
        n=max(steps,1)
        self.anyadapter_metrics=dict(source_anchor_kl=total_anchor/n,source_action_mae=total_drift/n,max_rollout_kl=max_kl,skipped_minibatches=skipped,bank_size=self.bank_size,world_model_loss=0.,anchor_coef_effective=anchor_coef_t,actor_lr=self.ppo_optimizer.param_groups[0]['lr'],std_mean=float(actor.std.mean()),ema_decay=self.ema_decay,transition_anchor_weight_mean=trans_w_mean,transition_fraction=trans_frac,**self._trans_debug)
        if hasattr(self,"audit_path"):
            with self.audit_path.open("a") as f:f.write(json.dumps(dict(update=self.count,**self.anyadapter_metrics))+"\n")
        self.update_counter()
        return total_value/n,total_surrogate/n,0.,0.,total_anchor/n,0.

    def snapshot_ema(self):
        """Swap EMA-smoothed adapter weights into the live module; returns the
        live weights so the caller can restore them after saving."""
        if self.ema_decay<=0:return None
        current={k:v.detach().clone() for k,v in self.actor_critic.adapter.state_dict().items()}
        self.actor_critic.adapter.load_state_dict(self.ema)
        return current

    def restore_adapter(self,current):
        if current is not None:self.actor_critic.adapter.load_state_dict(current)

runner_module.PPO4900Improved=PPO4900Improved

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--actor-rate',type=float,default=2e-6);p.add_argument('--actor-rate-end',type=float,default=2e-7);p.add_argument('--lr-decay-steps',type=int,default=20000)
    p.add_argument('--anchor-coef',type=float,default=.1);p.add_argument('--anchor-coef-start',type=float,default=.3);p.add_argument('--anchor-decay-steps',type=int,default=500)
    p.add_argument('--exploration-std',type=float,default=.1);p.add_argument('--last-layer-only',action='store_true')
    p.add_argument('--critic-warmup-updates',type=int,default=10)
    p.add_argument('--ema-decay',type=float,default=.99);p.add_argument('--bank-capacity',type=int,default=16384)
    p.add_argument('--std-anneal-start',type=int,default=2000);p.add_argument('--std-floor',type=float,default=.03)
    p.add_argument('--anchor-dof-weights',type=str,default=None,
                   help='comma-separated 23 anchor weights: L-leg(6) R-leg(6) waist(3) arms(8); e.g. legs 2, ankles 3, waist 2, arms 1')
    p.add_argument('--transition-anchor-scale',type=float,default=9.,
                   help='per-sample anchor multiplier added while the reference is in a sustained transition (0 disables)')
    p.add_argument('--transition-anchor-dev-max',type=float,default=.4,
                   help='|ref speed - 1s EMA| (m/s) that saturates the transition anchor')
    p.add_argument('--transition-anchor-turn-max',type=float,default=1.5,help='|ref yaw rate| (rad/s) that saturates the transition anchor')
    p.add_argument('--resume-from',type=str,default=None)
    custom,remaining=p.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    if custom.anchor_dof_weights:
        custom.anchor_dof_weights=[float(x) for x in custom.anchor_dof_weights.split(',')]
        if len(custom.anchor_dof_weights)!=23:raise ValueError('--anchor-dof-weights needs 23 comma-separated values')
    args=get_args()
    if args.resume or args.resumeid:raise ValueError('use --resume-from <checkpoint.pt> instead of --resume/--resumeid')
    if not 1<=args.max_iterations<=50000:raise ValueError('max_iterations must be in [1,50000]')
    resume_path=Path(custom.resume_from) if custom.resume_from else None
    if resume_path is not None and not resume_path.is_file():raise FileNotFoundError(f'--resume-from not found: {resume_path}')
    args.task='g1_motion_wm_anyadapter';args.headless=True;args.no_wandb=True
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.motion.motion_file=str(ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml')
    cfg.motion.motion_curriculum=False
    cfg.motion_wm.mode_probabilities=(.5,0.,.5);cfg.motion_wm.artificial_corruption=False
    cfg.motion_wm.prefill_history=True
    cfg.env.num_envs=args.num_envs;cfg.seed=args.seed
    train.seed=args.seed;train.policy.fix_action_std=True;args.fix_action_std=True
    train.runner.algorithm_class_name='PPO4900Improved';train.runner.save_interval=50;train.runner.constant_save_interval=True;train.runner.resume=False
    train.algorithm.source_checkpoint=str(SOURCE)
    for k in ('actor_rate','actor_rate_end','lr_decay_steps','anchor_coef','anchor_coef_start','anchor_decay_steps','exploration_std','last_layer_only','critic_warmup_updates','ema_decay','bank_capacity','std_anneal_start','std_floor','anchor_dof_weights','transition_anchor_scale','transition_anchor_dev_max','transition_anchor_turn_max'):
        setattr(train.algorithm,k,getattr(custom,k))
    train.algorithm.num_learning_epochs=3;train.algorithm.num_mini_batches=16;train.algorithm.world_model_loss_coef=0.;train.algorithm.entropy_coef=0.
    out=ROOT/'legged_gym/logs/recovery_4900'/args.exptid
    # The dir may legitimately exist as a leftover from a crashed attempt
    # (launcher retries); refuse only when it already contains checkpoints
    # and this is a fresh run (protects against overwriting a previous run).
    out.mkdir(parents=True,exist_ok=True)
    if resume_path is None and list(out.glob('model_*.pt')):
        raise SystemExit(f'{out} already contains checkpoints; choose a fresh --exptid or pass --resume-from')
    wandb.init(mode='disabled')
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,train=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=str(out))
    runner.alg.audit_path=out/"anchor_metrics.jsonl"
    if resume_path is not None:
        runner.load(str(resume_path))
        runner.alg.count=runner.current_learning_iteration
        sidecar=resume_path.parent/'bank_state_latest.pt'
        if sidecar.is_file():
            sc=torch.load(sidecar,map_location='cpu')
            alg=runner.alg
            if sc.get('bank') is not None:
                alg.bank=sc['bank'].to(runner.device);alg.bank_targets=sc['bank_targets'].to(runner.device)
                alg.bank_size=int(sc.get('bank_size',len(alg.bank)));alg.bank_ptr=int(sc.get('bank_ptr',0))
                alg.bank_w=(sc['bank_w'].to(runner.device) if sc.get('bank_w') is not None
                            else torch.ones(alg.bank_capacity,device=runner.device))
            if sc.get('ema'):alg.ema={k:v.to(runner.device) for k,v in sc['ema'].items()}
    else:
        runner.save(str(out/'model_0.pt'))
    _orig_save=runner.save
    def save_with_ema_and_bank(path,infos=None,iteration=None):
        prev=runner.alg.snapshot_ema()
        try:_orig_save(path,infos=infos,iteration=iteration)
        finally:runner.alg.restore_adapter(prev)
        torch.save(dict(bank=runner.alg.bank,bank_targets=runner.alg.bank_targets,bank_size=runner.alg.bank_size,bank_ptr=runner.alg.bank_ptr,count=runner.alg.count,ema=runner.alg.ema,bank_w=runner.alg.bank_w),str(out/'bank_state_latest.pt'))
    runner.save=save_with_ema_and_bank
    manifest=dict(training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pilot_script_sha256=hashlib.sha256(PILOT_SCRIPT.read_bytes()).hexdigest(),
                  env_file_sha256=hashlib.sha256((ROOT/'legged_gym/legged_gym/envs/g1/g1_motion_wm_anyadapter.py').read_bytes()).hexdigest(),
                  pipeline_file_sha256=hashlib.sha256((ROOT/'motion_world_model/training_reference.py').read_bytes()).hexdigest(),
                  source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  resume_from=str(resume_path) if resume_path is not None else None,
                  resume_sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest() if resume_path is not None else None,
                  resumed_iteration=runner.current_learning_iteration,
                  mode_probabilities=tuple(cfg.motion_wm.mode_probabilities),prefill_history=True,
                  protocol_sha256=hashlib.sha256((ROOT/'reports/recovery_4900/protocol.json').read_bytes()).hexdigest(),
                  environment=class_to_dict(cfg),training=class_to_dict(train),custom=vars(custom),seed=args.seed)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    def _sigterm(signum,frame):
        # Best-effort checkpoint on SIGTERM so the auto-restart launcher can
        # resume from the latest update instead of the last periodic save.
        try:
            runner.save(str(out/f'model_{runner.alg.count}.pt'))
            print(f'[sigterm] checkpoint saved at update {runner.alg.count}',flush=True)
        except Exception as e:
            print(f'[sigterm] save failed ({e}); periodic checkpoint still valid',flush=True)
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,_sigterm)
    try:
        # learn_RL adds num_learning_iterations on top of the restored
        # iteration, so subtract the already-completed updates on resume
        # (otherwise an interrupted run would over-train by one full span).
        n_iters=max(1,args.max_iterations-(runner.current_learning_iteration if resume_path is not None else 0))
        runner.learn(num_learning_iterations=n_iters,init_at_random_ep_len=False)
    except KeyboardInterrupt:
        torch.save(dict(bank=runner.alg.bank,bank_targets=runner.alg.bank_targets,bank_size=runner.alg.bank_size,bank_ptr=runner.alg.bank_ptr,count=runner.alg.count,ema=runner.alg.ema,bank_w=runner.alg.bank_w),str(out/'interrupt_bank_state.pt'))
        raise
if __name__=='__main__':main()
