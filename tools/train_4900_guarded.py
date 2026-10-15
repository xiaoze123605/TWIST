"""Bounded source-anchored pilot. Does not authorize long continuation."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose')]
import isaacgym
import torch
import wandb
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args, class_to_dict
from rsl_rl.algorithms.ppo_anyadapter_opentrack import PPOAnyAdapterOpenTrack
import rsl_rl.runners.on_policy_runner_mimic as runner_module
SOURCE=ROOT/'legged_gym/logs/g1_motion_wm_anyadapter/pilot_gpu_seed42_200/model_4900.pt'

class PPO4900Guarded(PPOAnyAdapterOpenTrack):
    def __init__(self,*args,source_checkpoint,actor_rate=2e-6,anchor_coef=.1,exploration_std=.1,last_layer_only=False,**kwargs):
        super().__init__(*args,**kwargs)
        actor=self.actor_critic
        actor.load_state_dict(torch.load(source_checkpoint,map_location=self.device)['model_state_dict'],strict=True)
        self.source=deepcopy(actor).eval().requires_grad_(False)
        actor.requires_grad_(False)
        actor.adapter.requires_grad_(not last_layer_only);actor.adapter[-1].requires_grad_(True);actor.critic.requires_grad_(True)
        adapter_params=[p for p in actor.adapter.parameters() if p.requires_grad]
        with torch.no_grad():actor.std.clamp_(max=exploration_std)
        self.actor_rate=actor_rate;self.anchor_coef=anchor_coef
        self.ppo_params=adapter_params+list(actor.critic.parameters())
        self.ppo_optimizer=torch.optim.Adam([dict(params=adapter_params,lr=actor_rate),dict(params=list(actor.critic.parameters()),lr=1e-4)])
        self.optimizer=self.ppo_optimizer;self.learning_rate=actor_rate
        self.bank=None;self.bank_targets=None;self.count=0
        self.initial_frozen={k:v.detach().clone() for k,v in actor.state_dict().items() if k.startswith(('history_encoder.','world_model.','base_actor.','layerwise_actor.base_','layerwise_actor.motion_'))}
    def update(self):
        actor=self.actor_critic;total_value=total_surrogate=total_anchor=total_drift=0.;steps=0;skipped=0;max_kl=0.
        generator=self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
        for sample in generator:
            obs,critic_obs,actions,old_values,advantages,returns,old_logp,old_mu,old_std,_,_=sample
            with torch.no_grad():
                base,hist=actor.split_obs(obs);z=actor.history_encoder(hist)
                source_mu=self.source.layerwise_actor(base,z)
                if self.bank is None or len(self.bank)<16384:
                    indices=torch.randperm(len(base),device=self.device)[:128]
                    values=torch.cat((base[indices],z[indices]),-1)
                    self.bank=values.clone() if self.bank is None else torch.cat((self.bank,values))[:16384]
                    self.bank_targets=source_mu[indices].clone() if self.bank_targets is None else torch.cat((self.bank_targets,source_mu[indices]))[:16384]
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
            anchor=((mu-source_mu)/actor.std).square().mean()*.5
            indices=torch.randint(len(self.bank),(min(1024,len(self.bank)),),device=self.device)
            bank=self.bank[indices];bank_mu=actor.layerwise_actor(bank[:,:1155],bank[:,1155:])
            replay=((bank_mu-self.bank_targets[indices])/actor.std).square().mean()*.5
            loss=surrogate+self.value_loss_coef*value_loss+self.anchor_coef*(anchor+replay)
            self.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ppo_params,.5);self.optimizer.step()
            total_value+=float(value_loss.detach());total_surrogate+=float(surrogate.detach());total_anchor+=float(anchor.detach());total_drift+=float((mu-source_mu).abs().mean().detach());steps+=1
        self.storage.clear();self.count+=1
        if not all(torch.equal(actor.state_dict()[k],v) for k,v in self.initial_frozen.items()):raise RuntimeError('Frozen source parameters changed')
        n=max(steps,1)
        self.anyadapter_metrics=dict(source_anchor_kl=total_anchor/n,source_action_mae=total_drift/n,max_rollout_kl=max_kl,skipped_minibatches=skipped,bank_size=len(self.bank),world_model_loss=0.)
        if hasattr(self,"audit_path"):
            with self.audit_path.open("a") as f:f.write(json.dumps(dict(update=self.count,**self.anyadapter_metrics))+"\n")
        self.update_counter()
        return total_value/n,total_surrogate/n,0.,0.,total_anchor/n,0.
    def on_save_checkpoint(self):
        return dict(guarded4900=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),actor_rate=self.actor_rate,anchor_coef=self.anchor_coef,pilot_updates=self.count,bank=self.bank,bank_targets=self.bank_targets))

runner_module.PPO4900Guarded=PPO4900Guarded

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--actor-rate',type=float,default=2e-6);p.add_argument('--anchor-coef',type=float,default=.1);p.add_argument('--exploration-std',type=float,default=.1);p.add_argument('--last-layer-only',action='store_true')
    custom,remaining=p.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    args=get_args()
    if args.resume or args.resumeid:raise ValueError('Pilot only: fresh from verified model_4900')
    if not 1<=args.max_iterations<=200:raise ValueError('Pilot capped at 200 updates until closed-loop gate passes')
    args.task='g1_motion_wm_anyadapter';args.headless=True;args.no_wandb=True
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.motion.motion_file=str(ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml')
    cfg.motion.motion_curriculum=False
    cfg.motion_wm.mode_probabilities=(.5,0.,.5);cfg.motion_wm.artificial_corruption=False
    cfg.env.num_envs=args.num_envs;cfg.seed=args.seed
    train.seed=args.seed;train.policy.fix_action_std=True;args.fix_action_std=True
    train.runner.algorithm_class_name='PPO4900Guarded';train.runner.save_interval=50;train.runner.constant_save_interval=True;train.runner.resume=False
    train.algorithm.source_checkpoint=str(SOURCE);train.algorithm.actor_rate=custom.actor_rate;train.algorithm.anchor_coef=custom.anchor_coef;train.algorithm.exploration_std=custom.exploration_std;train.algorithm.last_layer_only=custom.last_layer_only
    train.algorithm.num_learning_epochs=3;train.algorithm.num_mini_batches=16;train.algorithm.world_model_loss_coef=0.;train.algorithm.entropy_coef=0.
    out=ROOT/'legged_gym/logs/recovery_4900'/args.exptid;out.mkdir(parents=True,exist_ok=False)
    wandb.init(mode='disabled')
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,train=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=str(out))
    runner.alg.audit_path=out/"anchor_metrics.jsonl"
    manifest=dict(training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'reports/recovery_4900/protocol.json').read_bytes()).hexdigest(),environment=class_to_dict(cfg),training=class_to_dict(train),custom=vars(custom),seed=args.seed)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    runner.save(str(out/'model_0.pt'))
    runner.learn(num_learning_iterations=args.max_iterations,init_at_random_ep_len=False)
if __name__=='__main__':main()
