"""Independent fixed-policy audit; no optimizer construction, step, or training."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose'),str(ROOT/'legged_gym/scripts')]
import isaacgym  # before torch
import torch
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args
from heading23500_env import Heading23500Env
from export_twist_anyadapter_opentrack_jit import build_actor
from rsl_rl.storage.rollout_storage import RolloutStorage
from train_4900_improved import PPO4900Improved
from heading300_first_update_verify import RUN,sha


def restore(obj,values):
    for key,value in values.items():
        if isinstance(value,dict):
            child=getattr(obj,key,None)
            if isinstance(child,dict):setattr(obj,key,value);continue
            if not hasattr(child,'__dict__'):child=SimpleNamespace();setattr(obj,key,child)
            restore(child,value)
        else:setattr(obj,key,value)


def audit(output,blocks,args):
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((RUN/'manifest.json').read_text())
    cfg,train=task_registry.get_cfgs('g1_motion_wm_anyadapter');restore(cfg,manifest['environment'])
    args.task='heading300_first_update_readonly_audit';args.num_envs=512;args.seed=42;args.headless=True
    task_registry.register(args.task,Heading23500Env,cfg,train)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    actor=build_actor(RUN/'model_25.pt').to(env.device).eval().requires_grad_(False)
    settings=manifest['training']['algorithm']
    transition=SimpleNamespace(transition_anchor_scale=settings['transition_anchor_scale'],
        transition_anchor_dev_max=settings['transition_anchor_dev_max'],
        transition_anchor_turn_max=settings['transition_anchor_turn_max'],_ref_speed_ema=None)
    obs=env.get_observations();all_data=[];reset_count=0
    for block in range(blocks):
        storage=RolloutStorage(512,24,[7003],[1318],[23],device=env.device)
        metadata=[]
        with torch.no_grad():
            for step in range(24):
                # Capture metadata at the same start-of-action time as obs.
                contact=env.contact_forces[:,env.feet_indices,2]>5.
                clean_height=env._ref_root_pos[:,2]-env.episode_init_origin[:,2]
                clean_knee=env._ref_dof_pos[:,[3,9]].mean(-1)
                metadata.append(dict(contact=contact.clone(),clean_height=clean_height.clone(),
                    clean_knee=clean_knee.clone(),clean_speed=env._ref_root_vel[:,:2].norm(dim=-1).clone(),
                    clean_turn=env._ref_root_ang_vel[:,2].abs().clone(),
                    mode=env.motion_reference_pipeline.mode.clone(),valid=(obs.abs().amax(-1)>0)))
                tr=RolloutStorage.Transition();tr.observations=obs.clone()
                critic=env.get_privileged_observations();tr.critic_observations=critic.clone()
                tr.actions=actor.act(obs).detach();tr.values=actor.evaluate(critic).detach()
                tr.actions_log_prob=actor.get_actions_log_prob(tr.actions).detach()
                tr.action_mean=actor.action_mean.detach();tr.action_sigma=actor.action_std.detach()
                obs,_,rewards,dones,infos=env.step(tr.actions)
                rewards=rewards.clone()
                if 'time_outs' in infos:rewards+=settings['gamma']*tr.values.squeeze(-1)*infos['time_outs']
                tr.rewards=rewards;tr.dones=dones;storage.add_transitions(tr);reset_count+=int(dones.sum())
            storage.compute_returns(actor.evaluate(env.get_privileged_observations()).detach(),settings['gamma'],settings['lam'])
            flat=storage.observations.flatten(0,1);zs=[]
            for chunk in flat.split(2048):
                _,history=actor.split_obs(chunk);zs.append(actor.history_encoder(history))
            weights=PPO4900Improved._transition_weights(transition,storage.observations[...,:7001]).clamp(1.,3.)
            data=dict(base=flat[:,:1155].cpu(),latent=torch.cat(zs).cpu(),context=flat[:,-2:].cpu(),
                actions=storage.actions.flatten(0,1).cpu(),old_logp=storage.actions_log_prob.flatten().cpu(),
                advantages=storage.advantages.flatten().cpu(),old_mu=storage.mu.flatten(0,1).cpu(),
                weights=weights.cpu(),**{k:torch.stack([m[k] for m in metadata]).reshape((-1,2) if k=='contact' else (-1,)).cpu() for k in metadata[0]})
            all_data.append(data)
        print('Read-only audit block',block+1,'/',blocks,flush=True)
    merged={k:torch.cat([d[k] for d in all_data]) for k in all_data[0]}
    torch.save(merged,output/'audit_states.pt')
    receipt=dict(policy=str(RUN/'model_25.pt'),policy_sha256=sha(RUN/'model_25.pt'),
        source_manifest_sha256=sha(RUN/'manifest.json'),seed=42,num_envs=512,rollout_steps=24,
        blocks=blocks,frames=512*24*blocks,resets=reset_count,optimizer_steps=0,actor_weight_updates=0,
        provenance='new independent stochastic fixed-model25 audit; not a reconstruction of historical update26 rollout',
        original_configuration_preserved=True,gae='native RolloutStorage.compute_returns including native timeout bootstrap',
        contact_time='start of action interval; true Isaac contact forces>5N',
        caution='RNG history, environment state and optimizer minibatches of the historical first update were not saved; these audit gradients are independent probes, not historical per-group gradients')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blocks',type=int,default=8);custom,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining;args=get_args()
    audit(custom.output,custom.blocks,args)
