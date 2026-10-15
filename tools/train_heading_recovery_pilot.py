"""Bounded same-asset MuJoCo correction PPO, with full raw/WM stop gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from heading_recovery_core import HeadingRecoveryBatch,RecoveryActorCritic,ROOT,POLICY,PHYSICAL,XML,CONTRACT
from evaluate_heading_recovery import full77,recovery_task,protection,save,sha


def compute_gae(rewards,values,next_values,dones,falls,gamma=.995,lam=.98):
    # Timeout uses the pre-reset next value; recursion never crosses a reset.
    advantage=torch.zeros_like(rewards);running=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*(~falls[t]).float()*next_values[t]-values[t]
        running=delta+gamma*lam*(~dones[t]).float()*running
        advantage[t]=running
    return advantage,advantage+values


def task_score(rows):return float(np.mean([r['final2s_yaw_rmse_deg'] for r in rows]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-envs',type=int,default=16);p.add_argument('--max-updates',type=int,default=55)
    p.add_argument('--rollout-steps',type=int,default=128);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((a.baseline/'full77/results.json').read_text())
    baseline_task=json.loads((a.baseline/'task/results.json').read_text())
    assert (a.baseline/'COMPLETE').is_file()
    assert all(max(r['exact_original_replay'].values())==0 for r in baseline)
    assert a.max_updates<=55 and a.num_envs<=32 and a.rollout_steps<=128
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    learner=RecoveryActorCritic().to(a.device);rollout_actor=RecoveryActorCritic().eval()
    rollout_actor.load_state_dict(learner.state_dict())
    actor_opt=torch.optim.Adam(learner.actor.parameters(),lr=3e-4)
    critic_opt=torch.optim.Adam(learner.critic.parameters(),lr=1e-3)
    model_hashes={str(x.relative_to(ROOT)):sha(x) for x in [POLICY,PHYSICAL,XML]}
    manifest=dict(contract=CONTRACT,seed=42,device=a.device,num_envs=a.num_envs,rollout_steps=a.rollout_steps,
        max_updates=a.max_updates,critic_warmup_updates=5,eval_every_actor_updates=10,
        delta_bound_action=.1,exploration_std=.015,actor_lr=3e-4,critic_lr=1e-3,
        gamma=.995,gae_lambda=.98,ppo_epochs=4,minibatch=256,clip=.1,
        task='8s smooth +/-30deg turn at 2..4s then hold, initial yaw uniform +/-15deg',
        task_validation='six fixed +/-30deg targets x {-10,0,+10} initial yaw; developmental task gate',
        observation_contract='source7003 unchanged, correction168 with true state51, goal errors61, physical latent32, source action23, readiness1',
        privileged_teacher_only=True,source_and_physical_frozen=True,
        gates='all full77 raw/WM yaw/joint/velocity <=105% original and no new falls; any failure stops without additional training',
        task_gain_target=.10,multi_motion_regression_required_before_general_promotion=True,
        model_hashes=model_hashes,baseline=str(a.baseline.resolve()),
        code={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__).resolve(),ROOT/'tools/heading_recovery_core.py',ROOT/'tools/evaluate_heading_recovery.py']})
    save(out/'manifest.json',manifest)
    env=HeadingRecoveryBatch(a.num_envs);observation=env.reset();updates=[];actor_updates=0;start=time.perf_counter()
    for update in range(1,a.max_updates+1):
        record={k:[] for k in ['features','source','actions','old_logp','values','next_values','rewards','dones','falls']}
        rollout_actor.eval()
        for _ in range(a.rollout_steps):
            features=observation['features'];source=observation['source']
            with torch.no_grad():
                mean=rollout_actor.mean(features,source);dist=torch.distributions.Normal(mean,rollout_actor.std)
                actions=dist.sample();logp=dist.log_prob(actions).sum(-1);value=rollout_actor.value(features)
            next_observation,reward,done,info=env.step(actions.numpy(),source.numpy())
            with torch.no_grad():next_value=rollout_actor.value(info['terminal_features'])
            for key,val in [('features',features),('source',source),('actions',actions),('old_logp',logp),('values',value),
                            ('next_values',next_value),('rewards',reward),('dones',done),('falls',info['falls'])]:
                # Observation reset updates selected sibling slots, so own all tensors.
                record[key].append(val.detach().clone())
            observation=next_observation
        record={k:torch.stack(v) for k,v in record.items()}
        advantages,returns=compute_gae(record['rewards'],record['values'],record['next_values'],record['dones'],record['falls'])
        features=record['features'].flatten(0,1).to(a.device);sources=record['source'].flatten(0,1).to(a.device)
        actions=record['actions'].flatten(0,1).to(a.device);old_logp=record['old_logp'].flatten().to(a.device)
        returns=returns.flatten().to(a.device);advantages=advantages.flatten().to(a.device)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        old_actor=[v.detach().clone() for v in learner.actor.parameters()]
        losses=[];kls=[]
        for _ in range(4):
            order=torch.randperm(len(features),device=a.device)
            for ids in order.split(256):
                value=learner.value(features[ids]);value_loss=(value-returns[ids]).square().mean()
                critic_opt.zero_grad();value_loss.backward();torch.nn.utils.clip_grad_norm_(learner.critic.parameters(),.5);critic_opt.step()
                if update>5:
                    mean=learner.mean(features[ids],sources[ids]);dist=torch.distributions.Normal(mean,learner.std)
                    logp=dist.log_prob(actions[ids]).sum(-1);ratio=(logp-old_logp[ids]).exp()
                    loss=torch.maximum(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(.9,1.1)).mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite recovery PPO objective')
                    actor_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learner.actor.parameters(),.5);actor_opt.step()
                    losses.append(float(loss.detach()));kls.append(float((old_logp[ids]-logp).mean().detach()))
        drift=float(torch.stack([(x-y).square().sum() for x,y in zip(learner.actor.parameters(),old_actor)]).sum().sqrt())
        if update<=5:assert drift==0
        else:actor_updates+=1
        rollout_actor.load_state_dict(learner.state_dict())
        metrics=dict(update=update,actor_updates=actor_updates,actor_parameter_l2=drift,
            mean_reward=float(record['rewards'].mean()),falls=int(record['falls'].sum()),
            completed_episodes=int(record['dones'].sum()),actor_loss=float(np.mean(losses)) if losses else None,
            approximate_kl=float(np.mean(kls)) if kls else None)
        updates.append(metrics)
        with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        print('UPDATE',metrics,flush=True)
        if update==5 or (actor_updates>0 and actor_updates%10==0):
            checkpoint=out/f'model_{update}.pt'
            torch.save(dict(contract=CONTRACT,model_state_dict=learner.state_dict(),actor_optimizer=actor_opt.state_dict(),
                critic_optimizer=critic_opt.state_dict(),update=update,actor_updates=actor_updates,
                source_policy_sha256=sha(POLICY),physical_wm_sha256=sha(PHYSICAL),manifest=manifest),checkpoint)
            if update==5:
                assert all(float(v.abs().max())==0 for v in [learner.actor[-1].weight,learner.actor[-1].bias])
                continue
            evaluation=out/f'evaluation_{update}';evaluation.mkdir()
            task=recovery_task(rollout_actor,evaluation/'task')
            full=full77(rollout_actor,evaluation/'full77')
            gate=protection(full,baseline)
            gain=1-task_score(task)/task_score(baseline_task)
            evaluation_receipt=dict(update=update,actor_updates=actor_updates,checkpoint=str(checkpoint),
                task_final_hold_yaw_gain=gain,task_gain_met=gain>=.1,
                task_no_new_falls=all(not r['fall_proxy'] for r in task),full77_protection=gate,
                general_policy_promoted=False,privileged_teacher_only=True)
            save(evaluation/'receipt.json',evaluation_receipt)
            (evaluation/'COMPLETE').write_text('Six recovery cases and two full77 modes completed.\n')
            if not gate['passed']:
                save(out/'STOPPED.json',dict(reason='full77 protection failed; no more training in this pilot',**evaluation_receipt))
                break
            if gain>=.1 and evaluation_receipt['task_no_new_falls']:
                save(out/'SPECIALIST_CANDIDATE.json',evaluation_receipt)
    for path,digest in model_hashes.items():assert sha(ROOT/path)==digest
    save(out/'training_receipt.json',dict(updates=len(updates),actor_updates=actor_updates,
        wall_seconds=time.perf_counter()-start,source_hashes_unchanged=True,
        stopped_on_gate_failure=(out/'STOPPED.json').exists(),general_policy_promoted=False))
    (out/'COMPLETE').write_text('Bounded recovery pilot finished; failed candidates never replace heading300.\n')


if __name__=='__main__':main()
