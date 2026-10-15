"""One fixed-budget B1 pilot; complete archives and immediate full-mode stop."""
import isaacgym
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from coherent_recovery import CONTRACT,DIAG,teacher_features,prepare_start,LocalRecoveryBatch,evaluate_local,evaluate_full
from heading_recovery_core import ROOT,POLICY,PHYSICAL,XML,RecoveryActorCritic
from train_heading_recovery_pilot import compute_gae
from evaluate_heading_recovery import sha,save,protection


def local_gate(rows,baseline):
    reasons=[];gains=[];finals=[];successes=0
    for row,base in zip(rows,baseline):
        assert row['name']==base['name'] and row['frames']==400
        for key in ['yaw_rmse_deg','joint_rmse_rad']:
            if row[key]>1.05*base[key]:reasons.append(row['name']+': '+key+' regressed >5%')
        if row['fall_proxy']:reasons.append(row['name']+': fall proxy')
        if row['sign']:
            r=row['paired_recovery'];b=base['paired_recovery']
            gains.append(1-r['extra_yaw_iae_deg_seconds']/max(b['extra_yaw_iae_deg_seconds'],1e-8))
            finals.append(r['final1s_extra_yaw_rms_deg']/max(b['final1s_extra_yaw_rms_deg'],1e-8))
            successes+=row['sustained_recovery_seconds'] is not None
    mean_gain=float(np.mean(gains));mean_final=float(np.mean(finals))
    return dict(passed=not reasons and mean_gain>=.1 and mean_final<=.9 and successes>=2,
                protection_reasons=reasons,mean_paired_iae_gain=mean_gain,mean_final_extra_error_ratio=mean_final,
                sustained_success_cases=successes,required='>=10% mean paired IAE gain, >=10% mean final error gain, >=2 of4 sustained recoveries, <=5% absolute task regressions,no falls')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['baseline','train'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--device',default='cpu')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [POLICY,PHYSICAL,XML,ROOT/'tools/coherent_recovery.py',Path(__file__).resolve(),
           ROOT/'tools/heading_recovery_core.py',ROOT/'tools/diagnose_recovery_task.py',ROOT/'tools/evaluate_heading_recovery.py',
           ROOT/'tools/train_heading_recovery_pilot.py']}
    manifest=dict(contract=CONTRACT,seed=42,stage=a.stage,device=a.device,hashes=hashes,
        source_policy_frozen=True,physical_latent_masked=True,privileged_teacher_only=True,
        task='raw mocap77 control frames2100..2499, coherent turning+squats; source39.00s reset and3s real physical pre-roll',
        training_disturbance='worldZ pelvis torque chosen from {-1,0,+1}Nm, first0.2s only',
        evaluation_cases='nominal, +/-1Nm at42s, +/-1Nm at45s; latter timing withheld from training',
        correction_scale=.1,std=.015,execution_residual_clip=.1,num_envs=16,rollout_steps=128,
        max_updates=35,critic_warmup_updates=5,max_actor_updates=30,evaluate_every_actor_updates=10,
        actor_lr=.0003,critic_lr=.001,epochs=4,minibatch=256,clip=.1,gamma=.995,gae_lambda=.98,
        full_policy_scope='scheduled specialist active only control frames2100..2499; original source at all other frames',
        stop_gate='full77 raw/WM yaw,joint,velocity <=105% heading300,no new fall; immediate stop on first failed full protection',
        required_local_gate='10% paired IAE and final error mean gains; >=2 of4 sustained recoveries; absolute task<=105% baseline; no fall',
        archives='every rollout: source7003 and teacher168 inputs, sampled and executed actions, PD targets, advantages/returns/logprobs/values, physical before/after, phase, disturbance, done/fall/timeout; every minibatch indices and pre-update policy')
    save(out/'manifest.json',manifest);initial=prepare_start();actor=RecoveryActorCritic().eval()
    if a.stage=='baseline':
        local=evaluate_local(actor,initial,out/'local');full=evaluate_full(actor,out/'full77')
        for mode in ['raw','wm']:
            x=np.load(out/'full77'/f'{mode}.npz');old=np.load(ROOT/'reports/heading_recovery_p1_20261013/zero_baseline/full77'/f'{mode}.npz')
            for key in ['qpos','qvel','rpy','joint','final_action','pd_target']:assert np.array_equal(x[key],old[key]),(mode,key)
        old=np.load(DIAG/'reference_at39_preroll.npz');new=np.load(out/'local/nominal.npz')
        for key in ['qpos','qvel','rpy','final_action','pd_target']:assert np.array_equal(old[key][150:],new[key]),key
        save(out/'receipt.json',dict(contract=CONTRACT,full_zero_parity_exact=True,local_diagnosis_parity_exact=True,optimizer_steps=0))
        (out/'COMPLETE').write_text('Zero teacher baseline,5local cases and2complete full77 modes; exact original parity.\n');return
    assert a.baseline and (a.baseline/'COMPLETE').is_file()
    base_manifest=json.loads((a.baseline/'manifest.json').read_text())
    assert base_manifest['hashes']==hashes and base_manifest['contract']==CONTRACT
    baseline=json.loads((a.baseline/'full77/results.json').read_text());base_local=json.loads((a.baseline/'local/results.json').read_text())
    learner=RecoveryActorCritic().to(a.device);actor.load_state_dict(learner.state_dict())
    actor_opt=torch.optim.Adam(learner.actor.parameters(),lr=3e-4);critic_opt=torch.optim.Adam(learner.critic.parameters(),lr=1e-3)
    env=LocalRecoveryBatch(16,initial);obs=env.observe();actor_updates=0;start=time.perf_counter();(out/'rollouts').mkdir()
    for update in range(1,36):
        torch.save(dict(contract=CONTRACT,model_state_dict=actor.state_dict(),update=update-1),out/'rollouts'/f'policy_before_{update}.pt')
        rec={k:[] for k in ['features','legacy','source','actions','old_logp','values','next_values','rewards','dones','falls',
                            'timeouts','phase','disturbance','executed','pd_target','actual_action','physical_before','physical_after']}
        for _ in range(128):
            features=teacher_features(obs['features']);source=obs['source'].clone();legacy=obs['legacy'].clone()
            with torch.no_grad():
                dist=torch.distributions.Normal(actor.mean(features,source),actor.std);actions=dist.sample()
                logp=dist.log_prob(actions).sum(-1);value=actor.value(features)
            new,reward,done,info=env.step(actions.numpy(),source.numpy())
            with torch.no_grad():nv=actor.value(info['terminal_features'])
            vals=dict(features=features,legacy=legacy,source=source,actions=actions,old_logp=logp,values=value,
                      next_values=nv,rewards=reward,dones=done,falls=info['falls'],timeouts=info['timeouts'],
                      phase=info['reference_phase'],disturbance=info['disturbance_sign'],executed=info['executed_commands'],
                      pd_target=info['pd_target'],actual_action=info['actual_action'],physical_before=info['physical_before'],physical_after=info['physical_after'])
            for k,v in vals.items():rec[k].append(v.detach().clone() if torch.is_tensor(v) else torch.from_numpy(v.copy()))
            obs=new
        rec={k:torch.stack(v) for k,v in rec.items()}
        adv,ret=compute_gae(rec['rewards'],rec['values'],rec['next_values'],rec['dones'],rec['falls'])
        rec['advantages_raw']=adv.clone();rec['returns']=ret.clone();norm=(adv-adv.mean())/(adv.std()+1e-8);rec['advantages_normalized']=norm.clone()
        arrays={k:v.numpy() for k,v in rec.items()};np.savez_compressed(out/'rollouts'/f'rollout_{update}.npz',**arrays)
        x=rec['features'].flatten(0,1).to(a.device);s=rec['source'].flatten(0,1).to(a.device)
        action=rec['actions'].flatten(0,1).to(a.device);lp=rec['old_logp'].flatten().to(a.device)
        advantages=norm.flatten().to(a.device);returns=ret.flatten().to(a.device)
        losses=[];kls=[];clips=[];value_losses=[];orders=[];before=[v.detach().clone() for v in learner.actor.parameters()]
        for epoch in range(4):
            order=torch.randperm(len(x),device=a.device);orders.append(order.cpu().numpy())
            for ids in order.split(256):
                value_loss=(learner.value(x[ids])-returns[ids]).square().mean()
                assert torch.isfinite(value_loss);critic_opt.zero_grad();value_loss.backward();torch.nn.utils.clip_grad_norm_(learner.critic.parameters(),.5);critic_opt.step()
                value_losses.append(float(value_loss.detach()))
                if update>5:
                    distribution=torch.distributions.Normal(learner.mean(x[ids],s[ids]),learner.std)
                    logp=distribution.log_prob(action[ids]).sum(-1);ratio=(logp-lp[ids]).exp()
                    loss=torch.maximum(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(.9,1.1)).mean()
                    assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learner.actor.parameters(),.5);actor_opt.step()
                    losses.append(float(loss.detach()));kls.append(float((lp[ids]-logp).mean().detach()));clips.append(float(((ratio-1).abs()>.1).float().mean().detach()))
        np.savez_compressed(out/'rollouts'/f'minibatch_orders_{update}.npz',orders=np.stack(orders))
        drift=float(torch.stack([(v-b).square().sum() for v,b in zip(learner.actor.parameters(),before)]).sum().sqrt())
        if update<=5:assert drift==0
        else:actor_updates+=1
        actor.load_state_dict(learner.state_dict())
        row=dict(update=update,actor_updates=actor_updates,mean_reward=float(rec['rewards'].mean()),
                 actor_parameter_l2=drift,falls=int(rec['falls'].sum()),episode_completions=int(rec['dones'].sum()),
                 advantage_mean=float(adv.mean()),advantage_std=float(adv.std()),value_loss=float(np.mean(value_losses)),
                 actor_loss=float(np.mean(losses)) if losses else None,kl=float(np.mean(kls)) if kls else None,
                 clip_fraction=float(np.mean(clips)) if clips else None,execution_bound_fraction=float(((rec['actions']-rec['source']).abs()>.1).float().mean()))
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('UPDATE',row,flush=True)
        if update==5 or actor_updates and actor_updates%10==0:
            ckpt=out/f'model_{update}.pt';torch.save(dict(contract=CONTRACT,model_state_dict=learner.state_dict(),
                actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),source_policy_sha256=sha(POLICY),
                manifest=manifest,update=update,actor_updates=actor_updates),ckpt)
            if update==5:continue
            evaluation=out/f'evaluation_{update}';evaluation.mkdir()
            local=evaluate_local(actor,initial,evaluation/'local');full=evaluate_full(actor,evaluation/'full77')
            receipt=dict(update=update,actor_updates=actor_updates,full_protection=protection(full,baseline),
                         local_gate=local_gate(local,base_local),checkpoint=str(ckpt),general_policy_promoted=False,
                         scope='phase scheduled local specialist; no multimotion or hardware validation')
            save(evaluation/'receipt.json',receipt);(evaluation/'COMPLETE').write_text('All local and complete raw/WM evaluations recorded.\n')
            if not receipt['full_protection']['passed']:
                save(out/'STOPPED.json',dict(reason='full raw/WM protection failed; no further training',**receipt));break
            if receipt['local_gate']['passed']:
                save(out/'LOCAL_CANDIDATE.json',receipt);break
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    save(out/'training_receipt.json',dict(updates=update,actor_updates=actor_updates,wall_seconds=time.perf_counter()-start,
        source_hashes_unchanged=True,rollouts_fully_archived=True,stopped_on_full_gate=(out/'STOPPED.json').exists(),
        local_candidate=(out/'LOCAL_CANDIDATE.json').exists(),general_policy_promoted=False))
    (out/'COMPLETE').write_text('Bounded B1 pilot complete; incumbent unchanged.\n')


if __name__=='__main__':main()
