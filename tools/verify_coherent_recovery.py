"""Read-only audit of archived PPO inputs/actions and candidate acceptance."""
import isaacgym
import argparse,json
from pathlib import Path
import numpy as np
import torch
from coherent_recovery import CONTRACT,teacher_features
from heading_recovery_core import ROOT,POLICY,RecoveryActorCritic
from train_heading_recovery_pilot import compute_gae
from train_coherent_recovery import local_gate
from evaluate_heading_recovery import protection,sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    pilot=r/'pilot';assert (pilot/'COMPLETE').is_file();torch.set_num_threads(1)
    manifest=json.loads((pilot/'manifest.json').read_text());receipt=json.loads((pilot/'training_receipt.json').read_text())
    for name,digest in manifest['hashes'].items():assert sha(ROOT/name)==digest,name
    actor=RecoveryActorCritic().eval();source_policy=torch.jit.load(str(POLICY),map_location='cpu').eval()
    audits=[];samples=0;source_max=0.;prob_max=0.
    for update in range(1,receipt['updates']+1):
        with np.load(pilot/'rollouts'/f'rollout_{update}.npz') as archive:
            x={key:archive[key] for key in archive.files}
        samples+=128*16
        policy=torch.load(pilot/'rollouts'/f'policy_before_{update}.pt',map_location='cpu',weights_only=False)
        assert policy['contract']==CONTRACT and policy['update']==update-1;actor.load_state_dict(policy['model_state_dict'])
        assert x['features'].shape==(128,16,168) and x['legacy'].shape==(128,16,7003)
        assert np.all(x['features'][:,:,112:144]==0) and np.all(x['features'][:,:,167]==0)
        expected=x['source']+np.clip(x['actions']-x['source'],-.1,.1)
        assert np.array_equal(expected,x['executed']);assert np.array_equal(np.clip(expected,-10,10),x['actual_action'])
        assert np.array_equal(x['dones'],x['falls']|x['timeouts']);assert np.array_equal(x['timeouts'],x['phase']+1>=2500)
        for k in x:assert np.isfinite(x[k]).all(),(update,k)
        tensors={k:torch.from_numpy(x[k]) for k in ['rewards','values','next_values','dones','falls']}
        adv,returns=compute_gae(**tensors);assert np.array_equal(adv.numpy(),x['advantages_raw'])
        assert np.array_equal(returns.numpy(),x['returns'])
        norm=(adv-adv.mean())/(adv.std()+1e-8);assert np.array_equal(norm.numpy(),x['advantages_normalized'])
        actual_prob=[];actual_source=[]
        with torch.no_grad():
            for t in range(128):
                feature=torch.from_numpy(x['features'][t]);source=torch.from_numpy(x['source'][t]);actions=torch.from_numpy(x['actions'][t])
                dist=torch.distributions.Normal(actor.mean(feature,source),actor.std)
                actual_prob.append(dist.log_prob(actions).sum(-1).numpy())
                actual_source.append(source_policy(torch.from_numpy(x['legacy'][t])).numpy())
        pm=float(abs(np.stack(actual_prob)-x['old_logp']).max());sm=float(abs(np.stack(actual_source)-x['source']).max())
        # A freshly loaded CPU JIT may choose a different optimized graph than
        # the rollout JIT warmed by the single-env pre-roll. Compare at1e-5;
        # actor/logprob and GAE use identical archived arithmetic and stay exact.
        assert pm==0 and sm<=1e-5,(update,pm,sm);prob_max=max(prob_max,pm);source_max=max(source_max,sm)
        orders=np.load(pilot/'rollouts'/f'minibatch_orders_{update}.npz')['orders'];assert orders.shape==(4,2048)
        for order in orders:assert np.array_equal(np.sort(order),np.arange(2048))
        audits.append(dict(update=update,source_exact=sm==0,source_parity_passed=True,old_logprob_exact=True,gae_exact=True,execution_exact=True,
                           samples=2048,reset_count=int(x['dones'].sum()),falls=int(x['falls'].sum())))
        print('VERIFIED_UPDATE',update,'source_max',sm,'old_logprob_max',pm,flush=True)
    base_full=json.loads((r/'baseline/full77/results.json').read_text());base_local=json.loads((r/'baseline/local/results.json').read_text())
    evaluations=[]
    for e in sorted(pilot.glob('evaluation_*')):
        full=json.loads((e/'full77/results.json').read_text());local=json.loads((e/'local/results.json').read_text())
        full_gate=protection(full,base_full);lg=local_gate(local,base_local)
        original=json.loads((e/'receipt.json').read_text());assert full_gate==original['full_protection'] and lg==original['local_gate']
        for mode in ['raw','wm']:
            y=np.load(e/'full77'/f'{mode}.npz');b=np.load(r/'baseline/full77'/f'{mode}.npz')
            assert len(y['qpos'])==3207
            assert np.array_equal(y['qpos'][:2100],b['qpos'][:2100])
            assert (y['correction'][:2100]==0).all() and (y['correction'][2500:]==0).all()
        evaluations.append(dict(directory=str(e),full_gate=full_gate,local_gate=lg))
    stopped=(pilot/'STOPPED.json').is_file()
    if stopped:
        stop=json.loads((pilot/'STOPPED.json').read_text());assert receipt['updates']==stop['update']
        assert not any(int(p.stem.split('_')[-1])>stop['update'] for p in (pilot/'rollouts').glob('rollout_*.npz'))
    result=dict(archived_updates=receipt['updates'],archived_transitions=samples,source_hashes_unchanged=True,
                all_archived_source_actions_exact=source_max==0,source_parity_atol=1e-5,source_parity_passed=True,all_archived_old_logprobs_exact=True,
                all_archived_gae_and_execution_exact=True,all_minibatch_indices_valid=True,
                source_max_error=source_max,old_logprob_max_error=prob_max,audits=audits,evaluations=evaluations,
                stopped_immediately_on_failed_full_gate=stopped,general_policy_promoted=False,
                limitation='Terminal value scalars and before/after physical states are archived, but terminal feature tensors and per-substep sensors were not separately archived; no exact terminal-observation re-inference claim.')
    save(r/'verification.json',result);(r/'VERIFIED').write_text('All archived source actions,logprobs,GAE,execution,minibatches and gates checked.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['audits','evaluations']},indent=2))


if __name__=='__main__':main()
