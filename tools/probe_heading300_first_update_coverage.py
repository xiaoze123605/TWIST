"""Read-only historical coverage and independent conditional gradient probes."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
from export_twist_anyadapter_opentrack_jit import build_actor
from heading300_first_update_verify import RUN
from analyze_heading300_first_update import masks_from_data,csv_write
from recovery1250_core import anchor_error


def masked_statistics(dataset,device):
    old=build_actor(RUN/'model_25.pt').to(device).eval().requires_grad_(False)
    new=build_actor(RUN/'model_26.pt').to(device).eval().requires_grad_(False)
    core=[];head=[];error=[]
    with torch.no_grad():
        for i in range(0,len(dataset['base']),2048):
            b=dataset['base'][i:i+2048].to(device);z=dataset['latent'][i:i+2048].to(device);c=dataset['context'][i:i+2048].to(device)
            a0=old.layerwise_actor(b,z);h0=old.heading_delta(z,c)
            a1=new.layerwise_actor(b,z);h1=new.heading_delta(z,c)
            core.append((a1-a0).cpu());head.append((h1-h0).cpu());error.append((a1+h1-a0-h0).cpu())
    return torch.cat(core).numpy(),torch.cat(head).numpy(),torch.cat(error).numpy()


def gradient_probe(actor,data,indices,device,cap=2048):
    # Fixed deterministic thinning; no RNG draw, optimizer, backward accumulation,
    # gradient clipping or parameter mutation. Conditional means, not actual minibatches.
    indices=indices[np.linspace(0,len(indices)-1,min(cap,len(indices))).astype(int)]
    b=data['base'][indices].to(device);z=data['latent'][indices].to(device);c=data['context'][indices].to(device)
    mu=actor.layerwise_actor(b,z)+actor.heading_delta(z,c)
    ratio=(torch.distributions.Normal(mu,actor.std).log_prob(data['actions'][indices].to(device)).sum(-1)-data['old_logp'][indices].to(device)).exp()
    adv=data['advantages'][indices].to(device)
    ppo=torch.maximum(-adv*ratio,-adv*ratio.clamp(.9,1.1)).mean()
    errors=anchor_error(mu,data['old_mu'][indices].to(device),.087,data['weights'][indices].to(device))
    anchor=1.56*errors.mean()+.01*errors.topk(max(1,math.ceil(.1*len(errors)))).values.mean()
    core=list(actor.adapter[:5].parameters());head=list(actor.adapter[5].parameters());params=core+head
    pg=torch.autograd.grad(ppo,params,retain_graph=True);ag=torch.autograd.grad(anchor,params)
    out=dict(gradient_probe_frames=len(indices),ppo_loss=float(ppo.detach()),anchor_loss=float(anchor.detach()))
    for name,start,end in [('core',0,len(core)),('heading',len(core),len(params))]:
        pn=float(torch.stack([x.square().sum() for x in pg[start:end]]).sum().sqrt())
        an=float(torch.stack([x.square().sum() for x in ag[start:end]]).sum().sqrt())
        dot=float(torch.stack([(x*y).sum() for x,y in zip(pg[start:end],ag[start:end])]).sum())
        out.update({name+'_ppo_gradient':pn,name+'_anchor_gradient':an,name+'_anchor_to_ppo':an/pn if pn else None,
                    name+'_cosine':dot/(pn*an) if pn and an else None})
    return out


def analyze(root,device):
    out=root/'coverage';out.mkdir(exist_ok=False)
    # Actual historical first-update archive: observation metadata exists but
    # actor observations/actions/minibatch indices do not. Never invent them.
    archive=torch.load(RUN/'critic_archives/rollout_026.pt',map_location='cpu')
    obs=archive['critic_obs'].numpy();contact=obs[:,1265:1267]
    assert set(np.unique(contact)).issubset({0.,1.})
    contact=contact.reshape(24,512,2)>0
    switch=np.zeros((24,512),dtype=bool);switch[1:]=np.any(contact[1:]!=contact[:-1],axis=-1)
    near=binary_dilation(switch,structure=np.ones((5,1),dtype=bool))
    meta=dict(context=archive['context'].reshape(-1,2).numpy(),turn=abs(obs[:,7]),contact=contact.reshape(-1,2),
        near_switch=near.reshape(-1),squat=(obs[:,0]<.7)&(obs[:,[11,17]].mean(-1)>.8),
        walking=(obs[:,0]>.74)&(np.linalg.norm(obs[:,4:6],axis=-1)>.15))
    historical=[]
    for mode,name in [(0,'raw'),(2,'wm')]:
        mode_mask=archive['modes'].flatten().numpy()==mode
        for group,mask in masks_from_data(meta).items():
            m=mask&mode_mask;historical.append(dict(provenance='actual historical update26 rollout',mode=name,group=group,
                frames=int(m.sum()),fraction_total=float(m.mean()),fraction_within_mode=float(m.sum()/mode_mask.sum()),
                actor_drift=None,conditional_ppo_gradient=None,missing='actor observations, actions, minibatch indices not archived',
                turn_and_phase_source='first clean future reference in critic; contact indices1265:1267 derived from1160+74+3+1+9*3'))
    csv_write(out/'historical_rollout_coverage.csv',historical)
    audit=torch.load(root/'readonly_audit/audit_states.pt',map_location='cpu')
    contact=audit['contact'].numpy().reshape(-1,512,2)
    switch=np.zeros(contact.shape[:2],dtype=bool);switch[1:]=np.any(contact[1:]!=contact[:-1],axis=-1)
    near=binary_dilation(switch,structure=np.ones((5,1),dtype=bool)).reshape(-1)
    meta=dict(context=audit['context'].numpy(),turn=audit['clean_turn'].numpy(),contact=contact.reshape(-1,2),near_switch=near,
        squat=(audit['clean_height'].numpy()<.7)&(audit['clean_knee'].numpy()>.8),
        walking=(audit['clean_height'].numpy()>.74)&(audit['clean_speed'].numpy()>.15))
    cd,hd,delta=masked_statistics(audit,device)
    actors={k:build_actor(RUN/f'model_{k}.pt').to(device).eval().requires_grad_(False) for k in [25,26]}
    for a in actors.values():a.adapter.requires_grad_(True)
    weights=audit['weights'].numpy();valid=audit['valid'].numpy();stats=[]
    for mode,name in [(0,'raw'),(2,'wm')]:
        mode_mask=(audit['mode'].numpy()==mode)&valid
        for group,mask in masks_from_data(meta).items():
            m=mask&mode_mask;idx=np.flatnonzero(m)
            r=dict(provenance='new independent fixed-policy audit; not historical minibatches',mode=name,group=group,
                frames=len(idx),fraction_total=float(m.mean()),fraction_within_mode=float(m.sum()/mode_mask.sum()),
                weight_mass_within_mode=float(weights[m].sum()/weights[mode_mask].sum()),
                core_drift_rms=float(np.sqrt(np.mean(cd[m]**2))) if len(idx) else None,
                head_drift_rms=float(np.sqrt(np.mean(hd[m]**2))) if len(idx) else None,
                full_teacher_drift_rms=float(np.sqrt(np.mean(delta[m]**2))) if len(idx) else None,
                full_teacher_drift_max=float(abs(delta[m]).max()) if len(idx) else None)
            if len(idx):
                for update,actor in actors.items():
                    probe=gradient_probe(actor,audit,idx,device)
                    r.update({f'model{update}_'+k:v for k,v in probe.items()})
            stats.append(r)
        print('Conditional audit probes',name,flush=True)
    csv_write(out/'independent_audit_groups.csv',stats)
    # Exact saved anchor-bank inputs can be replayed without needing histories:
    # frozen encoder latents and complete teacher targets were saved explicitly.
    pool=torch.load(RUN/'model_25.pt',map_location='cpu')['recovery1250']['replay']['groups']
    bank_rows=[]
    for group,data in pool.items():
        x=data['inputs'];ds=dict(base=x[:,:1155],latent=x[:,1155:1283],context=x[:,-2:])
        c,h,d=masked_statistics(ds,device);mode='wm' if int(group)>=3 else 'raw'
        angle=np.degrees(abs(np.arctan2(x[:,-2].numpy(),1-x[:,-1].numpy())))
        masks={'all':np.ones(len(x),dtype=bool),'turn_gt_0.5':abs(x[:,7].numpy())>.5,
            'squat_processed_proxy':(x[:,0].numpy()<.7)&(x[:,[11,17]].numpy().mean(-1)>.8)}
        for lo,hi in [(0,15),(15,30),(30,60),(60,120),(120,181)]:masks[f'heading_{lo}_{hi}']=(angle>=lo)&(angle<hi)
        for label,m in masks.items():
            if not m.any():continue
            bank_rows.append(dict(mode=mode,stratum=int(group),group=label,frames=int(m.sum()),fraction_in_stratum=float(m.mean()),
                core_drift_rms=float(np.sqrt(np.mean(c[m]**2))),head_drift_rms=float(np.sqrt(np.mean(h[m]**2))),
                full_drift_rms=float(np.sqrt(np.mean(d[m]**2))),full_drift_max=float(abs(d[m]).max()),
                contact_metadata_available=False,temporal_order_available=False))
    csv_write(out/'historical_anchor_bank.csv',bank_rows)
    recorded=[json.loads(x) for x in (RUN/'objective_gradient_metrics.jsonl').read_text().splitlines() if json.loads(x)['update']==26]
    result=dict(historical_first_minibatch_gradient=recorded,historical_coverage=historical,
        independent_audit_groups=stats,anchor_bank=bank_rows,optimizer_steps=0,
        gradient_scope='conditional on-policy anchor plus conditional10% tail; historical replay contact labels are unavailable; no claim of exact historical grouped PPO or total replay gradients',
        conditional_gradient_cap=2048,sparse_groups_caution='low frame counts produce noisy conditional gradients; do not treat them as causal evidence')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();analyze(a.root,a.device)
