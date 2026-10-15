"""Read-only gradient and critic diagnostics. Does not draw random numbers."""
import json
import math
import torch

HEADING_BINS=(0,15,30,60,120,180)


def gradient_norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def gradient_groups(actor):
    return dict(core=gradient_norm(actor.adapter[:5].parameters()),
                heading=gradient_norm(actor.adapter[5].parameters()),
                critic=gradient_norm(actor.critic.parameters()))


def objective_gradients(alg,surrogate,anchoring):
    """Observe the first policy minibatch without accumulating parameter .grad."""
    core=list(alg.actor_critic.adapter[:5].parameters())
    head=list(alg.actor_critic.adapter[5].parameters());params=core+head
    policy=torch.autograd.grad(surrogate,params,retain_graph=True)
    anchor=torch.autograd.grad(anchoring,params,retain_graph=True)
    row=dict(update=alg.count+1,scope='first minibatch; un-clipped policy gradients')
    for name,start,end in [('core',0,len(core)),('heading',len(core),len(params))]:
        p=policy[start:end];a=anchor[start:end]
        pn=float(torch.stack([x.detach().square().sum() for x in p]).sum().sqrt())
        an=float(torch.stack([x.detach().square().sum() for x in a]).sum().sqrt())
        dot=float(torch.stack([(x.detach()*y.detach()).sum() for x,y in zip(p,a)]).sum())
        row[name+'_surrogate_gradient_norm']=pn;row[name+'_anchor_gradient_norm']=an
        row[name+'_anchor_to_surrogate']=an/pn if pn else None
        row[name+'_gradient_cosine']=dot/(pn*an) if pn and an else None
    with (alg.audit_path.parent/'objective_gradient_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def residual_groups(context,modes,**signals):
    degrees=torch.rad2deg(torch.atan2(context[...,0],1-context[...,1])).abs().flatten()
    modes=modes.flatten()
    result=[]
    for mode,name in [(0,'raw'),(2,'wm')]:
        for low,high in zip(HEADING_BINS[:-1],HEADING_BINS[1:]):
            mask=(modes==mode)&(degrees>=low)&(degrees<high if high<180 else degrees<=high)
            n=int(mask.sum())
            if not n:continue
            row=dict(reference_mode=name,heading_low_degrees=low,heading_high_degrees=high,frames=n)
            for key,values in signals.items():
                sample=values.detach().flatten()[mask]
                row[key+'_mean']=float(sample.mean())
                row[key+'_rmse']=float(sample.square().mean().sqrt())
            result.append(row)
    return result


@torch.no_grad()
def critic_before(alg):
    storage=alg.storage
    flat=storage.privileged_observations.flatten(0,1)
    source=torch.cat([alg.source.evaluate(x) for x in flat.split(4096)]).reshape_as(storage.values)
    alg._diag_modes=torch.stack(alg.rollout_modes).detach()
    values=storage.values;target=storage.returns
    nxt=torch.cat((values[1:],alg._diag_last_values.unsqueeze(0)),0)
    timeouts=torch.stack(alg._diag_timeouts).reshape_as(storage.values)
    raw=storage.rewards-alg.gamma*values*timeouts
    delta=storage.rewards+alg.gamma*(1-storage.dones.float())*nxt-values
    alg._diag_critic=dict(context=storage.observations[...,-2:],modes=alg._diag_modes,
                         gae_residual_pre=target-values,source_critic_gae_residual=target-source,
                         td_residual_pre=delta)
    if alg.count+1 in (1,25,26,30,40,50):
        archive=dict(update=alg.count+1,critic_obs=flat.cpu(),context=storage.observations[...,-2:].cpu(),
                     modes=alg._diag_modes.cpu(),rewards_raw=raw.cpu(),dones=storage.dones.cpu(),
                     timeouts=timeouts.cpu(),values=values.cpu(),gae_returns=target.cpu(),
                     last_critic_obs=alg._diag_last_critic_obs.cpu(),gamma=alg.gamma,lam=alg.lam,
                     definition='native GAE targets; raw reward de-bootstrapped using native timeout semantics')
        path=alg.audit_path.parent/'critic_archives';path.mkdir(exist_ok=True)
        torch.save(archive,path/f'rollout_{alg.count+1:03d}.pt')


@torch.no_grad()
def critic_after(alg):
    storage=alg.storage
    flat=storage.privileged_observations.flatten(0,1)
    post=torch.cat([alg.actor_critic.evaluate(x) for x in flat.split(4096)]).reshape_as(storage.values)
    diagnostic=alg._diag_critic
    rows=residual_groups(diagnostic.pop('context'),diagnostic.pop('modes'),
                         **diagnostic,gae_residual_post=storage.returns-post)
    with (alg.audit_path.parent/'critic_heading_residuals.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(update=alg.count+1,groups=rows,
                    caution='GAE returns depend on current bootstrap; grouped residuals are not ground truth or proof of observation aliasing'))+'\n')
    alg._diag_timeouts.clear();alg._diag_critic=None


def append_gradients(alg,records,clip_norms):
    result=dict(update=alg.count+1,gradient_global_preclip_max=max(clip_norms),
                gradient_global_preclip_mean=sum(clip_norms)/len(clip_norms),clip_limit=.5,
                critic_lr=alg.optimizer.param_groups[1]['lr'],heading_lr=alg.optimizer.param_groups[2]['lr'])
    for label in ('core','heading','critic'):
        result['gradient_'+label+'_preclip_mean']=sum(r[label] for r in records)/len(records)
        result['gradient_'+label+'_preclip_max']=max(r[label] for r in records)
    with (alg.audit_path.parent/'gradient_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
