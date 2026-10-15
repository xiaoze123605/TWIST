"""Read-only objective gradients when the existing heading head is frozen."""
import json
import torch


def objective_gradients(alg,surrogate,anchoring):
    parameters=list(alg.actor_critic.adapter[:5].parameters())
    p=torch.autograd.grad(surrogate,parameters,retain_graph=True)
    a=torch.autograd.grad(anchoring,parameters,retain_graph=True)
    pn=float(torch.stack([v.detach().square().sum() for v in p]).sum().sqrt())
    an=float(torch.stack([v.detach().square().sum() for v in a]).sum().sqrt())
    dot=float(torch.stack([(x.detach()*y.detach()).sum() for x,y in zip(p,a)]).sum())
    row=dict(update=alg.count+1,core_surrogate_gradient_norm=pn,core_anchor_gradient_norm=an,
             core_anchor_to_surrogate=an/pn if pn else None,
             core_gradient_cosine=dot/(pn*an) if pn and an else None,
             heading_frozen=True,heading_surrogate_gradient_norm=0.,heading_anchor_gradient_norm=0.)
    with (alg.audit_path.parent/'objective_gradient_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
