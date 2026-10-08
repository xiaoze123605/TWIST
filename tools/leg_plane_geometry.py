"""Translation-invariant leg-plane orientation costs."""
import torch

def plane_normals(points):
    return torch.cross(points[...,1,:]-points[...,0,:],points[...,2,:]-points[...,1,:],dim=-1)

def plane_alignment_cost(actual,reference,eps=1e-4):
    an=actual.norm(dim=-1,keepdim=True);rn=reference.norm(dim=-1,keepdim=True)
    dot=((actual/an.clamp(min=eps))*(reference/rn.clamp(min=eps))).sum(-1).clamp(-1.,1.)
    # A degenerate actual leg does not evade a valid target plane.
    return (1.-dot)*(rn.squeeze(-1)>eps).float()
