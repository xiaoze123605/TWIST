import torch


def quat_mul(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_inv(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def quat_exp(v):
    theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat((.5 * torch.sinc(theta / (2 * torch.pi)) * v,
                      torch.cos(.5 * theta)), -1)


def quat_log(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    q = torch.where(q[..., 3:] < 0, -q, q)
    n = torch.linalg.vector_norm(q[..., :3], dim=-1, keepdim=True)
    scale = 2 * torch.atan2(n, q[..., 3:]) / n.clamp_min(1e-7)
    return q[..., :3] * torch.where(n < 1e-7, torch.full_like(scale, 2), scale)


def quat_matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def rpy(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-.999999, .999999)),
                        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), -1)


def increments(s, sn):
    inv = quat_inv(s[..., 3:7])
    return torch.cat((quat_rotate(inv, sn[..., :3]-s[..., :3]),
                      quat_log(quat_mul(inv, sn[..., 3:7])),
                      quat_rotate(inv, sn[..., 7:10]-s[..., 7:10]),
                      quat_rotate(inv, sn[..., 10:13]-s[..., 10:13]),
                      sn[..., 13:]-s[..., 13:]), -1)


def integrate(s, d):
    q = s[..., 3:7]
    return torch.cat((s[..., :3]+quat_rotate(q, d[..., :3]),
                      torch.nn.functional.normalize(quat_mul(q, quat_exp(d[..., 3:6])), dim=-1),
                      s[..., 7:10]+quat_rotate(q, d[..., 6:9]),
                      s[..., 10:13]+quat_rotate(q, d[..., 9:12]), s[..., 13:]+d[..., 12:]), -1)


def finite(name, x):
    if not torch.isfinite(x).all():
        raise FloatingPointError(name + '出现非有限数值')
