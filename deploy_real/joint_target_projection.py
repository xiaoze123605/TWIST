"""Joint-name based PD target bounds shared by deployment and training."""
import xml.etree.ElementTree as ET
import numpy as np


def safe_bounds(lower, upper, names, margin=.02, knee_min=0.):
    lower = np.asarray(lower, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64).copy()
    if lower.shape != (len(names),) or upper.shape != lower.shape:
        raise ValueError('joint bounds and names must have matching dimensions')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(margin) or margin < 0:
        raise ValueError('invalid joint bounds or margin')
    lo, hi = lower + margin, upper - margin
    if knee_min is not None:
        if not np.isfinite(knee_min):
            raise ValueError('knee minimum must be finite')
        for i, name in enumerate(names):
            if 'knee' in name:
                lo[i] = max(lo[i], knee_min)
    if np.any(lo >= hi):
        raise ValueError('projection margin leaves an empty joint interval')
    return lo, hi


def xml_bounds(path, names, margin=.02, knee_min=0.):
    joints = {j.get('name'): j for j in ET.parse(path).getroot().iter('joint') if j.get('name')}
    bounds = []
    for name in names:
        joint = joints[name]
        values = [float(x) for x in joint.get('range', '').split()]
        if len(values) != 2:
            raise ValueError('missing joint range: '+name)
        bounds.append(values)
    bounds = np.asarray(bounds)
    return safe_bounds(bounds[:, 0], bounds[:, 1], names, margin, knee_min)


def project_numpy(target, lower, upper):
    target = np.asarray(target)
    if not np.isfinite(target).all():
        raise ValueError('non-finite PD target')
    return np.clip(target, lower, upper)


def project_torch(target, lower, upper):
    import torch
    return torch.minimum(torch.maximum(target, lower), upper)
