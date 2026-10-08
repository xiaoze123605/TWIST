"""CPU contract detection and relative heading features; no robot connection."""
import numpy as np
import torch


def policy_context_dim(policy_path):
    policy=torch.jit.load(policy_path,map_location='cpu').eval()
    with torch.no_grad():
        for size,context in [(7001,0),(7003,2)]:
            try:result=policy(torch.zeros(1,size))
            except RuntimeError:continue
            if result.shape==(1,23) and torch.isfinite(result).all():return context
    raise ValueError('expected7001-D or7003-D OpenTrack policy')


def heading_features(reference_yaw,robot_yaw,offset=0.):
    error=reference_yaw-(robot_yaw-offset)
    return np.asarray([np.sin(error),1.-np.cos(error)],dtype=np.float32)
