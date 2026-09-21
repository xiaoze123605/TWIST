"""Cheap structural checks shared by motion loading and offline preflight."""
import numpy as np


def validate_motion_data(data, source='<motion>', expected_dofs=None):
    required = ('fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f'{source}: missing fields {missing}')
    fps = float(data['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'{source}: fps must be finite and positive')
    arrays = {key: np.asarray(data[key]) for key in required[1:-1]}
    root = arrays['root_pos']
    if root.ndim != 2 or root.shape[1] != 3 or len(root) < 2:
        raise ValueError(f'{source}: root_pos must be [T>=2,3]')
    frames = len(root)
    if arrays['root_rot'].shape != (frames, 4):
        raise ValueError(f'{source}: root_rot must be [T,4]')
    dofs = arrays['dof_pos']
    if dofs.ndim != 2 or len(dofs) != frames or (expected_dofs is not None and dofs.shape[1] != expected_dofs):
        raise ValueError(f'{source}: inconsistent dof_pos shape {dofs.shape}')
    links = tuple(data['link_body_list'])
    if not links or len(set(links)) != len(links):
        raise ValueError(f'{source}: body names must be nonempty and unique')
    if arrays['local_body_pos'].shape != (frames, len(links), 3):
        raise ValueError(f'{source}: local_body_pos does not match frame/body counts')
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{source}: {key} contains NaN or Inf')
    norms = np.linalg.norm(arrays['root_rot'], axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError(f'{source}: root_rot contains a zero quaternion')
    return dofs.shape[1], links


def normalized_continuous_quaternions(quaternions):
    """Match the frozen Motion-WM dataset's normalization and sign convention."""
    q = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms < 1e-8):
        raise ValueError('Invalid root quaternion')
    q /= norms
    if len(q) > 1:
        signs = np.where(np.sum(q[:-1]*q[1:], axis=-1) < 0, -1., 1.)
        q[1:] *= np.cumprod(signs)[:, None]
    return q.astype(np.float32)
