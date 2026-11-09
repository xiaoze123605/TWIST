import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp

from pose.utils.motion_validation import validate_motion_data, normalized_continuous_quaternions
from .geometry import rpy, quat_inv, quat_rotate


class MotionUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy新版保存的模块名称与当前TWIST环境不同，仅在读取该文件时转换名称。
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def reference(state):
    angles = rpy(state[..., 3:7])
    velocity = quat_rotate(quat_inv(state[..., 3:7]), state[..., 7:10])
    angular = quat_rotate(quat_inv(state[..., 3:7]), state[..., 10:13])
    return torch.cat((state[..., 2:3], angles, velocity, angular[..., 2:3], state[..., 13:36]), -1)


def read_motion(path):
    with Path(path).open('rb') as stream:
        data = MotionUnpickler(stream).load()
    validate_motion_data(data, str(path), expected_dofs=23)
    fps = float(data['fps'])
    source_t = np.arange(len(data['root_pos'])) / fps
    times = np.minimum(np.arange(int(np.floor(source_t[-1]*50+1e-6))+1) / 50, source_t[-1])
    q = normalized_continuous_quaternions(data['root_rot'])
    rotation = Slerp(source_t, Rotation.from_quat(q))(times)
    pos = np.stack([np.interp(times, source_t, data['root_pos'][:, i]) for i in range(3)], -1)
    joints = np.stack([np.interp(times, source_t, data['dof_pos'][:, i]) for i in range(23)], -1)
    if len(times) < 3:
        raise ValueError(str(path)+'动作不足三个控制帧')
    vel = np.gradient(pos, .02, axis=0)
    dq = np.gradient(joints, .02, axis=0)
    angular = (rotation[2:]*rotation[:-2].inv()).as_rotvec() / .04
    angular = np.concatenate((angular[:1], angular, angular[-1:]), 0)
    state = np.concatenate((pos, rotation.as_quat(), vel, angular, joints, dq), -1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError(str(path)+'转换结果包含非有限数值')
    return torch.from_numpy(state)


def prepare(source, output, limit=None, protection_manifest=None):
    started = time.monotonic()
    source, output = Path(source), Path(output)
    cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
    if protection_manifest is None:
        protection_manifest = Path(__file__).resolve().parents[1]/'configs/twist_supermimic_local_motions.json'
    protection = json.loads(Path(protection_manifest).read_text(encoding='utf-8'))
    base = Path(cfg['root_path'])
    found, duplicates, missing, seen = [], [], [], set()
    for row in cfg['motions']:
        rel = row['file']
        if rel in seen:
            duplicates.append(rel)
            continue
        seen.add(rel)
        if not (base/rel).is_file():
            missing.append(rel)
            continue
        found.append(rel)
    unexpected = sorted(set(missing)-set(protection['missing_files']))
    if unexpected:
        raise FileNotFoundError('本地动作缺失：'+repr(unexpected[:20]))
    if limit is not None:
        found = found[:limit]
    if len(found) < 10:
        raise ValueError('至少需要十条真实动作建立独立划分')
    aliases = {name: '__'.join(Path(motion).with_suffix('').parts)
               for name, motion in protection['protected_motions'].items()}
    held_out = set(aliases.values())
    reserved_groups = {str(Path(n).parent/Path(n).stem.split('_seg')[0].replace('_mirrored', '').replace('_mirror', ''))
                       for n in found if '__'.join(Path(n).with_suffix('').parts) in held_out}
    groups = sorted({str(Path(n).parent/Path(n).stem.split('_seg')[0].replace('_mirrored', '').replace('_mirror', '')) for n in found})
    random.Random(20261109).shuffle(groups)
    train_end, dev_end = int(.9*len(groups)), int(.95*len(groups))
    roles = {g: 'train' if i < train_end else 'dev' if i < dev_end else 'test' for i, g in enumerate(groups)}
    states, rows, offset = [], [], 0
    for i, rel in enumerate(found):
        state = read_motion(base/rel)
        name = '__'.join(Path(rel).with_suffix('').parts)
        group = str(Path(rel).parent/Path(rel).stem.split('_seg')[0].replace('_mirrored', '').replace('_mirror', ''))
        ref = reference(state)
        knee = ref[:, [11, 17]].mean(-1)
        walking = ((ref[:, 0] > .74) & (ref[:, 4:6].norm(dim=-1) > .15)).nonzero().flatten()
        squat = ((ref[:, 0] < .78) & (knee > .5)).nonzero().flatten()
        rows.append(dict(name=name, source=str(base/rel), group=group,
                         role='legacy' if group in reserved_groups else roles[group],
                         offset=offset, frames=len(state), walking=walking, squat=squat))
        states.append(state.half())
        offset += len(state)
        if (i+1) % 500 == 0:
            print('已转换动作：', i+1, '/', len(found), flush=True)
    payload = dict(format_version=1, fps=50, states=torch.cat(states), motions=rows,
                   source=str(source.resolve()), missing=missing, duplicates=duplicates,
                   split_seed=20261109, aliases=aliases, protection_manifest=str(protection_manifest),
                   split_unit='完整源动作；缺少表演者元数据，未声称按表演者隔离')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {k: v for k, v in payload.items() if k not in ('states', 'motions')}
    report.update(motions=len(rows), frames=offset, seconds=time.monotonic()-started,
                  roles={r: sum(m['role']==r for m in rows) for r in ('train', 'dev', 'test', 'legacy')})
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


class MotionLibrary:
    def __init__(self, path, device):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['format_version'] != 1 or payload['fps'] != 50:
            raise ValueError('动作库格式或控制帧率不一致')
        self.path = str(Path(path).resolve())
        self.states = payload['states'].to(device)
        self.rows = payload['motions']
        self.aliases = payload['aliases']
        self.device = device
        self.offsets = torch.tensor([r['offset'] for r in self.rows], device=device)
        self.lengths = torch.tensor([r['frames'] for r in self.rows], device=device)
        self.roles = {role: torch.tensor([i for i, r in enumerate(self.rows) if r['role']==role], device=device, dtype=torch.long)
                      for role in ('train', 'dev', 'test', 'legacy')}
        self.walking = torch.tensor([i for i, r in enumerate(self.rows) if r['role']=='train' and len(r['walking']) >= 50], device=device)
        self.squat = torch.tensor([i for i, r in enumerate(self.rows) if r['role']=='train' and len(r['squat']) >= 50], device=device)
        self.phase_frames, self.phase_offsets, self.phase_counts = {}, {}, {}
        for phase in ('walking', 'squat'):
            counts = torch.tensor([len(r[phase]) for r in self.rows], device=device)
            self.phase_counts[phase] = counts
            self.phase_offsets[phase] = torch.cat((torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(0)[:-1]))
            self.phase_frames[phase] = torch.cat([r[phase] for r in self.rows]).to(device)
        if len(self.roles['train']) == 0:
            raise ValueError('动作库没有训练动作')
        # 按一秒片段累计真实访问次数与代价，未访问片段保持较高优先级。
        self.bin_counts = torch.zeros(len(self.rows), int(self.lengths.max().item()//50)+1, device=device)
        self.bin_cost = torch.zeros_like(self.bin_counts)

    def state(self, ids, frames):
        last = self.lengths[ids]-1
        beyond = frames > last
        index = self.offsets[ids]+torch.minimum(frames.clamp_min(0), last)
        s = self.states[index].float()
        rotation = torch.nn.functional.normalize(s[..., 3:7], dim=-1)
        zero = torch.zeros_like(s[..., 7:13])
        return torch.cat((s[..., :3], rotation, torch.where(beyond[..., None], zero, s[..., 7:13]),
                          s[..., 13:36], torch.where(beyond[..., None], torch.zeros_like(s[..., 36:]), s[..., 36:])), -1)

    def sample(self, n):
        if len(self.walking) == 0 or len(self.squat) == 0:
            raise ValueError('训练动作库需要包含至少一秒行走与下蹲状态')
        pools = (self.roles['train'], self.walking, self.squat)
        choice = torch.rand(n, device=self.device)
        ids = pools[0][torch.randint(len(pools[0]), (n,), device=self.device)]
        for low, high, pool in ((.5, .75, pools[1]), (.75, 1., pools[2])):
            mask = (choice >= low) & (choice < high)
            ids[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=self.device)]
        frames = (torch.rand(n, device=self.device)*(self.lengths[ids]-1)).long()
        # 一半起点保留均匀分布，一半使用真实执行结果调整片段概率。
        weights = torch.where(self.bin_counts[ids] > 0,
                              self.bin_cost[ids]/self.bin_counts[ids].clamp_min(1), torch.full_like(self.bin_cost[ids], 5))
        valid = torch.arange(weights.shape[1], device=self.device)[None]*50 < self.lengths[ids, None]
        bins = torch.multinomial(weights.clamp_min(.05)*valid, 1).squeeze(-1)
        hard = torch.rand(n, device=self.device) < .5
        frames = torch.where(hard, torch.minimum(bins*50+torch.randint(50, (n,), device=self.device), self.lengths[ids]-1), frames)
        for phase, low, high in (('walking', .5, .75), ('squat', .75, 1.)):
            selected = (choice >= low) & (choice < high)
            picked = ids[selected]
            local = (torch.rand(len(picked), device=self.device)*self.phase_counts[phase][picked]).long()
            frames[selected] = self.phase_frames[phase][self.phase_offsets[phase][picked]+local]
        return ids, frames

    def record(self, ids, frames, reward):
        bins = torch.minimum(frames//50, (self.lengths[ids]-1)//50)
        flat = ids*self.bin_counts.shape[1]+bins
        self.bin_counts.view(-1).index_add_(0, flat, torch.ones_like(reward))
        self.bin_cost.view(-1).index_add_(0, flat, (1-reward/.1).clamp(.05, 5))
