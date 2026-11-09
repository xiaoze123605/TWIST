from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import pytorch_kinematics as pk

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg
from .geometry import quat_matrix


class Robot:
    def __init__(self, device):
        self.cfg = G1MimicPrivCfg()
        self.path = Path(self.cfg.asset.file)
        self.chain = pk.build_chain_from_urdf(self.path.read_text(encoding='utf-8')).to(device=device, dtype=torch.float32)
        self.names = list(self.cfg.init_state.default_joint_angles)
        chain_names = self.chain.get_joint_parameter_names()
        if set(chain_names) != set(self.names) or len(self.names) != 23:
            raise ValueError('机器人关节集合与原生TWIST不一致')
        self.order = torch.tensor([self.names.index(n) for n in chain_names], device=device)
        self.default = torch.tensor([self.cfg.init_state.default_joint_angles[n] for n in self.names], device=device)
        self.kp = self._gains(self.cfg.control.stiffness, device)
        self.kd = self._gains(self.cfg.control.damping, device)
        lo, hi = self.chain.get_joint_limits()
        lookup = {n: (lo[i], hi[i]) for i, n in enumerate(chain_names)}
        self.lower = torch.tensor([lookup[n][0] for n in self.names], device=device)
        self.upper = torch.tensor([lookup[n][1] for n in self.names], device=device)
        self.tracking_count = len(self.cfg.motion.key_bodies)
        self.key_names = list(self.cfg.motion.key_bodies)+['torso_link']
        self.body_names = list(self.chain.get_frame_names(exclude_fixed=False))
        self.ancestors = {}
        self.joint_frames = {}

        def visit(frame, ancestors):
            if frame.joint.joint_type != 'fixed':
                if frame.joint.joint_type != 'revolute':
                    raise ValueError('机器人包含未支持的关节类型')
                ancestors = ancestors + [frame.joint.name]
                self.joint_frames[frame.joint.name] = frame.name
            self.ancestors[frame.name] = ancestors
            for child in frame.children:
                visit(child, ancestors)

        visit(self.chain._root, [])
        self.axes = {n: self.chain.find_joint(n).axis.to(device=device) for n in self.names}
        self.axis_tensor = torch.stack([self.axes[n] for n in self.names])
        self.ancestor_mask = torch.tensor([[float(j in self.ancestors[k]) for j in self.names] for k in self.key_names], device=device)
        parsed = ET.fromstring(self.path.read_text(encoding='utf-8'))
        efforts = {j.attrib['name']: float(j.find('limit').attrib['effort'])
                   for j in parsed.findall('joint') if j.attrib['type'] != 'fixed'}
        self.torque_limits = torch.tensor([efforts[n] for n in self.names], device=device)*self.cfg.rewards.torque_safety_limit

    def _gains(self, values, device):
        return torch.tensor([next(v for k, v in values.items() if k in n) for n in self.names], device=device)

    def bodies(self, state):
        # 使用成熟的URDF运动学库，速度由同一关节轴和位置计算。
        transforms = self.chain.forward_kinematics(state[:, 13:36][:, self.order])
        root_r = quat_matrix(state[:, 3:7])
        bodies = torch.stack([transforms[k].get_matrix() for k in self.key_names], 1)
        joints = torch.stack([transforms[self.joint_frames[j]].get_matrix() for j in self.names], 1)
        offset = (root_r[:, None] @ bodies[:, :, :3, 3:4]).squeeze(-1)
        positions = state[:, None, :3]+offset
        rotations = root_r[:, None] @ bodies[:, :, :3, :3]
        axes = (root_r[:, None] @ (joints[:, :, :3, :3] @ self.axis_tensor[None, :, :, None])).squeeze(-1)
        origins = state[:, None, :3]+(root_r[:, None] @ joints[:, :, :3, 3:4]).squeeze(-1)
        weighted_axes = axes*state[:, 36:59, None]
        angular = state[:, None, 10:13]+torch.einsum('kj,bjc->bkc', self.ancestor_mask, weighted_axes)
        lever = positions[:, :, None]-origins[:, None]
        contributions = torch.cross(weighted_axes[:, None].expand_as(lever), lever, dim=-1)
        velocities = (state[:, None, 7:10]+torch.cross(state[:, None, 10:13].expand_as(offset), offset, dim=-1)
                      +(contributions*self.ancestor_mask[None, :, :, None]).sum(2))
        return positions, rotations, velocities, angular
