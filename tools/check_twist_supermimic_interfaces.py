import argparse
from pathlib import Path
import sys

import isaacgym
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.robot import Robot
from twist_supermimic.data import MotionLibrary, reference
from twist_supermimic.geometry import quat_inv, quat_mul, quat_rotate
from twist_supermimic.learner import save_json
from twist_supermimic.mdp import Task


def main():
    parser = argparse.ArgumentParser(description='使用真实采样验证运动学、留出动作、实时输入与导出')
    parser.add_argument('--directory', type=Path, default=ROOT/'artifacts/twist_supermimic_local/verification')
    parser.add_argument('--library', type=Path, default=ROOT/'artifacts/twist_supermimic_local/motions.pt')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--legacy-observations', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    robot = Robot('cuda:0')
    library = MotionLibrary(args.library, 'cuda:0')
    actual = torch.load(args.directory/'real_samples.pt', map_location='cuda:0', weights_only=False)
    state = actual['state'].detach().requires_grad_(True)
    velocity = torch.zeros_like(state)
    velocity[:, :3] = state[:, 7:10]
    body_angular = quat_rotate(quat_inv(state[:, 3:7]), state[:, 10:13])
    velocity[:, 3:7] = .5*quat_mul(state[:, 3:7], torch.cat((body_angular, torch.zeros_like(body_angular[:, :1])), -1))
    velocity[:, 13:36] = state[:, 36:59]
    (positions, rotations), (dp, dr) = torch.autograd.functional.jvp(
        lambda s: robot.bodies(s)[:2], state, velocity)
    _, _, linear, angular = robot.bodies(state)
    matrix = dr @ rotations.transpose(-1, -2)
    angular_by_gradient = torch.stack((matrix[:, :, 2, 1], matrix[:, :, 0, 2], matrix[:, :, 1, 0]), -1)
    torch.testing.assert_close(dp, linear, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(angular_by_gradient, angular, atol=1e-4, rtol=1e-4)
    groups = {}
    for row in library.rows:
        groups.setdefault(row['group'], set()).add(row['role'])
    if any(len(roles) != 1 for roles in groups.values()):
        raise AssertionError('同一源动作跨越数据划分')
    lookup = {row['name']: row for row in library.rows}
    for name in library.aliases.values():
        if name in lookup and lookup[name]['role'] != 'legacy':
            raise AssertionError('原有保护动作进入了训练数据')
    task = Task(robot, library, noisy=False)
    ids = actual['motion'].squeeze(-1)
    frames = actual['frame'].squeeze(-1)
    ref = library.state(ids, frames)
    quaternion_error = float((ref[:, 3:7].norm(dim=-1)-1).abs().max())
    if quaternion_error > 1e-6:
        raise AssertionError('紧凑动作库读取后没有恢复单位四元数')
    last = library.state(ids, library.lengths[ids]-1)
    tail = library.state(ids, library.lengths[ids]+32)
    torch.testing.assert_close(last[:, :7], tail[:, :7], atol=0, rtol=0)
    torch.testing.assert_close(last[:, 13:36], tail[:, 13:36], atol=0, rtol=0)
    if torch.count_nonzero(tail[:, 7:13]) or torch.count_nonzero(tail[:, 36:]):
        raise AssertionError('动作结束后没有保持末帧并清零参考速度')
    current = task.clean_frame(state.detach(), actual['previous_request'], reference(ref))
    future = library.state(ids, frames+32)
    if torch.equal(ref, future):
        raise AssertionError('真实样本未覆盖参考变化，不能验证时间隔离')
    # 未来参考只供损失计算，策略输入使用当前参考的31维信息。
    torch.testing.assert_close(current[:, :31], reference(ref), atol=0, rtol=0)
    checkpoint = torch.load(args.directory/'training_checkpoint.pt', map_location='cuda:0', weights_only=False)
    from twist_supermimic.networks import policy
    actor = policy().cuda().eval()
    actor.load_state_dict(checkpoint['actor'])
    jit = torch.jit.load(str(args.model or args.directory/'policy_jit.pt'), map_location='cuda:0')
    with torch.no_grad():
        max_error = float((actor(actual['full_obs'])-jit(actual['full_obs'])).abs().max())
    if max_error > 1e-5:
        raise AssertionError('独立重新加载的JIT与检查点不同')
    legacy_errors = {}
    if args.legacy_observations:
        with np.load(args.legacy_observations) as captured:
            legacy = torch.from_numpy(captured['observation'][:16].copy()).cuda()
        if legacy.shape[1] != 7003:
            raise ValueError('该检查需要旧部署入口保存的真实7003维观测')
        with torch.no_grad():
            expected = actor(legacy[:, :1155])
            for width in (7001, 7003):
                error = float((jit(legacy[:, :width])-expected).abs().max())
                if error > 1e-5:
                    raise AssertionError('旧部署输入与原生策略的动作不同')
                legacy_errors[str(width)] = error
    save_json(args.directory/'interfaces.json', dict(passed=True,
        linear_velocity_max_error=float((dp-linear).abs().max()),
        angular_velocity_max_error=float((angular_by_gradient-angular).abs().max()),
        source_groups=len(groups), protected_aliases=len(library.aliases), jit_max_error=max_error,
        reference_quaternion_max_norm_error=quaternion_error, clip_tail_hold_passed=True,
        legacy_input_errors=legacy_errors,
        real_state_samples=len(state), model_input_dimension=1155, current_reference_dimension=31))
    print((args.directory/'interfaces.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
