import argparse
import json
from pathlib import Path
import sys
import time

import isaacgym
import numpy as np
import torch
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.robot import Robot
from twist_supermimic.data import MotionLibrary, reference
from twist_supermimic.geometry import quat_rotate, quat_inv
from twist_supermimic.learner import save_json
from twist_supermimic.mdp import Task
from twist_supermimic.baseline import Heading300


def main():
    parser = argparse.ArgumentParser(description='原生23关节JIT策略，使用现有MuJoCo模型播放本地动作')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, default=ROOT/'artifacts/twist_supermimic_local/motions.pt')
    parser.add_argument('--motion', default='mocap1__77')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--baseline-heading300', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    robot = Robot(args.device)
    library = MotionLibrary(args.library, args.device)
    task = Task(robot, library, noisy=False)
    name = library.aliases.get(args.motion, args.motion)
    names = [r['name'] for r in library.rows]
    if name not in names:
        raise ValueError('动作库不存在该动作：'+name)
    mid = torch.tensor([names.index(name)], device=args.device)
    actor = (Heading300(args.model, args.device) if args.baseline_heading300 else
             torch.jit.load(str(args.model), map_location=args.device).eval())
    model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/g1/g1_sim2sim.xml'))
    if model.nu != 23 or model.nv != 29 or model.nq != 30:
        raise ValueError('MuJoCo模型必须是23关节浮动根部模型')
    joints = [model.joint(n).id for n in robot.names]
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dadr = np.array([model.jnt_dofadr[j] for j in joints])
    actuator_ids = np.array([model.actuator(n).id for n in robot.names])
    model.opt.timestep = robot.cfg.sim.dt
    model.dof_armature[dadr] = robot.cfg.asset.dof_armature
    data = mujoco.MjData(model)
    initial = library.state(mid, torch.zeros_like(mid))[0]
    initial_np = initial.cpu().numpy()
    data.qpos[:3] = initial_np[:3]
    data.qpos[3:7] = initial_np[[6, 3, 4, 5]]
    data.qpos[qadr] = initial_np[13:36]
    data.qvel[:3] = initial_np[7:10]
    data.qvel[3:6] = quat_rotate(quat_inv(initial[3:7]), initial[10:13]).cpu().numpy()
    data.qvel[dadr] = initial_np[36:59]
    mujoco.mj_forward(model, data)
    limit = int(library.lengths[mid].item())+32
    count = limit if args.frames is None else args.frames
    if not 0 < count <= limit:
        raise ValueError('播放长度必须处于完整动作及末尾32帧范围内')
    output = args.output or args.model.parent/('mujoco_'+name+'_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    previous = torch.zeros(1, 23, device=args.device)
    executed_previous = torch.zeros_like(previous)
    context = torch.zeros(1, 28, device=args.device)
    context[:, 0] = 1
    history = processor = None
    kp, kd, caps = (x.cpu().numpy() for x in (robot.kp, robot.kd, robot.torque_limits))

    def state():
        q = torch.tensor(data.qpos[[4, 5, 6, 3]].copy(), device=args.device, dtype=torch.float32)
        angular = quat_rotate(q, torch.tensor(data.qvel[3:6].copy(), device=args.device, dtype=torch.float32))
        return torch.cat((torch.tensor(data.qpos[:3].copy(), device=args.device, dtype=torch.float32), q,
                          torch.tensor(data.qvel[:3].copy(), device=args.device, dtype=torch.float32), angular,
                          torch.tensor(data.qpos[qadr].copy(), device=args.device, dtype=torch.float32),
                          torch.tensor(data.qvel[dadr].copy(), device=args.device, dtype=torch.float32)))[None]

    def run(viewer=None):
        nonlocal previous, executed_previous, history, processor
        failed = False
        actual = 0
        with torch.no_grad(), (output/'frames.jsonl').open('w', encoding='utf-8') as stream:
            for frame in range(count):
                if viewer is not None and not viewer.is_running():
                    break
                start = time.monotonic()
                ref = library.state(mid, torch.full_like(mid, frame))
                before = state()
                obs, history, processor = task.observe(before, previous, reference(ref), history, processor)
                request = actor(obs, before, ref) if args.baseline_heading300 else actor(obs)
                executed = task.action(request, previous, context)
                target = task.pd_target(executed, context)[0].cpu().numpy()
                for _ in range(robot.cfg.control.decimation):
                    data.ctrl[actuator_ids] = np.clip(kp*(target-data.qpos[qadr])-kd*data.qvel[dadr], -caps, caps)
                    mujoco.mj_step(model, data)
                after = state()
                next_reference = library.state(mid, torch.full_like(mid, frame+1))
                reward = task.reward(after, next_reference, executed, executed_previous)
                failed |= bool(task.termination(after, next_reference).item())
                stream.write(json.dumps(dict(frame=frame, state=before[0].cpu().tolist(), next_state=after[0].cpu().tolist(),
                    reference=ref[0].cpu().tolist(), request=request[0].cpu().tolist(), executed=executed[0].cpu().tolist(),
                    pd_target=target.tolist(), reward=float(reward), failed=failed), allow_nan=False)+'\n')
                previous, executed_previous = request, executed
                actual += 1
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0, .02-(time.monotonic()-start)))
        save_json(output/'result.json', dict(model=str(args.model.resolve()), motion=name, frames=actual,
                  requested_frames=count, complete=actual == count, failed=failed, physics='MuJoCo CPU',
                  policy_device=args.device, base_observation_dimension=1155,
                  model_input_dimension=7003 if args.baseline_heading300 else 1155, action_dimension=23))

    if args.headless:
        run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer)
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
