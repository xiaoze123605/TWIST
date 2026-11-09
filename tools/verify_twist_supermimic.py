import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import isaacgym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.config import TrainConfig
from twist_supermimic.data import MotionLibrary, reference
from twist_supermimic.robot import Robot
from twist_supermimic.environment import GymEnvironment
from twist_supermimic.evaluation import prediction_errors, evaluate
from twist_supermimic.geometry import finite, increments
from twist_supermimic.learner import Learner, save_json
from twist_supermimic.mdp import Task


def main():
    parser = argparse.ArgumentParser(description='用真实TWIST动作和GPU物理验证SuperMimic训练')
    parser.add_argument('--library', type=Path, default=ROOT/'artifacts/twist_supermimic_local/motions.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/twist_supermimic_local/verification')
    parser.add_argument('--restore-only', type=Path)
    args = parser.parse_args()
    if not args.restore_only:
        (args.output/'verification.json').unlink(missing_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    cfg = TrainConfig(library=str(args.library), output=str(args.output), num_envs=4, buffer_steps=96,
                      wm_batch=4, pi_batch=4, micro_batch=2, wm_width=2048, wm_batches=1,
                      critic_epochs=1, critic_minibatches=2, iterations=2, wm_only_iters=0,
                      checkpoint_every=2, evaluation_clips=2, evaluation_envs=2)
    cfg.validate()
    result = exercise(args, cfg)
    if args.restore_only:
        return
    summary, checkpoint = result
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--library', str(args.library),
                    '--output', str(args.output), '--restore-only', str(checkpoint)], check=True)
    save_json(args.output/'verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def exercise(args, cfg):
    robot = Robot(cfg.device)
    library = MotionLibrary(cfg.library, cfg.device)
    task = Task(robot, library, noisy=True)
    env = GymEnvironment(cfg, robot, library, task)
    try:
        learner = Learner(cfg, env, task, library)
        if args.restore_only:
            learner.restore(args.restore_only)
            before = learner.policy_updates
            learner.collect(64)
            world = learner.world_update()
            actor = learner.policy_update()
            if learner.policy_updates != before+1:
                raise AssertionError('独立进程没有完成恢复更新')
            save_json(args.output/'independent_restore.json', dict(passed=True, world=world, actor=actor,
                                                                 physical_controls=env.physical_controls))
            return
        learner.collect(64)
        actual = learner.replay.starts(4)
        torch.save(actual, args.output/'real_samples.pt')
        if actual['full_obs'].shape != (4, 1155):
            raise AssertionError('原生输入维度错误')
        expected = task.pd_target(actual['executed'], actual['context'])
        torch.testing.assert_close(expected, actual['pd_target'], atol=0, rtol=0)
        applied = task.action(actual['request'], actual['previous_request'], actual['context'])
        torch.testing.assert_close(applied, actual['executed'], atol=0, rtol=0)
        ref = library.state(actual['motion'].squeeze(-1), actual['frame'].squeeze(-1)+1)
        reward = task.reward(actual['next_state'], ref, actual['executed'], actual['previous_action'])
        torch.testing.assert_close(reward[:, None], actual['reward'], atol=1e-6, rtol=1e-5)
        original_actor = {k: v.clone() for k, v in learner.actor.state_dict().items()}
        world_metrics = []
        for _ in range(4):
            world_metrics.append(learner.world_update())
        learner.world.amp = False
        state = actual['state'][:1].detach()
        context = actual['context'][:1]
        action = actual['executed'][:1].clone().requires_grad_(True)
        weights = increments(state, actual['next_state'][:1]).detach()
        score = (increments(state, learner.world(state, action, context))*weights).sum()
        grad = torch.autograd.grad(score, action)[0]
        finite('真实状态动作梯度', grad)
        index = int(grad.abs().argmax())
        if grad.abs().max() < 1e-10:
            raise AssertionError('动力学模型没有产生动作梯度')
        eps = .01
        direction = torch.zeros_like(action)
        direction.flatten()[index] = eps
        with torch.no_grad():
            plus = (increments(state, learner.world(state, action+direction, context))*weights).sum()
            minus = (increments(state, learner.world(state, action-direction, context))*weights).sum()
            difference = (plus-minus)/(2*eps)
        torch.testing.assert_close(grad.flatten()[index], difference, atol=2e-5, rtol=.05)
        learner.world.amp = cfg.amp
        frozen_world = {k: v.clone() for k, v in learner.world.state_dict().items()}
        policy_metrics = learner.policy_update()
        if not any(not torch.equal(v, learner.actor.state_dict()[k]) for k, v in original_actor.items()):
            raise AssertionError('策略参数没有实际更新')
        for key, value in frozen_world.items():
            torch.testing.assert_close(value, learner.world.state_dict()[key], atol=0, rtol=0)
        learner.iteration = 1
        checkpoint = learner.checkpoint()
        jit_error = learner.export()
        predicted = prediction_errors(learner, batch=4)
        torch.cuda.synchronize()
        summary = dict(passed=True, source='本地TWIST真实动作及Isaac Gym GPU物理', physical_controls=env.physical_controls,
                       world_updates=learner.model_updates, policy_updates=learner.policy_updates,
                       critic_updates=learner.critic_updates, input_dimension=1155, action_dimension=23,
                       horizon=32, world_width=2048, jit_max_error=jit_error,
                       autodiff=float(grad.flatten()[index]), central_difference=float(difference),
                       world=world_metrics, policy=policy_metrics, predictions=predicted,
                       pytorch_peak_allocated=torch.cuda.max_memory_allocated(),
                       pytorch_peak_reserved=torch.cuda.max_memory_reserved(),
                       new_dataset_downloaded=False, long_training_started=False)
    finally:
        env.close()
    return summary, checkpoint


if __name__ == '__main__':
    main()
