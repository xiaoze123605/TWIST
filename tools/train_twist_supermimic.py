import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import isaacgym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.config import TrainConfig
from twist_supermimic.data import MotionLibrary
from twist_supermimic.robot import Robot
from twist_supermimic.environment import GymEnvironment
from twist_supermimic.evaluation import prediction_errors
from twist_supermimic.learner import Learner, save_json, export_policy
from twist_supermimic.networks import policy
from twist_supermimic.mdp import Task


def main():
    parser = argparse.ArgumentParser(description='使用本地TWIST数据，从头训练原生23关节SuperMimic策略')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--evaluation-motions', nargs='+')
    for name, typ in (('library', str), ('output', str), ('device', str), ('seed', int), ('num-envs', int),
                      ('iterations', int), ('micro-batch', int), ('wm-batch', int), ('pi-batch', int),
                      ('wm-width', int), ('buffer-steps', int), ('warmup-steps', int), ('wm-only-iters', int),
                      ('wm-batches', int), ('checkpoint-every', int), ('evaluate-every', int), ('max-gpu-hours', float)):
        parser.add_argument('--'+name, type=typ)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        settings = dict(resumed['config'])
    for name in TrainConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    cfg = TrainConfig(**settings)
    if args.export_only and not args.resume:
        raise ValueError('导出必须提供真实训练检查点')
    out = Path(cfg.output)
    if args.export_only:
        actor = policy().to(cfg.device)
        actor.load_state_dict(resumed['actor'])
        error = export_policy(actor, resumed['export_observations'].to(cfg.device), out,
                              dict(iteration=resumed['iteration'], policy_updates=resumed['policy_updates'],
                                   source_checkpoint=str(args.resume.resolve())))
        torch.save(dict(world=resumed['world'], config=cfg.to_dict()), out/'world_model_checkpoint.pt')
        print(json.dumps(dict(jit_max_error=error)), flush=True)
        return
    cfg.validate()
    if not args.resume and (out/'training_checkpoint.pt').exists():
        raise FileExistsError('输出目录已有训练，请使用--resume或新的输出目录')
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)
    robot = Robot(cfg.device)
    library = MotionLibrary(cfg.library, cfg.device)
    task = Task(robot, library, cfg.observation_noise)
    env = GymEnvironment(cfg, robot, library, task)
    completed = False
    try:
        learner = Learner(cfg, env, task, library)
        save_json(out/'status.json', dict(status='running', long_training=cfg.iterations >= 12000, dataset='本地TWIST'))
        if args.resume:
            learner.restore(args.resume)
        def evaluation_callback(current):
            rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            folder = out/'evaluation'
            folder.mkdir(exist_ok=True)
            checkpoint = current.checkpoint(out/'evaluation_checkpoint.pt')
            result_path = folder/('%06d.json' % current.iteration)
            # Isaac Gym的GPU仿真在独立进程中评价，避免同进程创建第二个GPU仿真。
            command = [sys.executable, str(ROOT/'tools/evaluate_twist_supermimic.py'),
                            '--checkpoint', str(checkpoint), '--role', 'dev', '--count', str(cfg.evaluation_clips),
                            '--output', str(result_path)]
            if cfg.evaluation_motions:
                command += ['--motions']+list(cfg.evaluation_motions)
            torch.cuda.empty_cache()
            subprocess.run(command, check=True)
            result = json.loads(result_path.read_text(encoding='utf-8'))
            result['iteration'] = current.iteration
            result['training_window_prediction'] = prediction_errors(current)
            if current.iteration % 4000 == 0:
                legacy_path = folder/('%06d_legacy.json' % current.iteration)
                subprocess.run([sys.executable, str(ROOT/'tools/evaluate_twist_supermimic.py'),
                                '--checkpoint', str(checkpoint), '--role', 'legacy', '--output', str(legacy_path)], check=True)
                legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
                result['legacy'] = legacy
            save_json(result_path, result)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            print(json.dumps(dict(iteration=current.iteration, stability_cost=result['stability_cost'],
                                  fall_fraction=result['fall_fraction']), ensure_ascii=False), flush=True)
            return result

        print(json.dumps(learner.run(evaluation_callback), ensure_ascii=False), flush=True)
        completed = True
    finally:
        env.close()
        if out.is_dir():
            save_json(out/'status.json', dict(status='completed' if completed else 'interrupted',
                                             long_training=cfg.iterations >= 12000, dataset='本地TWIST'))


if __name__ == '__main__':
    main()
