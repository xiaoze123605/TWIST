import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.config import TrainConfig
from twist_supermimic.data import MotionLibrary
from twist_supermimic.robot import Robot
from twist_supermimic.environment import GymEnvironment
from twist_supermimic.learner import Learner, save_json
from twist_supermimic.mdp import Task


def main():
    parser = argparse.ArgumentParser(description='使用真实GPU物理和完整训练批量测量速度与显存，不启动长训练')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/twist_supermimic_local.json')
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--micro-batch', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 10:
        raise ValueError('性能检查轮数必须处于1到10之间')
    cfg = TrainConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    cfg.num_envs, cfg.micro_batch, cfg.output = args.num_envs, args.micro_batch, str(args.output)
    cfg.validate()
    if (args.output/'profile.json').exists():
        raise FileExistsError('该性能结果已经存在，请使用新的输出目录')
    args.output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    readings = []

    def sample_memory():
        index = torch.device(cfg.device).index or 0
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            readings.append(int(result.stdout.strip()))
            stop.wait(.5)

    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    env = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        robot = Robot(cfg.device)
        library = MotionLibrary(cfg.library, cfg.device)
        task = Task(robot, library, cfg.observation_noise)
        env = GymEnvironment(cfg, robot, library, task)
        learner = Learner(cfg, env, task, library)
        torch.cuda.reset_peak_memory_stats()
        learner.collect(cfg.warmup_steps)
        measurements = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.monotonic()
            learner.collect(3)
            torch.cuda.synchronize()
            collected = time.monotonic()
            for _ in range(cfg.wm_batches):
                learner.world_update()
            torch.cuda.synchronize()
            model_updated = time.monotonic()
            learner.policy_update()
            torch.cuda.synchronize()
            finished = time.monotonic()
            measurements.append(dict(total_seconds=finished-started, collection_seconds=collected-started,
                                     world_seconds=model_updated-collected, policy_and_critic_seconds=finished-model_updated))
        result = dict(num_envs=cfg.num_envs, micro_batch=cfg.micro_batch, wm_batch=cfg.wm_batch, pi_batch=cfg.pi_batch,
                      horizon=cfg.horizon, physical_controls=env.physical_controls, measurements=measurements,
                      median_iteration_seconds=statistics.median(m['total_seconds'] for m in measurements),
                      pytorch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      pytorch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu_total_peak_mib_sampled=max(readings), sampling_period_seconds=.5,
                      gpu_total_includes_other_processes=True, long_training_started=False)
        save_json(args.output/'profile.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        stop.set()
        sampler.join()


if __name__ == '__main__':
    main()
