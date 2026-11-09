import argparse
from pathlib import Path
import subprocess
import sys

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.config import TrainConfig
from twist_supermimic.learner import save_json


def main():
    parser = argparse.ArgumentParser(description='真实执行两轮训练，覆盖完整评价、保存及恢复入口')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/twist_supermimic_local/runner_verification')
    args = parser.parse_args()
    if (args.output/'training_checkpoint.pt').exists():
        raise FileExistsError('请为本次检查选择新的输出目录')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(output=str(args.output), num_envs=4, buffer_steps=96,
                      wm_batch=4, pi_batch=4, micro_batch=2, wm_width=2048,
                      wm_batches=1, critic_epochs=1, critic_minibatches=2,
                      iterations=2, wm_only_iters=0, checkpoint_every=1,
                      evaluate_every=1, evaluation_clips=2, evaluation_envs=2)
    library = torch.load(cfg.library, map_location='cpu', weights_only=False)
    motions = sorted([r for r in library['motions'] if r['role']=='dev' and r['frames'] >= 64], key=lambda r: r['frames'])
    cfg.evaluation_motions = tuple(r['name'] for r in motions[:2])
    del library
    save_json(args.output/'runner_config.json', cfg.to_dict())
    subprocess.run([sys.executable, str(ROOT/'tools/train_twist_supermimic.py'),
                    '--config', str(args.output/'runner_config.json')], check=True)
    subprocess.run([sys.executable, str(ROOT/'tools/train_twist_supermimic.py'),
                    '--resume', str(args.output/'training_checkpoint.pt'), '--iterations', '3'], check=True)
    save_json(args.output/'runner_verified.json', dict(passed=True, long_training=False, iterations=3,
                                                     source='真实GPU物理、完整开发动作评价和独立进程恢复'))


if __name__ == '__main__':
    main()
