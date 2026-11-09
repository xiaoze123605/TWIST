import argparse
from dataclasses import replace
from pathlib import Path
import sys

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.config import TrainConfig
from twist_supermimic.data import MotionLibrary
from twist_supermimic.robot import Robot
from twist_supermimic.evaluation import evaluate
from twist_supermimic.learner import save_json
from twist_supermimic.networks import WorldModel, policy
from twist_supermimic.baseline import Heading300


def main():
    parser = argparse.ArgumentParser(description='独立GPU进程评价SuperMimic检查点与多步动力学预测')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=('dev', 'legacy', 'test'), default='dev')
    parser.add_argument('--count', type=int)
    parser.add_argument('--motions', nargs='+')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--baseline-heading300', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = TrainConfig(**saved['config'])
    torch.manual_seed(20261109)
    robot = Robot(cfg.device)
    library = MotionLibrary(cfg.library, cfg.device)
    if [(r['name'], r['role'], r['frames']) for r in library.rows] != saved['dataset_names']:
        raise ValueError('评价动作库与检查点中的动作名称、长度或划分不同')
    if args.baseline_heading300:
        actor = Heading300(ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt', cfg.device)
        world = None
    else:
        actor = policy().to(cfg.device).eval()
        actor.load_state_dict(saved['actor'])
        world = WorldModel(cfg.wm_width, cfg.amp).to(cfg.device).eval()
        world.load_state_dict(saved['world'])
    result = evaluate(cfg, actor, robot, library, role=args.role, count=args.count,
                      max_frames=args.max_frames, names=args.motions, world=world, baseline=args.baseline_heading300)
    result['iteration'] = saved['iteration']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
