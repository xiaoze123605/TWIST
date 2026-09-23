"""Format an already-running DynamicsTracker JSONL log without restarting it."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rsl_rl'))
from rsl_rl.runners.dynamics_tracker_log import format_iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path, help='PPO run directory containing train_metrics.jsonl')
    parser.add_argument('--every', type=int, default=10, help='Display every Nth iteration')
    parser.add_argument('--all', action='store_true', help='Replay the whole existing log first')
    args = parser.parse_args()
    if args.every < 1:
        parser.error('--every must be positive')
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    train = manifest['training']
    envs = int(manifest['environment']['env']['num_envs'])
    steps = int(train['runner']['num_steps_per_env'])
    goal = int(train['runner']['max_iterations'])
    path = args.run_dir / 'train_metrics.jsonl'
    with path.open() as stream:
        if not args.all:
            # Show the latest complete record, then follow newly appended ones.
            last = None
            for line in stream:
                if line.strip():
                    last = json.loads(line)
            if last is not None:
                print(format_iteration(last, goal, envs, steps), flush=True)
        while True:
            line = stream.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                metrics = json.loads(line)
            except json.JSONDecodeError:
                stream.seek(stream.tell() - len(line))
                time.sleep(0.5)
                continue
            if args.all or int(metrics['iteration']) % args.every == 0:
                print(format_iteration(metrics, goal, envs, steps), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
