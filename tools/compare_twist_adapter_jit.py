"""Paired MuJoCo evaluation of TWIST and adapter JITs on fixed motion clips.

Each invocation owns an isolated Redis instance and records every policy/clip
combination with the same seed and motion-server settings. This is a screening
test, not a replacement for the full held-out motion validation set.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIGH = ROOT / 'deploy_real/server_high_level_motion_lib.py'
LOW = ROOT / 'deploy_real/server_low_level_g1_sim.py'


def labeled_path(value):
    name, separator, path = value.partition('=')
    if not separator or not name or not Path(path).is_file():
        raise argparse.ArgumentTypeError('expected LABEL=existing_file')
    return name, str(Path(path).resolve())


def motion(value):
    path, separator, seconds = value.rpartition('=')
    if not separator or not Path(path).is_file():
        raise argparse.ArgumentTypeError('expected existing_motion.pkl=seconds')
    try:
        duration = float(seconds)
    except ValueError:
        raise argparse.ArgumentTypeError('motion duration must be numeric')
    if not 0 < duration <= 120:
        raise argparse.ArgumentTypeError('motion duration must be in (0,120]')
    return str(Path(path).resolve()), duration


def free_port():
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


def run_case(python, port, label, jit, motion_path, duration, output, seed,
             reference_mode='raw', trace=False):
    case_dir = output / Path(motion_path).stem / label
    case_dir.mkdir(parents=True, exist_ok=False)
    high_cmd = [python, '-u', str(HIGH), '--motion_file', motion_path,
                '--device', 'cpu', '--steps', '1', '--reference-mode', reference_mode,
                '--wait-for-sim-ready', '--sim-ready-timeout', '30',
                '--redis-port', str(port), '--seed', str(seed)]
    low_cmd = [python, '-u', str(LOW), '--policy_path', jit, '--device', 'cpu',
               '--headless', '--sync-reference', '--sim_duration', str(duration),
               '--metrics_out', str(case_dir / 'metrics.json'),
               '--redis-port', str(port), '--seed', str(seed)]
    if trace:
        low_cmd.extend(['--trace_out', str(case_dir / 'frames.jsonl')])
    with (case_dir / 'high.log').open('w') as high_log, (case_dir / 'low.log').open('w') as low_log:
        high = subprocess.Popen(high_cmd, cwd=ROOT, stdout=high_log,
                                stderr=subprocess.STDOUT)
        try:
            result = subprocess.run(low_cmd, cwd=ROOT, stdout=low_log,
                                    stderr=subprocess.STDOUT, timeout=duration + 90)
        finally:
            high.terminate()
            try:
                high.wait(timeout=5)
            except subprocess.TimeoutExpired:
                high.kill()
                high.wait()
    if result.returncode:
        raise RuntimeError(f'{label} failed on {motion_path}; see {case_dir / "low.log"}')
    metrics = json.loads((case_dir / 'metrics.json').read_text())
    metrics['fall_proxy'] = (metrics['minimum_pelvis_height'] < 0.35 or
                             metrics['max_tilt'] > 1.0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jit', type=labeled_path, action='append', required=True,
                        help='LABEL=path/to/policy-jit.pt; repeat for baseline and candidates')
    parser.add_argument('--motion', type=motion, action='append', required=True,
                        help='path/to/motion.pkl=duration_seconds; repeat for more clips')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-mode', choices=('raw', 'wm'), default='raw')
    parser.add_argument('--trace', action='store_true',
                        help='Record per-frame reference, joint and policy actions')
    args = parser.parse_args()
    if len({name for name, _ in args.jit}) != len(args.jit):
        parser.error('JIT labels must be unique')
    if len({Path(path).stem for path, _ in args.motion}) != len(args.motion):
        parser.error('motion file stems must be unique')
    args.output.mkdir(parents=True, exist_ok=False)
    port = free_port()
    redis_log = (args.output / 'redis.log').open('w')
    server = subprocess.Popen(['redis-server', '--bind', '127.0.0.1', '--port', str(port),
                               '--save', '', '--appendonly', 'no'], cwd=ROOT,
                              stdout=redis_log, stderr=subprocess.STDOUT)
    results = []
    try:
        import redis
        client = redis.Redis(host='127.0.0.1', port=port)
        for _ in range(100):
            try:
                if client.ping():
                    break
            except redis.RedisError:
                time.sleep(0.05)
        else:
            raise RuntimeError('isolated Redis did not start')
        for motion_path, duration in args.motion:
            for label, jit in args.jit:
                metrics = run_case(sys.executable, port, label, jit, motion_path,
                                   duration, args.output, args.seed,
                                   args.reference_mode, args.trace)
                row = dict(motion=motion_path, label=label, jit=jit, **metrics)
                results.append(row)
                print(f'{Path(motion_path).name:40s} {label:12s} '
                      f'joint={metrics["joint_rmse"]:.3f} '
                      f'vel={metrics["root_velocity_rmse"]:.3f} '
                      f'height={metrics["minimum_pelvis_height"]:.3f} '
                      f'tilt={metrics["max_tilt"]:.3f} fall={metrics["fall_proxy"]}', flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        redis_log.close()
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
