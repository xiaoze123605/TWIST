"""Summarize synchronized MuJoCo traces by joint group and action smoothness."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path):
    squared = np.zeros(23, dtype=np.float64)
    action_changes = []
    previous_action = None
    frames = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            reference = np.asarray(row['processed'], dtype=np.float64)
            joint = np.asarray(row['joint'], dtype=np.float64)
            action = np.asarray(row['final_action'], dtype=np.float64)
            if reference.shape != (31,) or joint.shape != (23,) or action.shape != (23,):
                raise ValueError(f'{path}: unexpected reference/joint/action dimensions')
            if not all(np.isfinite(value).all() for value in (reference, joint, action)):
                raise ValueError(f'{path}: non-finite trace values')
            squared += (joint - reference[8:31]) ** 2
            if previous_action is not None:
                action_changes.append(float(np.mean(np.abs(action - previous_action))))
            previous_action = action
            frames += 1
    if not frames:
        raise ValueError(f'{path}: empty trace')
    per_joint = np.sqrt(squared / frames)
    return dict(frames=frames, joint_rmse_per_index=per_joint.tolist(),
                leg_joint_rmse=float(np.sqrt(squared[:12].sum() / (frames * 12))),
                other_joint_rmse=float(np.sqrt(squared[12:].sum() / (frames * 11))),
                mean_abs_action_change=float(np.mean(action_changes))
                if action_changes else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {str(path): summarize(path) for path in args.trace}
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
