"""Screen a continuation checkpoint against the validated TWIST incumbent.

Run with the twist Python environment. Exit status 0 means the candidate
improves on the incumbent on the eight-clip paired screen; 1 means it does
not. The screen is a pilot gate, not a substitute for full validation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from select_twist_adapter_checkpoint import select


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt'
SOURCE = ROOT / 'legged_gym/logs/g1_twist_baseline_adapter/anchored_opt_v2_long/model_1600.pt'
INCUMBENT = ROOT / ('legged_gym/logs/g1_twist_baseline_adapter/'
                    'anchored_guarded_1600_pilot_v1_200_20260923/model_200.pt')
MOTIONS = (
    ('accad/General_A3___Swing_Arms_While_Stand.pkl', 5.6),
    ('mocap1/1.pkl', 8.0),
    ('mydata2/3_seg00.pkl', 8.0),
    ('mydata3/42_seg00.pkl', 8.0),
    ('mydata2/6_seg00.pkl', 8.0),
    ('mydata3/47_seg03.pkl', 8.0),
    ('mocap/13.pkl', 8.0),
    ('mocap/17.pkl', 8.0),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, default=INCUMBENT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-improvement', type=float, default=0.005,
                        help='Required relative score improvement over the incumbent.')
    args = parser.parse_args()
    if not args.candidate.is_file():
        parser.error(f'candidate checkpoint does not exist: {args.candidate}')
    if not args.incumbent.is_file():
        parser.error(f'incumbent checkpoint does not exist: {args.incumbent}')
    if not 0 <= args.min_improvement < 1:
        parser.error('--min-improvement must be in [0,1)')
    output = args.output.resolve()
    if output.exists():
        parser.error(f'output already exists: {output}')
    output.mkdir(parents=True)

    policies = {'source1600': SOURCE, 'incumbent': args.incumbent.resolve(),
                'candidate': args.candidate.resolve()}
    export_script = ROOT / 'legged_gym/scripts/export_twist_anyadapter_opentrack_jit.py'
    jits = {'base': BASE}
    for label, checkpoint in policies.items():
        jit = output / f'{label}.pt'
        command = [sys.executable, str(export_script), str(checkpoint),
                   '--output', str(jit)]
        if label == 'source1600':
            command.extend(('--adapter-gain', '0.25'))
        subprocess.run(command, cwd=ROOT, check=True)
        jits[label] = jit

    command = [sys.executable, str(ROOT / 'tools/compare_twist_adapter_jit.py')]
    for label, jit in jits.items():
        command.extend(('--jit', f'{label}={jit}'))
    for motion, seconds in MOTIONS:
        path = ROOT / 'track_dataset/twist_motion_dataset' / motion
        command.extend(('--motion', f'{path}={seconds:g}'))
    command.extend(('--seed', str(args.seed), '--output', str(output / 'paired')))
    subprocess.run(command, cwd=ROOT, check=True)

    selection = select(json.loads((output / 'paired/results.json').read_text()))
    candidates = {row['label']: row for row in selection['candidates']}
    beats_incumbent = (
        selection['selected_label'] == 'candidate'
        and candidates['candidate']['score']
        < candidates['incumbent']['score'] * (1 - args.min_improvement)
    )
    selection['beats_incumbent'] = beats_incumbent
    selection['min_improvement'] = args.min_improvement
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(f"Selected: {selection['selected_label']}; beats_incumbent={beats_incumbent}; "
          f"selection={output / 'selection.json'}")
    if not beats_incumbent:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
