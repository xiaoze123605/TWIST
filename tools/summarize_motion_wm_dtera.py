"""Summarize every fixed scenario, with paired improvements and no best-run selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = json.loads((args.directory/'results.json').read_text())
    flat = [dict(motion=r['motion'], seed=r['seed'], mode=r['mode'], window=w, **r[w])
            for r in rows for w in ('full_sequence', 'after_warmup')]
    with (args.directory/'all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    modes = list(dict.fromkeys(r['mode'] for r in rows))
    aggregate = {}
    for window in ('full_sequence', 'after_warmup'):
        aggregate[window] = {}
        for mode in modes:
            values = [r[window] for r in rows if r['mode'] == mode]
            aggregate[window][mode] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            aggregate[window][mode]['episodes'] = len(values)
    pairs = []
    for window in ('full_sequence', 'after_warmup'):
        for base, improved in (('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')):
            paired = []
            for r in rows:
                if r['mode'] != base:
                    continue
                other = next((o for o in rows if o['mode'] == improved and o['motion'] == r['motion'] and o['seed'] == r['seed']), None)
                if other is None:
                    continue
                paired.append(dict(motion=r['motion'], seed=r['seed'],
                                   joint_improvement_pct=100*(r[window]['joint_rmse']-other[window]['joint_rmse'])/r[window]['joint_rmse']))
            pairs.append(dict(window=window, comparison=f'{improved} vs {base}', pairs=paired))
    (args.directory/'aggregate.json').write_text(json.dumps(dict(aggregate=aggregate, paired=pairs), indent=2)+'\n')
    branch_rows = []
    for path in sorted(args.directory.glob('*/*/*/frames.jsonl')):
        values = []
        with path.open() as stream:
            for index, line in enumerate(stream):
                value = json.loads(line)
                if index >= 24 and 'dtera' in value:
                    values.append(value['dtera'])
        if not values:
            continue
        dyn = np.asarray([v['gated_delta_dyn'] for v in values])
        err = np.asarray([v['gated_delta_err'] for v in values])
        cosine = np.sum(dyn*err, axis=1)/np.maximum(np.linalg.norm(dyn, axis=1)*np.linalg.norm(err, axis=1), 1e-12)
        entry = dict(run=str(path.parent.relative_to(args.directory)), window='after_warmup',
                     cosine_mean=float(cosine.mean()), opposing_frame_fraction=float((cosine<0).mean()),
                     dynamics_l2_mean=float(np.linalg.norm(dyn, axis=1).mean()),
                     tracking_l2_mean=float(np.linalg.norm(err, axis=1).mean()))
        for name in ('demand', 'confidence', 'effective_confidence', 'delta_risk', 'safety', 'gate', 'dynamics_gate', 'tracking_gate'):
            numbers = np.asarray([v[name] for v in values])
            entry[name] = dict(mean=float(numbers.mean()), p05=float(np.percentile(numbers, 5)), p95=float(np.percentile(numbers, 95)))
        branch_rows.append(entry)
    (args.directory/'branch_diagnostics.json').write_text(json.dumps(branch_rows, indent=2)+'\n')
    lines = ['# Motion-WM + DTERA fixed paired evaluation', '',
             'All scenarios are included. Aggregate values are equal-weight episode means.',
             'A=corrupt+TWIST; B=WM+TWIST; C=corrupt+DTERA; D=WM+DTERA(training demand_only); E=clean+DTERA.',
             'D is also the demand-only ablation. Gate-off means gates equal one, not residuals disabled.',
             'D_full enables demand, confidence and risk; branch-only modes retain training demand-only gates.',
             'Units: joint/roll/pitch/yaw/tilt rad; height m; root velocity m/s; action rate policy-action units/s.',
             'Fall rate is the fraction of episodes crossing height<0.35m or tilt>1rad; it is a threshold proxy, not a contact classifier.',
             'Inference time excludes Redis, JSON logging and diagnostic calls; reference pipeline includes corruption and Motion-WM.', '']
    for window in ('full_sequence', 'after_warmup'):
        lines += [f'## {window}', '', '| Mode | Episodes | Joint RMSE | Velocity RMSE | Height RMSE | Roll | Pitch | Yaw | Mean episode max tilt | Fall rate | Action rate | Policy ms |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for mode, value in aggregate[window].items():
            keys = ('joint_rmse', 'root_velocity_rmse', 'height_rmse', 'roll_rmse', 'pitch_rmse', 'yaw_rmse', 'max_tilt', 'fall_rate', 'action_rate_l2_per_second', 'inference_ms_mean')
            lines.append(f'| {mode} | {value["episodes"]} | '+' | '.join(f'{value[k]:.6f}' for k in keys)+' |')
        lines += ['']
    lines += ['## Evidence', '',
              '- `manifest.json`: fixed motions/seeds/checkpoint hashes and selection rule.',
              '- `all_metrics.csv`: every scenario and both windows, including negative outcomes.',
              '- `aggregate.json`: aggregate metrics and individual paired joint improvements.',
              '- `branch_diagnostics.json`: correction norms, opposing-frame fraction and gate distributions after warmup.',
              '- Per-run `commands.json`, `high.log`, `low.log`, `frames.jsonl`, `summary.json`.',
              '- `frames.jsonl` contains clean/corrupt/WM/processed references, robot state, both histories, corrections, gates and actions.',
              '- “Unseen” is relative to configured DTERA motion groups and Motion-WM train/val groups; original frozen TWIST pretraining provenance is not available.', '']
    (args.directory/'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
