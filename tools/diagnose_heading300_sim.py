"""Reusable synchronous CPU diagnostics; uses the existing physics runner."""
import argparse
import json
from pathlib import Path
import evaluate_4900_fixed as fixed

ROOT = Path(__file__).resolve().parents[1]


def run_diagnostic_case(case, mode, policy, output, *, seed=42, ema=0.,
                        projection=True, core_policy=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy = Path(policy).resolve()
    if fixed.sha(case['motion']) != case['sha256']:
        raise ValueError('reference motion hash changed')
    low_args = ['--trace_out', 'TRACE_PLACEHOLDER', '--dynamics-diagnostics',
                '--anyadapter_ema_alpha', str(ema), '--xml_file',
                str(ROOT/'assets/g1/g1_sim2sim_with_wrist_roll.xml')]
    if projection:
        low_args += ['--target-projection', '--projection-margin', '.02',
                     '--projection-knee-min', '0']
    if core_policy:
        low_args += ['--diagnostics-core-policy', str(Path(core_policy).resolve())]
    receipt = dict(policy=str(policy), policy_sha256=fixed.sha(policy),
                   mode=mode, ema=ema, target_projection=projection, seed=seed,
                   device='cpu', synchronized=True, case=case,
                   core_policy=str(core_policy) if core_policy else None,
                   code={str(f): fixed.sha(ROOT/f) for f in [
                       'tools/diagnose_heading300_sim.py', 'tools/evaluate_4900_fixed.py',
                       'deploy_real/server_low_level_g1_sim.py',
                       'deploy_real/sim_dynamics_diagnostics.py',
                       'deploy_real/twist_anyadapter_runtime.py',
                       'deploy_real/server_high_level_motion_lib.py']})
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    for attempt in range(1, 4):
        dest = output/f'attempt_{attempt}'
        trace = dest/case['id']/mode/'frames.jsonl'
        arguments = [str(trace.resolve()) if x == 'TRACE_PLACEHOLDER' else x
                     for x in low_args]
        try:
            row = fixed.run_case(case, mode, policy, dest, seed=seed,
                                 low_args=arguments)
            if len(trace.read_text().splitlines()) != row['frames']:
                raise ValueError('incomplete diagnostic trace')
            if fixed.sha(policy) != receipt['policy_sha256']:
                raise ValueError('policy changed during evaluation')
            row['trace'] = str(trace.resolve())
            (output/'results.json').write_text(json.dumps(row, indent=2))
            (output/'COMPLETE').write_text(str(row['frames']))
            return row
        except Exception as exc:
            with (output/'failures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(attempt=attempt, error=repr(exc)))+'\n')
    raise RuntimeError('All attempts failed; inspect '+str(output))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['raw', 'wm'], default='wm')
    p.add_argument('--ema', type=float, default=0.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--duration', type=float, default=64.14)
    p.add_argument('--no-projection', action='store_true')
    p.add_argument('--core-policy', type=Path)
    a = p.parse_args()
    if not 0 <= a.ema < 1 or a.duration <= 0:
        p.error('EMA must be in [0,1); duration must be positive')
    protocol = json.loads((ROOT/'reports/recovery_4900/protocol.json').read_text())
    case = dict(next(x for x in protocol['cases'] if x['id'] == 'mocap1__77'),
                seconds=a.duration)
    row = run_diagnostic_case(case, a.mode, a.policy, a.output, seed=a.seed,
                              ema=a.ema, projection=not a.no_projection,
                              core_policy=a.core_policy)
    print({k: row[k] for k in ['joint_rmse', 'root_velocity_rmse', 'yaw_error', 'fall_proxy']})


if __name__ == '__main__':
    main()
