"""Evaluate a saved live/EMA candidate against 1250, without stopping training."""
import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ema', action='store_true')
p.add_argument('--export-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
checkpoint = torch.load(a.checkpoint, map_location='cpu')
if 'recovery1250' not in checkpoint:
    raise ValueError('expected a coordinated continuation checkpoint')
reference = ROOT/'reports/recovery_1250/reference_1250_dev'
source_receipt = json.loads((reference/'reference_checkpoint.json').read_text())
if checkpoint['recovery1250']['source_sha256'] != source_receipt['sha256']:
    raise ValueError('candidate has a different source checkpoint')
export_source = a.checkpoint.resolve()
if a.ema:
    state = dict(checkpoint['model_state_dict'])
    for key, value in checkpoint['recovery1250']['ema'].items():
        full = 'layerwise_actor.adapter.' + key
        if full not in state:
            raise KeyError(full)
        state[full] = value
    export_source = a.output/'ema_weights.pt'
    torch.save({'model_state_dict': state}, export_source)
jit = a.output/'policy.pt'
subprocess.run([sys.executable, str(ROOT/'legged_gym/scripts/export_twist_anyadapter_opentrack_jit.py'), str(export_source), '--output', str(jit)], check=True)
if a.export_only:
    print('Exported', jit)
    sys.exit(0)
spec = importlib.util.spec_from_file_location('fixed_eval', ROOT/'tools/evaluate_4900_fixed.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
protocol = json.loads((ROOT/'reports/recovery_4900/protocol.json').read_text())
cases = [c for c in protocol['cases'] if c['phase'] == 'development']
development = a.output/'development'
development.mkdir()
receipt = dict(policy=str(jit.resolve()), policy_sha256=fixed.sha(jit),
               protocol_sha256=fixed.sha(ROOT/'reports/recovery_4900/protocol.json'), cases=cases,
               code={f: fixed.sha(ROOT/f) for f in ['tools/evaluate_4900_fixed.py',
                     'deploy_real/server_low_level_g1_sim.py', 'deploy_real/server_high_level_motion_lib.py']})
(development/'receipt.json').write_text(json.dumps(receipt, indent=2))
def evaluate(case, mode):
    for attempt in range(1, 4):
        dest = development/f'attempt_{attempt}'
        dest.mkdir(exist_ok=True)
        try:
            row = fixed.run_case(case, mode, jit.resolve(), dest.resolve())
            return row, str(dest/case['id']/mode/'metrics.json')
        except Exception as exc:
            print('Evaluation retry', case['id'], mode, attempt, repr(exc), flush=True)
    raise RuntimeError('Three failed attempts: '+case['id']+'/'+mode)
rows, provenance = [], {}
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(evaluate, case, mode) for case in cases for mode in protocol['modes']]
    for future in as_completed(futures):
        row, path = future.result()
        rows.append(row)
        rows.sort(key=lambda r: (r['case'], r['mode']))
        provenance[row['case']+'/'+row['mode']] = path
        (development/'results.json').write_text(json.dumps(rows, indent=2))
        (development/'case_provenance.json').write_text(json.dumps(provenance, indent=2))
        print(row['case'], row['mode'], 'fall', row['fall_proxy'], flush=True)
(development/'COMPLETE').write_text(str(len(rows)))
subprocess.run([sys.executable, str(ROOT/'tools/compare_4900_fixed.py'), '--reference', str(reference), '--candidate', str(a.output/'development'), '--output', str(a.output/'comparison_to_1250.json')], check=True)
print('Compare checkpoint candidates across time; a temporary regression does not stop training.')
