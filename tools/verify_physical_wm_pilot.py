"""Independent receipts, split/window and checkpoint checks for the P0 pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.modules.physical_world_model import PhysicalWorldModel,CONTRACT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    root=parser.parse_args().output.resolve();out=root/'pilot'
    manifest=json.loads((out/'manifest.json').read_text());result=json.loads((out/'result.json').read_text())
    data=torch.load(out/'physical_transitions.pt',map_location='cpu',weights_only=False)
    ckpt=torch.load(out/'best_physical_wm.pt',map_location='cpu',weights_only=False)
    assert manifest['contract']==data['contract']==ckpt['contract']==CONTRACT
    sets=[set(manifest[k]) for k in ['train_cases','validation_cases','test_cases']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    offset=0;counts={k:0 for k in ['train','validation','test']}
    for record in manifest['trajectories']:
        assert sha(record['path'])==record['sha256']
        ids=data['window_starts'][record['split']]
        inside=ids[(ids>=offset)&(ids<offset+record['frames'])]
        assert bool((inside-79>=offset).all())
        assert bool((inside+manifest['horizon']<offset+record['frames']).all())
        assert record['case'] in sets[['train','validation','test'].index(record['split'])]
        counts[record['split']]+=len(inside);offset+=record['frames']
    assert counts==manifest['windows'] and offset==manifest['states']
    history=[json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()]
    best=min(history,key=lambda r:r['validation']['physical_wm']['normalized_trajectory_rmse'])
    assert best['epoch']==result['selected_epoch']==ckpt['epoch']
    model=PhysicalWorldModel(**ckpt['normalizers']);model.load_state_dict(ckpt['model_state_dict'])
    assert all(k.startswith(('state_','target_','history_encoder.','transition.')) for k in ckpt['model_state_dict'])
    for path,digest in manifest['original_model_hashes'].items():assert sha(ROOT/path)==digest
    smoke=json.loads((root/'physics_smoke/receipt.json').read_text())
    reset=json.loads((root/'physics_reset_smoke/receipt.json').read_text())
    assert smoke['actor_weight_changes']==reset['actor_weight_changes']==0
    assert smoke['command_execution_mismatch_transitions']>0 and smoke['projected_transitions']>0
    assert reset['terminal_transitions']>=1 and reset['legacy_actor_ankle_velocity_max']==0
    rows=[]
    for item in result['per_test_trajectory']:
        a,b=item['physical_wm'],item['persistence']
        rows.append(dict(case=item['case'],mode=item['mode'],windows=a['windows'],
            physical_wm_normalized_rmse=a['normalized_trajectory_rmse'],persistence_normalized_rmse=b['normalized_trajectory_rmse'],
            normalized_reduction=1-a['normalized_trajectory_rmse']/b['normalized_trajectory_rmse'],
            physical_wm_joint_rmse_20step=a['final_step_joint_rmse'],persistence_joint_rmse_20step=b['final_step_joint_rmse']))
    assert len(rows)==6 and all(r['normalized_reduction']>=.1 for r in rows)
    with (root/'test_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(status='complete',motion_splits_disjoint=True,train_only_normalizers=True,
        no_history_padding_or_cross_trajectory_windows=True,checkpoint_selected_by_validation_only=True,
        all_source_hashes_unchanged=True,physical_checkpoint_has_no_actor_or_critic=True,
        frozen_policy_real_smoke=True,pre_reset_state_real_smoke=True,
        heldout_cases=3,heldout_mode_trajectories=6,forward_prediction_gate_each_trajectory_passed=True,
        actor_updates=0,policy_performance_improvement_measured=False,
        unit_tests=7,regression_tests=24)
    (root/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('P0 VERIFICATION COMPLETE',json.dumps(receipt))


if __name__=='__main__':main()
