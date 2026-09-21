"""Validate all source PKLs and reuse the frozen Motion-WM motion-group splits.

No PKLs are generated or modified. Outputs weighted YAMLs and an audit receipt.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'pose')]
import numpy as np
import yaml
from motion_world_model.dataset import _NumpyCompatibilityUnpickler
from motion_world_model.utils import motion_group_id
from pose.utils.motion_validation import validate_motion_data


def verify_prepared_training_yaml(path):
    """Require the immutable audit receipt before a long WM-DTERA YAML run."""
    path = Path(path).resolve()
    if path.suffix.lower() not in ('.yaml', '.yml'):
        return None
    receipt = path.parent/'audit.json'
    if path.name != 'train.yaml' or not receipt.is_file():
        raise ValueError(
            'WM-DTERA YAML training requires an audited train.yaml beside audit.json; '
            'do not train directly on twist_dataset.yaml'
        )
    report = json.loads(receipt.read_text())
    expected = report.get('split_yaml_sha256', {}).get('train')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not report.get('passed') or report.get('errors') or expected != actual:
        raise ValueError(f'Training dataset audit is missing, failed, or stale: {receipt}')
    if report.get('splits', {}).get('train', {}).get('motions', 0) <= 0:
        raise ValueError(f'Training dataset is empty: {path}')
    return receipt


def split_membership(split_dir):
    assignments, groups = {}, {}
    for split in ('train', 'val', 'test'):
        for item in json.loads((Path(split_dir)/f'{split}_manifest.json').read_text())['motions']:
            name = item['motion_id']
            group = motion_group_id(name)
            if item['group_id'] != group:
                raise ValueError(f'Manifest group mismatch: {name}')
            if name in assignments or (group in groups and groups[group] != split):
                raise ValueError(f'Motion/group leakage: {name}')
            assignments[name], groups[group] = split, split
    return assignments


def prepare(source_yaml, split_dir, out, data_root=None):
    source_yaml, split_dir, out = Path(source_yaml).resolve(), Path(split_dir).resolve(), Path(out).resolve()
    config = yaml.safe_load(source_yaml.read_text())
    configured_root = Path(config['root_path']).expanduser()
    if not configured_root.is_absolute():
        configured_root = (source_yaml.parent/configured_root).resolve()
    source_root = configured_root if data_root is None else Path(data_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f'Motion data root not found: {source_root}')
    assignments = split_membership(split_dir)
    skipped = {r['motion_id']: r['error'] for r in json.loads((split_dir/'metadata.json').read_text())['skipped']}
    weights, duplicates = {}, []
    for item in config['motions']:
        name, weight = item['file'], float(item['weight'])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid weight: {name}')
        if name in weights:
            duplicates.append(name)
        weights[name] = weights.get(name, 0.) + weight
    unknown = set(weights) - set(assignments) - set(skipped)
    missing = set(assignments) - set(weights)
    if unknown or missing:
        raise ValueError(f'Source/WM manifests disagree: unknown={sorted(unknown)}, missing={sorted(missing)}')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(source_yaml=str(source_yaml), configured_source_root=str(configured_root),
                  source_root=str(source_root),
                  source_yaml_sha256=hashlib.sha256(source_yaml.read_bytes()).hexdigest(),
                  duplicate_entries_merged=duplicates, known_wm_exclusions=skipped,
                  errors=[], splits={}, motion_sha256={}, content_overlap=[])
    report['quaternion_norm_deviation_over_005'] = []
    entries = {s: [] for s in ('train','val','test')}
    for split in entries:
        report['splits'][split] = dict(motions=0, frames=0, resident_motion_bytes=0)
    signature, hashes, motion_stats = None, {}, {}
    for index, (name, weight) in enumerate(weights.items()):
        try:
            path = source_root/name
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            data = _NumpyCompatibilityUnpickler(io.BytesIO(raw)).load()
            current = validate_motion_data(data, str(path), expected_dofs=23)
            deviation = float(np.abs(np.linalg.norm(data['root_rot'], axis=-1)-1.).max())
            if deviation > .05:
                report['quaternion_norm_deviation_over_005'].append(dict(motion=name, max_deviation=deviation))
            if signature is None:
                signature = current
            if current != signature:
                raise ValueError('Inconsistent body ordering across dataset')
            if name in skipped:
                continue
            split = assignments[name]
            for other_split, other_name in hashes.get(digest, []):
                if other_split != split:
                    report['content_overlap'].append([other_name, name])
            hashes.setdefault(digest, []).append((split, name))
            report['motion_sha256'][name] = digest
            entries[split].append(dict(file=name, weight=weight))
            summary = report['splits'][split]
            summary['motions'] += 1
            frames = len(data['root_pos'])
            summary['frames'] += frames
            summary['resident_motion_bytes'] += frames * (3+4+3+3+23+23+len(current[1])*3)*4
            motion_stats[name] = (frames, frames * (3+4+3+3+23+23+len(current[1])*3)*4)
        except Exception as exc:
            report['errors'].append(dict(motion=name, error=str(exc)))
        if (index + 1) % 1000 == 0:
            print(f'Validated {index+1}/{len(weights)} motions', flush=True)
    # Frozen WM already saw its train split. Quarantine every group involved
    # in cross-split byte-identical files, rather than relabeling a leaked test
    # example as held out. Keep the complete exclusion receipt.
    quarantine = {motion_group_id(name) for pair in report['content_overlap'] for name in pair}
    report['quarantined_groups'] = sorted(quarantine)
    report['quarantined_motions'] = []
    for split, rows in entries.items():
        kept = []
        for row in rows:
            name = row['file']
            if motion_group_id(name) in quarantine:
                report['quarantined_motions'].append(dict(motion=name, split=split))
                frames, size = motion_stats[name]
                report['splits'][split]['motions'] -= 1
                report['splits'][split]['frames'] -= frames
                report['splits'][split]['resident_motion_bytes'] -= size
            else:
                kept.append(row)
        entries[split] = kept
    report['passed'] = not report['errors']
    for split, rows in entries.items():
        if not rows or sum(r['weight'] for r in rows) <= 0:
            report['passed'] = False
            report['errors'].append(dict(split=split, error='Empty or zero-weight split'))
    if report['passed']:
        for split, rows in entries.items():
            (out/f'{split}.yaml').write_text(yaml.safe_dump(dict(root_path=str(source_root), motions=rows), sort_keys=False))
        report['split_yaml_sha256'] = {s: hashlib.sha256((out/f'{s}.yaml').read_bytes()).hexdigest() for s in entries}
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], splits=report['splits'],
                         errors=len(report['errors']), raw_content_overlaps=len(report['content_overlap']),
                         quarantined_motions=len(report['quarantined_motions']),
                         quaternion_repairs=len(report['quaternion_norm_deviation_over_005']),
                         receipt=str(out/'audit.json')), indent=2))
    if not report['passed']:
        raise ValueError(f'Dataset audit failed; inspect {out}/audit.json. No training YAML emitted.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'legged_gym/motion_data_configs/twist_dataset.yaml')
    parser.add_argument('--wm-splits', type=Path, default=ROOT/'track_dataset/motion_world_model_50hz')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Validated PKL root written into output YAML; useful when the source YAML points at another checkout')
    parser.add_argument('--out', type=Path, required=True, help='New directory for prepared YAMLs and audit')
    args = parser.parse_args()
    prepare(args.source, args.wm_splits, args.out, data_root=args.data_root)
