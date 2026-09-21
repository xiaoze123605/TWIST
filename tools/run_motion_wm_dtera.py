"""Fixed, paired MuJoCo experiments; no training or motion-specific tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TWIST = ROOT / 'legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt'
WM = ROOT / 'legged_gym/logs/motion_world_model/full_stable_v2/best.pt'
MODES = [('A', 'raw', 'twist'), ('B', 'wm', 'twist'),
         ('C', 'raw', 'demand_only'), ('D', 'wm', 'demand_only'),
         ('D_gate_off', 'wm', 'off'),
         ('D_demand_confidence', 'wm', 'demand_confidence'), ('D_full', 'wm', 'full'),
         ('D_dyn_only', 'wm', 'dyn_only'), ('D_err_only', 'wm', 'err_only')]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def summarize(rows):
    clean = np.asarray([r['clean'] for r in rows])
    joints = np.asarray([r['joint'] for r in rows])
    rpy = np.asarray([r['rpy'] for r in rows])
    angles = np.arctan2(np.sin(rpy-clean[:, 1:4]), np.cos(rpy-clean[:, 1:4]))
    height = np.asarray([r['height'] for r in rows])
    velocity = np.asarray([r['root_velocity'] for r in rows])
    action = np.asarray([r['final_action'] for r in rows])
    tilt = np.arccos(np.clip(np.cos(rpy[:, 0])*np.cos(rpy[:, 1]), -1, 1))
    fallen = (height < 0.35) | (tilt > 1.0)
    result = dict(frames=len(rows), joint_rmse=float(np.sqrt(np.mean((joints-clean[:, 8:])**2))),
                  root_velocity_rmse=float(np.sqrt(np.mean((velocity-clean[:, 4:7])**2))),
                  height_rmse=float(np.sqrt(np.mean((height-clean[:, 0])**2))),
                  max_tilt=float(tilt.max()), fall_rate=float(fallen.any()),
                  fallen_frame_fraction=float(fallen.mean()),
                  action_rate_l2_per_second=float(np.linalg.norm(np.diff(action, axis=0), axis=1).mean()/0.02),
                  inference_ms_mean=float(np.mean([r['inference_ms'] for r in rows])),
                  inference_ms_p95=float(np.percentile([r['inference_ms'] for r in rows], 95)),
                  reference_pipeline_ms_mean=float(np.mean([r['reference_pipeline_ms'] for r in rows])))
    result.update({name+'_rmse': float(np.sqrt(np.mean(angles[:, i]**2))) for i, name in enumerate(('roll', 'pitch', 'yaw'))})
    if 'dtera' in rows[0]:
        dyn = np.asarray([r['dtera']['gated_delta_dyn'] for r in rows])
        err = np.asarray([r['dtera']['gated_delta_err'] for r in rows])
        norm = np.linalg.norm(dyn, axis=1)*np.linalg.norm(err, axis=1)
        result['branch_cosine_mean'] = float(np.mean(np.sum(dyn*err, axis=1)/np.maximum(norm, 1e-12)))
        result['branch_cancellation_ratio'] = float(1-np.linalg.norm(dyn+err, axis=1).sum()/max((np.linalg.norm(dyn, axis=1)+np.linalg.norm(err, axis=1)).sum(), 1e-12))
    return result


def validate(rows, mode, adapter_gain=1.0):
    assert [r['frame_id'] for r in rows] == list(range(len(rows)))
    for i, r in enumerate(rows):
        assert np.isfinite(np.asarray(r['final_action'])).all()
        if mode == 'wm' and i < 24:
            np.testing.assert_allclose(r['processed'], r['raw'], atol=1e-6)
        if 'dtera' in r:
            dyn, err = np.asarray(r['dynamics_history']), np.asarray(r['tracking_history'])
            assert dyn.shape == (20, 74) and err.shape == (20, 53)
            np.testing.assert_allclose(err[-1, :23], np.asarray(r['processed'])[8:]-r['joint'], atol=1e-6)
            if i:
                np.testing.assert_array_equal(dyn[:-1], np.asarray(rows[i-1]['dynamics_history'])[1:])
                np.testing.assert_array_equal(err[:-1], np.asarray(rows[i-1]['tracking_history'])[1:])
            diag = r['dtera']
            np.testing.assert_allclose(
                r['final_action'],
                np.clip(
                    np.asarray(diag['base_action'])
                    + adapter_gain * np.asarray(diag['applied_delta']),
                    -10,
                    10,
                ),
                atol=1e-5,
            )


def export_policies(args, checkpoint, policies, env):
    policies.mkdir(exist_ok=True)
    for name in ('demand_only', 'off', 'demand_confidence', 'full', 'dyn_only', 'err_only'):
        command = [sys.executable, 'legged_gym/scripts/export_twist_dtera_jit.py', '--ckpt', str(checkpoint),
                   '--preset', args.export_preset,
                   '--out', str(policies/(name+'.pt')), '--device', 'cpu', '--gate_mode',
                   name if name in ('off', 'demand_confidence', 'full') else 'demand_only']
        if args.adapter_gain is not None:
            command += ['--adapter_gain', str(args.adapter_gain)]
        if name in ('demand_confidence', 'full'):
            command += ['--confidence_gate_strength', '1.0']
        if name in ('dyn_only', 'err_only'):
            command += ['--branch_mode', name]
        if args.dynamics_gain is not None:
            command += ['--dynamics_branch_gain', str(args.dynamics_gain)]
        if args.tracking_gain is not None:
            command += ['--tracking_branch_gain', str(args.tracking_gain)]
        with (policies/(name+'.log')).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def check_resume_manifest(out, manifest, resume):
    existing = out/'manifest.json'
    if resume:
        if not existing.is_file():
            raise ValueError('Resume requires an existing manifest; use a new output directory')
        # Normalize tuples to their JSON representation before comparison.
        if json.loads(existing.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Resume manifest mismatch: checkpoint, inputs, code or settings changed; use a new output directory')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserving existing experiment: {out}; use a new output directory')


def saved_policy_config(checkpoint):
    import torch
    loaded = torch.load(checkpoint, map_location='cpu')
    return loaded.get('training_config', {}).get('policy', {})


def run(args):
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    checkpoint = Path(args.ckpt).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'DTERA checkpoint not found: {checkpoint}')
    trained_policy = saved_policy_config(checkpoint)
    effective_adapter_gain = (trained_policy.get('adapter_gain', 1.0)
                              if args.adapter_gain is None else args.adapter_gain)
    preset_dynamics_gain = {'motion_wm_v2': 0.5, 'motion_wm_deploy_v3': 1.0,
                            'motion_wm_deploy_v4': 0.75}.get(
        args.export_preset, 1.0)
    preset_tracking_gain = {'motion_wm_v2': 0.25, 'motion_wm_deploy_v3': 0.5,
                            'motion_wm_deploy_v4': 0.35}.get(
        args.export_preset, 1.0)
    effective_dynamics_gain = (trained_policy.get('dynamics_branch_gain',
                               preset_dynamics_gain)
                               if args.dynamics_gain is None else args.dynamics_gain)
    effective_tracking_gain = (trained_policy.get('tracking_branch_gain',
                               preset_tracking_gain)
                               if args.tracking_gain is None else args.tracking_gain)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    policies = out/'policies'
    motions = ['accad/B3___walk1.pkl']
    if args.motion_file:
        motion_path = Path(args.motion_file).resolve()
        motions = [str(motion_path.relative_to(ROOT/'track_dataset/twist_motion_dataset'))]
    if args.multi:
        import pickle
        import yaml
        sys.path.insert(0, str(ROOT))
        from motion_world_model.utils import motion_group_id
        config = yaml.safe_load((ROOT/'legged_gym/motion_data_configs/twist_dataset.yaml').read_text())
        used = {motion_group_id(item['file']) for item in config['motions']}
        for split in ('train', 'val'):
            entries = json.loads((ROOT/f'track_dataset/motion_world_model_50hz/{split}_manifest.json').read_text())['motions']
            used.update(item['group_id'] for item in entries)
        seen = set()
        motions = []
        # Predeclared before rollout: lexicographic order, distinct source collections,
        # 4–10 s to bound run cost; no performance/pose filtering.
        source_root = ROOT/'track_dataset/twist_motion_dataset'
        for path in sorted(source_root.rglob('*.pkl')):
            relative = path.relative_to(source_root)
            if relative.parts[0] in seen or motion_group_id(str(relative)) in used:
                continue
            with path.open('rb') as stream:
                source = pickle.load(stream)
            duration = (len(source['root_pos'])-1)/float(source['fps'])
            if not 4 <= duration <= 10:
                continue
            seen.add(relative.parts[0])
            motions.append(str(relative))
            if len(motions) == 3:
                break
        assert len(motions) == 3
    seeds = args.seeds if args.seeds is not None else ([42, 43, 44] if args.multi else [42])
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    modes = [m for m in MODES if m[0] in ('A', 'B', 'C', 'D')] if args.multi else MODES
    if args.mode_labels:
        requested = {label.strip() for label in args.mode_labels.split(',') if label.strip()}
        unknown = requested.difference(label for label, _, _ in MODES)
        if unknown:
            raise ValueError(f'Unknown mode labels: {sorted(unknown)}')
        modes = [entry for entry in MODES if entry[0] in requested]
    manifest = dict(motions=motions, seeds=seeds, modes=modes,
                    reference_pipeline='raw input; no artificial corruption', motion_speed=1,
                    motion_sha256={motion: hashlib.sha256((ROOT/'track_dataset/twist_motion_dataset'/motion).read_bytes()).hexdigest() for motion in motions},
                    motion_scale=1, hip_yaw_scale=1, warmup_frames=24,
                    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    twist_sha256=hashlib.sha256(TWIST.read_bytes()).hexdigest(),
                    wm_sha256=hashlib.sha256(WM.read_bytes()).hexdigest(),
                    export_preset=args.export_preset,
                    training_gate=dict(gate_mode='demand_only', confidence_gate_strength=0.0,
                                       independent_branch_gates=args.export_preset != 'legacy',
                                       tracking_demand_mode='smoothstep' if args.export_preset != 'legacy' else 'legacy_exp',
                                       adapter_gain=effective_adapter_gain,
                                       dynamics_branch_gain=effective_dynamics_gain,
                                       tracking_branch_gain=effective_tracking_gain,
                                       dynamics_delta_scale=trained_policy.get(
                                           'dynamics_action_delta_scale', 0.03),
                                       tracking_delta_scale=trained_policy.get(
                                           'tracking_action_delta_scale', 0.03),
                                       dynamics_demand_low=trained_policy.get(
                                           'dynamics_demand_low'),
                                       dynamics_demand_high=trained_policy.get(
                                           'dynamics_demand_high'),
                                       tracking_demand_low=trained_policy.get(
                                           'tracking_demand_low'),
                                       tracking_demand_high=trained_policy.get(
                                           'tracking_demand_high'),
                                       predicted_improvement_mode=trained_policy.get(
                                           'predicted_improvement_mode', 'threshold'),
                                       feedback_residual_scales=trained_policy.get(
                                           'feedback_residual_scales'),
                                       feedback_temperature=trained_policy.get(
                                           'feedback_temperature')),
                    code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                 ('deploy_real/server_high_level_motion_lib.py', 'deploy_real/server_low_level_g1_sim.py',
                                  'deploy_real/twist_anyadapter_runtime.py', 'legged_gym/scripts/export_twist_dtera_jit.py')},
                    unseen_scope='Multi: group absent from configured DTERA twist_dataset.yaml and Motion-WM train/val manifests; frozen original TWIST pretraining provenance unavailable',
                    selection_rule='Multi: lexicographic files, first eligible file per source collection, 4–10 seconds, three collections; no pose/performance filter',
                    fall_definition='episode has height < 0.35 m or vertical-axis tilt > 1.0 rad',
                    full_definition='D retains training demand_only; D_full activates confidence and risk')
    check_resume_manifest(out, manifest, args.resume)
    save(out/'manifest.json', manifest)
    export_policies(args, checkpoint, policies, env)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    redis_log = (out/'redis.log').open('w')
    redis_proc = subprocess.Popen(['redis-server', '--port', str(port), '--bind', '127.0.0.1',
                                   '--save', '', '--appendonly', 'no'], stdout=redis_log, stderr=subprocess.STDOUT)
    results = []
    try:
        time.sleep(0.5)
        if redis_proc.poll() is not None:
            raise RuntimeError('Isolated Redis failed; see redis.log')
        for motion in motions:
            paired = {}
            paired_processed = {}
            for seed in seeds:
                for label, mode, policy in modes:
                    folder = out/motion.replace('/', '__').replace('.pkl', '')/str(seed)/label
                    folder.mkdir(parents=True, exist_ok=args.resume)
                    if args.resume and (folder/'summary.json').is_file() and (folder/'frames.jsonl').is_file():
                        rows = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
                        summary = json.loads((folder/'summary.json').read_text())
                        validate(rows, mode, adapter_gain=effective_adapter_gain)
                        processed_sequence = np.asarray([r['processed'] for r in rows])
                        key = (seed, mode)
                        if key in paired_processed:
                            np.testing.assert_array_equal(processed_sequence, paired_processed[key])
                        paired_processed[key] = processed_sequence
                        sequence = np.asarray([r['raw'] for r in rows])
                        if seed in paired:
                            np.testing.assert_array_equal(sequence, paired[seed])
                        paired[seed] = sequence
                        results.append(summary)
                        print(f'SKIP complete {motion} seed={seed} {label}', flush=True)
                        continue
                    high = [sys.executable, 'deploy_real/server_high_level_motion_lib.py', '--motion_file',
                            str(ROOT/'track_dataset/twist_motion_dataset'/motion), '--device', 'cpu', '--steps', '1',
                            '--motion-speed', '1.0', '--motion-scale', '1.0', '--hip-yaw-scale', '1.0',
                            '--reference-mode', mode, '--wm-checkpoint', str(WM),
                            '--seed', str(seed), '--wait-for-sim-ready', '--sim-ready-timeout', '120', '--redis-port', str(port)]
                    with (folder/'high.log').open('w') as high_log, (folder/'low.log').open('w') as low_log:
                        hp = subprocess.Popen(high, env=env, stdout=high_log, stderr=subprocess.STDOUT)
                        try:
                            deadline = time.monotonic()+120
                            while 'Waiting for sim_ready_g1' not in (folder/'high.log').read_text():
                                if hp.poll() is not None or time.monotonic() > deadline:
                                    raise RuntimeError(f'High-level startup failed: {folder}')
                                time.sleep(0.1)
                            # Source duration is identical to MotionLib's (N-1)/fps.
                            import pickle
                            with (ROOT/'track_dataset/twist_motion_dataset'/motion).open('rb') as stream:
                                source = pickle.load(stream)
                            frames = int(((len(source['root_pos'])-1)/float(source['fps']))/0.02)
                            low = [sys.executable, 'deploy_real/server_low_level_g1_sim.py',
                                   '--policy_path', str(TWIST if policy == 'twist' else policies/(policy+'.pt')),
                                   '--device', 'cpu', '--headless', '--sync-reference', '--redis-port', str(port),
                                   '--seed', str(seed),
                                   '--sim_duration', str(frames*0.02), '--metrics_out', str(folder/'legacy_summary.json'),
                                   '--trace_out', str(folder/'frames.jsonl')]
                            if policy != 'twist':
                                low += ['--require-dtera']
                            if not args.multi and not args.no_video:
                                low += ['--record_video', '--video_path', str(folder/'demo.mp4')]
                            save(folder/'commands.json', dict(high=high, low=low))
                            print(f'RUN {motion} seed={seed} {label} frames={frames}', flush=True)
                            for attempt in range(args.retries + 1):
                                try:
                                    subprocess.run(low, env=env, stdout=low_log, stderr=subprocess.STDOUT,
                                                   check=True, timeout=300)
                                    break
                                except subprocess.CalledProcessError:
                                    if attempt == args.retries:
                                        raise
                                    print(f'RETRY low-level startup ({attempt + 1}/{args.retries})', flush=True)
                                    time.sleep(1)
                        finally:
                            if hp.poll() is None:
                                hp.terminate()
                            hp.wait(timeout=15)
                    rows = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
                    assert len(rows) == frames
                    validate(rows, mode, adapter_gain=effective_adapter_gain)
                    processed_sequence = np.asarray([r['processed'] for r in rows])
                    key = (seed, mode)
                    if key in paired_processed:
                        np.testing.assert_array_equal(processed_sequence, paired_processed[key])
                    paired_processed[key] = processed_sequence
                    if policy != 'twist':
                        assert '[DTERA] Detected 3695-D dual-history policy' in (folder/'low.log').read_text()
                    sequence = np.asarray([r['raw'] for r in rows])
                    if seed in paired:
                        np.testing.assert_array_equal(sequence, paired[seed])
                    paired[seed] = sequence
                    summary = dict(motion=motion, seed=seed, mode=label, full_sequence=summarize(rows),
                                   after_warmup=summarize(rows[24:]))
                    save(folder/'summary.json', summary)
                    results.append(summary)
                    save(out/'results.json', results)
                    print(f'OK {label}: joint={summary["after_warmup"]["joint_rmse"]:.6f}', flush=True)
    finally:
        redis_proc.terminate()
        redis_proc.wait(timeout=15)
        redis_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--ckpt', type=Path, required=True, help='Explicit checkpoint; never silently evaluate the legacy 4800 run')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--motion-file', help='Explicit motion path within twist_motion_dataset')
    parser.add_argument('--no-video', action='store_true', help='Collect state traces without expensive video rendering')
    parser.add_argument('--dynamics-gain', type=float, default=None)
    parser.add_argument('--tracking-gain', type=float, default=None)
    parser.add_argument('--mode-labels', default=None,
                        help='Optional comma-separated subset of experiment labels')
    parser.add_argument('--adapter-gain', type=float, default=None,
                        help='Explicit export ablation; omitted means preserve checkpoint training config')
    parser.add_argument('--export-preset',
                        choices=['legacy', 'motion_wm_v2', 'motion_wm_deploy_v3',
                                 'motion_wm_deploy_v4'],
                        default='legacy')
    parser.add_argument('--resume', action='store_true', help='Skip completed runs and retry incomplete folders')
    parser.add_argument('--retries', type=int, choices=[0], default=0,
                        help='Partial low-level retries are disabled: restart both layers with --resume')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Fixed runtime seeds; defaults to 42 (single) or 42 43 44 (multi)')
    run(parser.parse_args())
