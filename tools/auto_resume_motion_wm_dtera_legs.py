#!/usr/bin/env python3
"""Supervise lower-body Motion-WM+DTERA training with safe checkpoint resume.

Each restart writes to a new experiment directory.  This preserves the
training entry point's no-overwrite guarantee while allowing the next process
to load the latest complete checkpoint from an earlier segment.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Dict, Iterable, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_FILE = (
    ROOT
    / "legged_gym/motion_data_configs/"
    "wm_dtera_prepared_20260916_local/train.yaml"
)
CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SUPPORTED_TASKS = {
    "g1_motion_wm_dtera_legs": {
        "dynamics_action_delta_scale": 0.03,
        "tracking_action_delta_scale": 0.03,
        "dynamics_branch_gain": 0.5,
        "tracking_branch_gain": 0.25,
        "residual_warmup_iterations": 1000,
    },
    "g1_motion_wm_dtera_deploy_v3": {
        "dynamics_action_delta_scale": 0.06,
        "tracking_action_delta_scale": 0.06,
        "dynamics_branch_gain": 1.0,
        "tracking_branch_gain": 0.5,
        "residual_warmup_iterations": 500,
    },
    "g1_motion_wm_dtera_deploy_v4": {
        "dynamics_action_delta_scale": 0.05,
        "tracking_action_delta_scale": 0.05,
        "dynamics_branch_gain": 0.75,
        "tracking_branch_gain": 0.35,
        "residual_warmup_iterations": 500,
        "use_predicted_improvement_gate": True,
        "predicted_improvement_mode": "feedback_line_search",
        "feedback_residual_scales": (0.0, 0.25, 0.50, 0.75, 1.0),
        "feedback_temperature": 0.01,
        "predicted_improvement_gate_start_iteration": 500,
        "predicted_improvement_gate_ramp_iterations": 500,
        "predicted_improvement_low": 0.0,
        "predicted_improvement_high": 0.02,
        "tracking_demand_low": 0.30,
        "tracking_demand_high": 0.90,
        "dynamics_demand_low": 0.20,
        "dynamics_demand_high": 0.90,
        "residual_joint_scales": (
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.75, 0.75, 0.75,
            0.50, 0.50, 0.50, 0.50,
            0.50, 0.50, 0.50, 0.50,
        ),
    },
}


def append_log(path: Path, message: str) -> None:
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    with path.open("a") as stream:
        stream.write(timestamped + "\n")


def load_checkpoint(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def validate_checkpoint(
    path: Path,
    filename_iteration: int,
    task: str = "g1_motion_wm_dtera_legs",
) -> Tuple[bool, str]:
    """Reject partial, mismatched, or non-finite checkpoints before resume."""
    try:
        checkpoint = load_checkpoint(path)
        saved_iteration = int(checkpoint.get("iter", filename_iteration))
        if saved_iteration != filename_iteration:
            return False, f"iter={saved_iteration}, filename={filename_iteration}"
        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "ppo_optimizer_state_dict",
            "wm_optimizer_state_dict",
            "risk_optimizer_state_dict",
            "training_config",
        )
        missing = [key for key in required if key not in checkpoint]
        if missing:
            return False, "missing keys: " + ", ".join(missing)
        for name, value in checkpoint["model_state_dict"].items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False, f"non-finite model tensor: {name}"
        policy = checkpoint["training_config"].get("policy", {})
        expected = {
            "base_obs_dim": 1155,
            "history_len": 20,
            "tracking_history_len": 20,
            "freeze_base": True,
            **SUPPORTED_TASKS[task],
        }
        mismatches = {
            key: (policy.get(key), expected_value)
            for key, expected_value in expected.items()
            if policy.get(key) != expected_value
        }
        if mismatches:
            return False, f"policy config mismatch: {mismatches}"
    except Exception as exc:  # A truncated torch archive must not be resumed.
        return False, f"load failed: {type(exc).__name__}: {exc}"
    return True, "ok"


def owned_run_dirs(run_root: Path, root_exptid: str) -> Iterable[Path]:
    base = run_root / root_exptid
    if base.is_dir():
        yield base
    prefix = root_exptid + "__resume_"
    if run_root.is_dir():
        for path in sorted(run_root.iterdir()):
            if path.is_dir() and path.name.startswith(prefix):
                yield path


def latest_valid_checkpoint(
    run_root: Path,
    root_exptid: str,
    log_path: Path,
    task: str = "g1_motion_wm_dtera_legs",
) -> Optional[Tuple[int, Path, Path]]:
    candidates = []
    for run_dir in owned_run_dirs(run_root, root_exptid):
        for path in run_dir.glob("model_*.pt"):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1)), path, run_dir))
    for iteration, path, run_dir in sorted(candidates, reverse=True):
        valid, reason = validate_checkpoint(path, iteration, task=task)
        if valid:
            return iteration, path, run_dir
        append_log(log_path, f"Skipping invalid checkpoint {path}: {reason}")
    return None


def next_segment_name(run_root: Path, root_exptid: str, iteration: int) -> str:
    index = 1
    while True:
        name = f"{root_exptid}__resume_{iteration:05d}_{index:03d}"
        if not (run_root / name).exists():
            return name
        index += 1


def build_command(
    args: argparse.Namespace,
    output_exptid: str,
    remaining: int,
    checkpoint: Optional[Tuple[int, Path, Path]],
) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        str(ROOT / "legged_gym/legged_gym/scripts/train.py"),
        "--task", args.task,
        "--proj_name", args.project,
        "--exptid", output_exptid,
        "--motion_file", str(Path(args.motion_file).resolve()),
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(remaining),
        "--seed", str(args.seed),
        "--device", args.device,
        "--rl_device", args.device,
        "--no_wandb",
        "--fix_action_std",
    ]
    if checkpoint is not None:
        iteration, _, source_dir = checkpoint
        command += [
            "--resume",
            "--resumeid", source_dir.name,
            "--checkpoint", str(iteration),
        ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically resume Motion-WM+DTERA training until a total iteration target"
    )
    parser.add_argument(
        "--task", choices=tuple(SUPPORTED_TASKS),
        default="g1_motion_wm_dtera_legs",
    )
    parser.add_argument("--root-exptid", default="legs_audited_seed42_v2")
    parser.add_argument("--target-iterations", type=int, default=5000)
    parser.add_argument("--project", default="g1_motion_wm_dtera_legs")
    parser.add_argument("--motion-file", default=str(DEFAULT_MOTION_FILE))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--restart-delay", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate the latest checkpoint and print the next command without launching",
    )
    args = parser.parse_args()
    if not SAFE_NAME_RE.fullmatch(args.root_exptid):
        parser.error("--root-exptid may contain only letters, digits, dot, underscore and dash")
    if not SAFE_NAME_RE.fullmatch(args.project):
        parser.error("--project may contain only letters, digits, dot, underscore and dash")
    if args.target_iterations <= 0 or args.num_envs <= 0:
        parser.error("--target-iterations and --num-envs must be positive")
    if args.max_restarts < 0 or args.restart_delay < 0:
        parser.error("restart limits must be nonnegative")
    motion_file = Path(args.motion_file).resolve()
    if not motion_file.is_file():
        parser.error(f"motion file does not exist: {motion_file}")
    return args


def main() -> int:
    args = parse_args()
    run_root = ROOT / "legged_gym/logs" / args.project
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "tools" / f"auto_resume_{args.root_exptid}.log"
    lock_path = ROOT / "tools" / f"auto_resume_{args.root_exptid}.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another supervisor already holds {lock_path}", file=sys.stderr)
        return 2

    append_log(
        log_path,
        f"Supervisor started: task={args.task}, root={args.root_exptid}, "
        f"target={args.target_iterations}, "
        f"num_envs={args.num_envs}, device={args.device}",
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    python_paths = [str(ROOT / name) for name in ("pose", "legged_gym", "rsl_rl")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    restart_count = 0
    while True:
        checkpoint = latest_valid_checkpoint(
            run_root, args.root_exptid, log_path, task=args.task
        )
        current = checkpoint[0] if checkpoint is not None else 0
        if current >= args.target_iterations:
            append_log(log_path, f"Target reached at iteration {current}: {checkpoint[1]}")
            return 0
        remaining = args.target_iterations - current
        if checkpoint is None and not (run_root / args.root_exptid).exists():
            output_exptid = args.root_exptid
        else:
            output_exptid = next_segment_name(
                run_root, args.root_exptid, current
            )
        command = build_command(args, output_exptid, remaining, checkpoint)
        source = str(checkpoint[1]) if checkpoint is not None else "fresh"
        append_log(
            log_path,
            f"Launching segment={output_exptid}, source={source}, "
            f"current={current}, additional_iterations={remaining}",
        )
        append_log(log_path, "Command: " + " ".join(command))
        if args.dry_run:
            return 0

        process = None
        try:
            with log_path.open("a") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_stream.write(line)
                    log_stream.flush()
                exit_code = process.wait()
        except KeyboardInterrupt:
            append_log(log_path, "Manual interrupt received; terminating child and stopping supervisor")
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
            return 130

        new_checkpoint = latest_valid_checkpoint(
            run_root, args.root_exptid, log_path, task=args.task
        )
        new_current = new_checkpoint[0] if new_checkpoint is not None else 0
        if new_current >= args.target_iterations:
            append_log(log_path, f"Training completed at iteration {new_current}")
            return 0
        restart_count += 1
        if restart_count > args.max_restarts:
            append_log(
                log_path,
                f"Giving up after {args.max_restarts} restarts at iteration {new_current}",
            )
            return 1
        append_log(
            log_path,
            f"Child exited code={exit_code}; latest valid iteration={new_current}; "
            f"restart {restart_count}/{args.max_restarts} in {args.restart_delay:.1f}s",
        )
        time.sleep(args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
