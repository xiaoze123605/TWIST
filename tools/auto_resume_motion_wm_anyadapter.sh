#!/usr/bin/env bash

# Foreground auto-resume launcher for Motion-WM + OpenTrack AnyAdapter.
#
# Usage:
#   bash tools/auto_resume_motion_wm_anyadapter.sh [num_envs] [target_iterations]
#
# Defaults:
#   num_envs=4096, target_iterations=20000

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/home/hank/anaconda3/envs/twist/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/legged_gym/legged_gym/scripts/train.py"

TASK="g1_motion_wm_anyadapter"
PROJ="g1_motion_wm_anyadapter"
EXPTID="pilot_gpu_seed42_200"
DEVICE="cuda:0"
SEED=42
NUM_ENVS="${1:-4096}"
TARGET_ITERATIONS="${2:-20000}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"
RESTART_DELAY="${RESTART_DELAY:-30}"
# CPU 12 and 13 are sibling threads of physical core 6.  Kernel crash logs
# repeatedly place Python/NumPy segfaults on CPU 12, and CPU 13 has shown the
# same failure in another process.  Keep every resumed trainer off that core.
CPU_LIST="${CPU_LIST:-0-11,14-31}"

RUN_DIR="$PROJECT_DIR/legged_gym/logs/$PROJ/$EXPTID"
LOG_FILE="$RUN_DIR/auto_resume_training.log"
LOCK_FILE="$RUN_DIR/auto_resume_training.lock"

mkdir -p "$RUN_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another Motion-WM AnyAdapter launcher is already running." >&2
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

latest_iteration() {
    local latest
    latest="$(
        find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
            | sed -nE 's/model_([0-9]+)\.pt/\1/p' \
            | sort -n \
            | tail -n 1
    )"
    echo "${latest:-0}"
}

log "Motion-WM AnyAdapter foreground auto-resume started"
log "task=$TASK run=$PROJ/$EXPTID device=$DEVICE seed=$SEED"
log "num_envs=$NUM_ENVS target_iterations=$TARGET_ITERATIONS"
log "cpu_affinity=$CPU_LIST (isolating unstable sibling CPUs 12,13)"

if ! command -v taskset >/dev/null 2>&1; then
    log "ERROR: taskset is required to enforce CPU isolation"
    exit 1
fi
if ! taskset --cpu-list "$CPU_LIST" true; then
    log "ERROR: invalid or unavailable CPU affinity: $CPU_LIST"
    exit 1
fi

restart_count=0
while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    current="$(latest_iteration)"
    if [ "$current" -ge "$TARGET_ITERATIONS" ]; then
        log "Target reached at model_${current}.pt"
        exit 0
    fi

    remaining=$((TARGET_ITERATIONS - current))
    train_cmd=(
        "$PYTHON_BIN" "$TRAIN_SCRIPT"
        --task "$TASK"
        --proj_name "$PROJ"
        --exptid "$EXPTID"
        --run_name "$EXPTID"
        --device "$DEVICE"
        --rl_device "$DEVICE"
        --num_envs "$NUM_ENVS"
        --seed "$SEED"
        --max_iterations "$remaining"
        --headless
        --no_wandb
    )

    if [ "$current" -gt 0 ]; then
        train_cmd+=(--resume --resumeid "$EXPTID")
        log "Resuming model_${current}.pt; remaining_iterations=$remaining"
    else
        log "No checkpoint found; starting fresh for $remaining iterations"
    fi

    taskset --cpu-list "$CPU_LIST" "${train_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}

    current="$(latest_iteration)"
    if [ "$current" -ge "$TARGET_ITERATIONS" ]; then
        log "Training completed at model_${current}.pt"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    log "Training exited code=$exit_code at model_${current}.pt; restarting in ${RESTART_DELAY}s ($restart_count/$MAX_RESTARTS)"
    sleep "$RESTART_DELAY"
done

log "Reached max restarts before target iteration"
exit 1
