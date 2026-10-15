#!/usr/bin/env bash
# Usage: bash tools/run_heading300_original_control.sh <run_name> [training arguments]
# SIGINT/SIGTERM saves at a completed update and exits; no automatic restart.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_NAME="${1:?Provide a run name}"; shift
PY=/home/hank/anaconda3/envs/twist/bin/python
export PYTHONUTF8=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/hank/anaconda3/envs/twist/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" tools/train_heading300_original_control.py --exptid "$RUN_NAME" "$@"
