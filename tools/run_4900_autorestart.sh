#!/bin/bash
# Auto-restarting launcher for tools/train_4900_improved.py.
# Usage: run_4900_autorestart.sh <exptid> [training args...]
# Whenever the run dies (crash, OOM, SIGTERM, ...) it restarts with
# --resume-from <latest checkpoint> + the bank/EMA sidecar. Stops on a clean
# exit (rc=0, max_iterations reached) or after 3 consecutive fast failures.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hank/anaconda3/envs/twist/bin/python
EXPTID="$1"; shift
OUT="legged_gym/logs/recovery_4900/${EXPTID}"
LOG="${OUT}_launch.log"

# strip any --resume-from from the user args; the launcher manages it
BASE_ARGS=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  if [ "$a" = "--resume-from" ]; then skip=1; continue; fi
  BASE_ARGS+=("$a")
done

fast_fails=0
for attempt in $(seq 1 50); do
  start=$(date +%s)
  echo "=== attempt $attempt start $(date '+%F %T') ===" >> "$LOG"
  if [ $attempt -gt 1 ]; then
    "$PY" tools/train_4900_improved.py --exptid "$EXPTID" --resume-from "$latest" "${BASE_ARGS[@]}" >> "$LOG" 2>&1
  else
    "$PY" tools/train_4900_improved.py --exptid "$EXPTID" "${BASE_ARGS[@]}" >> "$LOG" 2>&1
  fi
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    echo "=== attempt $attempt exited cleanly (rc=0); training complete ===" >> "$LOG"
    exit 0
  fi
  latest=$(ls -t "$OUT"/model_*.pt 2>/dev/null | grep -v '\.tmp$' | head -1)
  sidecar="$OUT/bank_state_latest.pt"
  if [ -z "$latest" ] || [ ! -f "$sidecar" ]; then
    # Crashed before the first checkpoint (startup). Transient isaacgym init
    # SIGSEGVs live here, so retry fresh and let the fast-fail counter catch
    # genuine config errors (which also fail before any checkpoint).
    if [ $dur -lt 120 ]; then
      fast_fails=$((fast_fails+1))
      echo "=== attempt $attempt failed (rc=$rc) after ${dur}s before first checkpoint (fast fail ${fast_fails}/3) ===" >> "$LOG"
      if [ $fast_fails -ge 3 ]; then
        echo "=== 3 consecutive fast failures; giving up (check config/log) ===" >> "$LOG"
        exit $rc
      fi
    else
      fast_fails=0
      echo "=== attempt $attempt failed (rc=$rc) after ${dur}s before first checkpoint; retrying fresh in 10s ===" >> "$LOG"
    fi
    sleep 10
    continue
  fi
  if [ $dur -lt 120 ]; then
    fast_fails=$((fast_fails+1))
    echo "=== attempt $attempt failed (rc=$rc) after ${dur}s (fast fail ${fast_fails}/3) ===" >> "$LOG"
    if [ $fast_fails -ge 3 ]; then
      echo "=== 3 consecutive fast failures; giving up (check config/log) ===" >> "$LOG"
      exit $rc
    fi
  else
    fast_fails=0
    echo "=== attempt $attempt failed (rc=$rc) after ${dur}s; resuming from $latest in 10s ===" >> "$LOG"
  fi
  sleep 10
done
echo "=== 50 attempts exhausted; giving up ===" >> "$LOG"
exit 1
