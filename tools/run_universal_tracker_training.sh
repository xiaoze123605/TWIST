#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/hank/anaconda3/envs/twist/bin/python}"
TEACHER="${TEACHER:-$ROOT/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt}"
TEMPLATE="${TEMPLATE:-$ROOT/legged_gym/logs/dynamics_tracker/dagger_b3_cumulative/best.pt}"
MOTIONS="$ROOT/legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml"
RUN_ROOT="${1:-$ROOT/legged_gym/logs/dynamics_tracker/universal_$(date +%Y%m%d_%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-2048}"
PPO_ENVS="${PPO_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
MOTION_COUNT="$("$PYTHON" -c 'import sys,yaml; print(len(yaml.safe_load(open(sys.argv[1]))["motions"]))' "$MOTIONS")"
COVERAGE_SEGMENT_STEPS="${COVERAGE_SEGMENT_STEPS:-50}"
COVERAGE_BATCHES="$(((MOTION_COUNT + NUM_ENVS - 1) / NUM_ENVS))"
COVERAGE_STEPS="${COVERAGE_STEPS:-$((COVERAGE_BATCHES * COVERAGE_SEGMENT_STEPS))}"

mkdir -p "$RUN_ROOT/dagger"
export LD_LIBRARY_PATH="/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Round 0 deliberately traverses every training clip. With 2048 environments,
# enough fixed-length segments are computed to cover all 12,248 clips once.
"$PYTHON" -u "$ROOT/tools/collect_dynamics_dagger.py" \
  --task g1_dynamics_tracker_universal \
  --motion_file "$MOTIONS" \
  --dataset "$RUN_ROOT/dagger/replay.pt" \
  --teacher "$TEACHER" \
  --round 0 --beta 1.0 --steps "$COVERAGE_STEPS" \
  --coverage-segment-steps "$COVERAGE_SEGMENT_STEPS" \
  --num_envs "$NUM_ENVS" --headless \
  --report "$RUN_ROOT/dagger/collect_round0.json"

if (( COVERAGE_STEPS >= COVERAGE_BATCHES * COVERAGE_SEGMENT_STEPS )); then
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); actual=d["latest_round"]["unique_motions_sampled"]; assert actual == expected, (actual, expected)' \
    "$RUN_ROOT/dagger/collect_round0.json" "$MOTION_COUNT"
fi

"$PYTHON" -u "$ROOT/tools/train_dynamics_dagger.py" \
  --task g1_dynamics_tracker_universal \
  --dataset "$RUN_ROOT/dagger/replay.pt" \
  --template-checkpoint "$TEMPLATE" \
  --reference-time-offset-steps 1 \
  --output-checkpoint "$RUN_ROOT/dagger/model_round0.pt" \
  --report "$RUN_ROOT/dagger/train_round0.json" \
  --epochs 30 --samples-per-epoch 1000000 --batch-size 2048 \
  --learning-rate 0.0001 --headless

# The second pass visits states induced by the learned student. Keeping a
# teacher-controlled fraction prevents immediate distribution collapse.
"$PYTHON" -u "$ROOT/tools/collect_dynamics_dagger.py" \
  --task g1_dynamics_tracker_universal \
  --motion_file "$MOTIONS" \
  --dataset "$RUN_ROOT/dagger/replay.pt" \
  --teacher "$TEACHER" \
  --student-checkpoint "$RUN_ROOT/dagger/model_round0.pt" \
  --round 1 --beta 0.5 --steps "$COVERAGE_STEPS" \
  --coverage-segment-steps "$COVERAGE_SEGMENT_STEPS" \
  --num_envs "$NUM_ENVS" --seed 43 --headless \
  --report "$RUN_ROOT/dagger/collect_round1.json"

if (( COVERAGE_STEPS >= COVERAGE_BATCHES * COVERAGE_SEGMENT_STEPS )); then
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); actual=d["latest_round"]["unique_motions_sampled"]; assert actual == expected, (actual, expected)' \
    "$RUN_ROOT/dagger/collect_round1.json" "$MOTION_COUNT"
fi

"$PYTHON" -u "$ROOT/tools/train_dynamics_dagger.py" \
  --task g1_dynamics_tracker_universal \
  --dataset "$RUN_ROOT/dagger/replay.pt" \
  --template-checkpoint "$TEMPLATE" \
  --reference-time-offset-steps 1 \
  --init-checkpoint "$RUN_ROOT/dagger/model_round0.pt" \
  --output-checkpoint "$RUN_ROOT/dagger/model_round1.pt" \
  --report "$RUN_ROOT/dagger/train_round1.json" \
  --epochs 30 --samples-per-epoch 1000000 --batch-size 2048 \
  --learning-rate 0.0001 --headless

"$PYTHON" -u "$ROOT/tools/train_dynamics_tracker.py" \
  --task g1_dynamics_tracker_universal \
  --output "$RUN_ROOT/ppo" \
  --warm-start "$RUN_ROOT/dagger/model_round1.pt" \
  --max_iterations "$MAX_ITERATIONS" --num_envs "$PPO_ENVS" --seed 202 \
  --save-interval 500 --policy-learning-rate 0.00005 \
  --disable-world-model --headless 2>&1 | tee "$RUN_ROOT/ppo_console.log"

echo "$RUN_ROOT"
