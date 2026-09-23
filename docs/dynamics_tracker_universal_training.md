# Universal DynamicsTracker training

The seven-motion configuration is a regression fixture only. A deployable
general tracker is trained on the audited 12,248-clip train split and selected
on the disjoint 1,519-clip validation split. The 1,455-clip test split remains
untouched until final reporting; `mocap1/77.pkl` is in that test split.

`tools/run_universal_tracker_training.sh` performs two full-corpus DAgger
passes followed by PPO. Coverage-segment collection assigns all training clips
before repeating one, while random reference start time supplies phase
diversity. PPO enables per-motion difficulty sampling and saves its curriculum
and cumulative motion coverage in every checkpoint, so an exact resume does
not reset the long-run sampling distribution.

The universal task uses the reference at `t+dt`, matching the original TWIST
teacher and the interval over which the selected PD target is applied. This
avoids the one-control-step phase lag that is especially harmful on fast clips.

Run on an otherwise idle RTX 4090:

```bash
cd '/home/hank/TWIST（anyadapter）'
NUM_ENVS=2048 PPO_ENVS=4096 MAX_ITERATIONS=30000 \
  bash tools/run_universal_tracker_training.sh
```

The production default is 4,096 PPO environments and 30,000 PPO iterations,
with a checkpoint every 100 iterations. `MAX_ITERATIONS` is an upper bound;
validation checkpoints should still determine the deployed model.

For a short pipeline preflight, use
`NUM_ENVS=256 PPO_ENVS=256 COVERAGE_STEPS=10`; this deliberately skips complete
corpus coverage and must not be used as the final policy run.

Evaluate a checkpoint against every held-out validation clip from phase zero:

```bash
python tools/evaluate_dynamics_tracker.py \
  --task g1_dynamics_tracker_universal \
  --model RUN/ppo/model_1000.pt \
  --motion_file legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/val.yaml \
  --num_envs 1519 --steps 6000 --episode-length-s 120 --headless \
  --require-all-motions --output RUN/val_model_1000.json
```

Repeat with `--random-phase` to measure recovery away from the first frame.
Choose checkpoints using validation completion, physical failures, joint RMSE,
height RMSE, and torque saturation together. Evaluate the chosen checkpoint
once on all 1,455 test clips; do not tune from test results.
