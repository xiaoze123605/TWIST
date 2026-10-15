# Clean Motion-WM + AnyAdapter restart

## Evidence and diagnosis (2026-09-24)

The W&B run `17fej5ag` ended at guarded update 14144 after a CUDA internal
assertion. Its local W&B output and `train_metrics.jsonl` are under
`legged_gym/logs/g1_twist_baseline_adapter/anchored_guarded_long_1000_20260923`.
The online page was unavailable to the analysis environment; the local W&B
run files and training log were used for the numbers below.

| Run / update | Action correction, mean absolute | Policy anchor difference | Action WM loss | Motion difficulty |
| --- | ---: | ---: | ---: | ---: |
| Guarded 500 | 0.020 | 0.011 | 0 (frozen) | 8.85 |
| Guarded 10000 | 0.024 | 0.017 | 0 (frozen) | 5.49 |
| Earlier Motion-WM adapter 500 | 0.187 | absent | 0.717 | 8.98 |
| Earlier Motion-WM adapter 3000 | 0.284 | absent | 0.970 | 8.92 |

The guarded run uses adapter gain 0.25, adapter L2 coefficient 4, policy
anchor coefficient 10, a frozen action WM/history encoder, and no Motion-WM
reference preprocessing. The earlier run uses gain 1, no anchor, a learning
action WM/history encoder, and clean/corrupt/Motion-WM reference modes. These
are causal configuration differences. The 10-fold difference in observed
action correction is consistent with the guarded actor having too little
authority for root and yaw tracking, but does not isolate any one coefficient
as the sole cause.

The older run's `twist_dataset.yaml` contains 15302 unique motions and all
1519 motions in the audited validation split, plus all 1455 motions in the
audited test split. This includes `mocap1/1.pkl` and all eight clips used in
the guarded screen. The guarded run's audited
`train.yaml` contains 12248 motions and has zero overlap with `val.yaml`.
Consequently, the older adapter's better visual result on `mocap1/1.pkl` may
partly reflect direct adapter training-set exposure. It must not be selected
as a generalization winner from that clip. The frozen TWIST base itself is
shared across these runs, so this split establishes adapter-level separation,
not proof that every pretrained component never saw those motions. Its
published `accad/B3___walk1.pkl`
comparison also reports a large root/yaw gain but a 43% joint-RMSE increase
versus frozen TWIST on raw reference.

On the eight-clip paired screen, the guarded update 200 scored 0.9314 and
the long-run update 500 scored 0.9589 (lower is better). The later guarded
checkpoints at 1000, 5000, 10000 and 14000 did not improve that screen;
update 1000 fell on one clip. Training reward alone is not a reliable model
selector because curriculum difficulty changed during the run.

## Code direction

`g1_motion_wm_anyadapter_clean` restores the earlier Motion-WM reference
pipeline and trainable action WM, with gain 1 and no source-policy anchor.
It starts from frozen TWIST weights on the audited training split, not from
the older adapter checkpoint that saw validation motions. Clean references
receive half the episodes to match raw-reference deployment; corrupted and
Motion-WM processed references receive a quarter each. A small action
penalty, a tail penalty, and an exploration-std ceiling target the earlier
joint-tracking and large-correction regression. The autoregressive WM uses
the action recorded in the next observation's history, and its configured
L1 objective is now genuinely L1.

## Pilot and selection

Run from the repository root. W&B is online unless `--no_wandb` is passed.
The first 500 updates are a pilot, not a commitment to 30000 updates.

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python -u \
legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_anyadapter_clean \
  --proj_name g1_motion_wm_anyadapter_clean \
  --exptid clean_motionwm_pilot_500_20260924 \
  --num_envs 4096 --max_iterations 500 --seed 42
```

Screen checkpoints 50 through 500 against frozen TWIST and the guarded
update-200 policy on the same audited validation motions. Continue only if
the candidate has no additional falls, does not materially worsen joint or
root-velocity error, and improves the paired aggregate. Inspect the
per-motion results as well as the aggregate. The existing
`tools/screen_twist_guarded_checkpoint.py` accepts any compatible candidate
checkpoint with `--candidate`; it exports the JIT and applies those gates.
Use a new `--output` directory for each candidate.

If a checkpoint wins, resume to update 1000 with a new experiment ID. For
this task, `--max_iterations` is the absolute update cap on resume; a run
resumed from update 500 to cap 1000 trains 500 additional updates. Repeat
the screen before extending further. The cap is 30000 updates, and W&B
remains online.
