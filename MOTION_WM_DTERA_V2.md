# Motion-WM + DTERA v2

Task: `g1_motion_wm_dtera_v2`. The v1 task and existing checkpoints retain their
configuration. This is an experimental response to checkpoint 1400's paired
results, not a demonstrated performance improvement.

## Configuration

- Independent dynamics/tracking gates; tracking smoothstep thresholds 0.30/0.80.
- Dynamics demand thresholds 0.10/0.50; both gate scales 1.0.
- Dynamics/tracking branch gains 0.5/0.25; global adapter gain 1.0.
  With each branch bounded by 0.03, the combined per-action residual bound
  is 0.0225, versus 0.06 in v1. Independent dynamics gating can open further
  than v1's shared gate; this is not uniformly a 0.25 multiplier of v1.
- Clean joint and root-pose reward weights 0.6 -> 0.8; other rewards inherited.
- Frozen Motion GRU and original TWIST; formal corruption, equal episode-mode
  probabilities, 50 Hz, 24-frame reference warmup, 1000-iteration residual ramp.
- Demand-only policy; confidence/risk remain diagnostic.

These changes combine gate, gain and reward hypotheses. If they help, isolate
their effects in subsequent ablations. Reward magnitudes across v1/v2 are not
directly comparable after changing weights.

## Fresh training

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 \
PYTHONPATH='/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_dtera_v2 \
  --proj_name g1_motion_wm_dtera_v2 --exptid fresh_v2_seed42 \
  --motion_file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/twist_dataset.yaml' \
  --num_envs 64 --max_iterations 1500 --seed 42 \
  --device cuda:0 --rl_device cuda:0 --no_wandb --fix_action_std
```

No resume arguments: the DTERA trainable modules and optimizer start fresh.
Checkpoints are saved every 50 iterations. Evaluate 1000 and 1500 against
WM+TWIST and v1 1150/1400 using identical scenes and clean-ground-truth errors.
Do not restart this fresh command in an existing experiment directory.

## Export with the matching preset

```bash
/home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_dtera_jit.py \
  --preset motion_wm_v2 \
  --ckpt legged_gym/logs/g1_motion_wm_dtera_v2/fresh_v2_seed42/model_1500.pt \
  --device cpu
```

The default legacy exporter settings are NOT the v2 training settings.
Motion-WM remains in the high-level reference service, outside this JIT.
For `tools/run_motion_wm_dtera.py`, supply `--export-preset motion_wm_v2`
with `--ckpt` pointing to the v2 checkpoint. Its default remains legacy.

## Mode-specific training diagnostics

`train_metrics.jsonl` now contains `MotionReference/{clean,corrupt,wm}/`:
steps, reward_per_step, clean_joint_rmse, completed_episodes,
episode_length_steps. Step metrics are weighted over actual environment steps
since the previous log; episode lengths are actual completed control steps,
attributed to the mode before reset. Empty modes have zero counts and omit
undefined means. The first interval may include initialization steps.
Warmup frames are included. These on-policy metrics do not provide paired
causal comparisons across reference modes.

## Verification and next evaluation

Actual fresh CUDA smoke: 2 environments, 3 PPO updates, 144 rollout steps.
98 reference frames restored; WM/base weights unchanged, adapter weights
changed, finite 3695-D observations. Proof:
`legged_gym/logs/g1_motion_wm_dtera_v2/fresh_v2_smoke/smoke_proof.json`.
The smoke checkpoint exported with v2 preset passes strict loading and
JIT/eager parity (max difference zero). This does not establish convergence.

The 1400 gain tests reused the original held-out scenarios, so those scenes
now inform development. Preselect additional held-out motion groups for final
testing; do not tune gains on that final set. Exclusion from original TWIST
pretraining cannot be guaranteed without its data provenance. Do not replace
the saved v1 1150 checkpoint until matched evaluation supports the change.
