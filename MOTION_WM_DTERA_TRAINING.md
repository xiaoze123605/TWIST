# Motion-WM + DTERA training

## Scope

New task: `g1_motion_wm_dtera`. Existing training tasks and deployment behavior
are unchanged. This first stage aligns the student reference distribution with
deployment; it is not a new model architecture or a claim of improved tracking.

- Freeze the original TWIST actor and external Motion GRU; fine-tune DTERA
  using the existing PPO implementation (including its critic/auxiliary losses).
- At each episode reset, independently choose clean / corrupt / WM reference
  with equal probability. This is an initial baseline, not a tuned mixture.
- Use the deployment `formal` corruption implementation exactly once, at 50 Hz.
- WM uses 25-frame histories and current-frame restoration. The first 24 frames
  pass corrupted references through while both DTERA histories continue updating.
- Policy observations and tracking-error history use the processed reference.
  Privileged references and reward targets remain clean.
- Reset per-environment WM history, corruption state, TWIST observation history
  and DTERA histories on episode reset. Other environments retain their state.
- Preserve inherited training gate parameters and branch gains; do not tune them
  to the demonstration motion. External Motion-WM is distinct from DTERA's
  internal trainable auxiliary world model.

The initial implementation retains per-environment NumPy corruption for runtime
parity and batches GRU inference on the training device. CPU/device transfers
make this a small-pilot implementation, not a validated 4096-environment setup.

## Pilot command (not yet run)

Run from the repository root in the local `twist` environment. Checkpoints must
already exist locally; they are not bundled with these source changes.

```bash
OMP_NUM_THREADS=1 /home/hank/anaconda3/envs/twist/bin/python legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_dtera \
  --proj_name g1_motion_wm_dtera --exptid pilot_mixed_100 \
  --resumeid ../g1_twist_dtera_revision4/dtera_revision4_frozen_output_bias_overnight \
  --checkpoint 4800 --num_envs 64 --max_iterations 100 \
  --seed 42 --rl_device cuda:0 --no_wandb --fix_action_std
```

This resumes the verified DTERA checkpoint into a separate experiment directory.
It uses the inherited motion dataset configuration; it does not establish a new
train/test split. Reserve fixed held-out motions before interpreting improvements.
The Motion-WM checkpoint defaults to
`legged_gym/logs/motion_world_model/full_stable_v2/best.pt`.

## Completed actual PPO smoke test

```bash
OMP_NUM_THREADS=1 /home/hank/anaconda3/envs/twist/bin/python tools/smoke_motion_wm_training.py \
  --task g1_motion_wm_dtera \
  --proj_name g1_motion_wm_dtera --exptid smoke_wm_active_v2 \
  --resumeid ../g1_twist_dtera_revision4/dtera_revision4_frozen_output_bias_overnight \
  --checkpoint 4800 --num_envs 2 --max_iterations 2 \
  --motion_file "$PWD/track_dataset/twist_motion_dataset/accad/B3___walk1.pkl" \
  --seed 42 --rl_device cuda:0 --no_wandb --fix_action_std
```

Use a new `--exptid` when rerunning: the smoke tool refuses to overwrite a run.
Only this smoke test forces WM-only episodes, to exercise inference after warmup.
It completed two PPO updates / 96 environment steps with 50 restored frames:

- 3695-D finite policy observations;
- external Motion-WM and original TWIST weights unchanged bit-for-bit;
- DTERA adapter weights changed;
- no gradients on external Motion-WM;
- clean reference/reward-buffer nonmutation check passed.

Original proof:
`legged_gym/logs/g1_motion_wm_dtera/smoke_wm_active_v2/smoke_proof.json`.
A portable copy is in `motion_wm_dtera_outputs/training_smoke/smoke_proof.json`.
The short run verifies integration, not convergence or superiority over TWIST.

## Next validation

After the small mixed-input pilot, export its DTERA checkpoint and rerun the
existing paired A–E evaluation and branch/gate ablations on fixed motions and
corruption seeds. Report full and post-warmup metrics, including regressions.
Do not select the best demonstration or infer improvement from PPO loss alone.
Dynamics-disturbance curricula, new gate tuning and joint WM training are not
implemented in this stage.
