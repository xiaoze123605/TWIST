# Frozen TWIST + dynamics WM + AnyAdapter

**2026-09-24 update:** The guarded continuation reached update 14144 and
regressed on the paired MuJoCo screen. Do not resume that run. Its update-500
checkpoint is the best screened checkpoint within that long run, but the
guarded update-200 checkpoint still wins the eight-clip aggregate. The older
Motion-WM + AnyAdapter run used a training list containing every clip in that
eight-clip screen, so its scores on those clips cannot establish held-out
generalization. The clean Motion-WM continuation task and audit are described
in `docs/motion_wm_anyadapter_clean.md`.

The `g1_twist_baseline_adapter` task preserves the TWIST student observation,
reward, reset, curriculum and PD control. Its frozen 0529 JIT supplies the
tracking policy. A layerwise AnyAdapter uses a 79-frame dynamics history; a
separate action world model trains the history encoder. Training uses the full
prepared training split. The four clips below are held out from training.

## What the training and evaluation data show

The mean motion difficulty changed from 8.90 at updates 100–299 to 7.84 at
2500–2699, while mean episode reward changed from 49.77 to 41.28. A falling
training reward therefore does not by itself show a worse policy. WM loss
fell from 0.1255 to 0.0727, but mean absolute raw adapter correction rose
from 0.0261 to 0.0643 and its maximum rose from 0.288 to 0.664. Adapter
weight norms grew throughout this period as its L2 penalty annealed from 4
to 2. This is evidence of policy drift, although it does not prove drift is
the only cause of the held-out regression.

Paired MuJoCo screening used four fixed validation clips, the same seed,
raw references and an isolated Redis server. Lower errors are better.

| Policy | Joint RMSE | Root error | Yaw error | Root velocity RMSE | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen TWIST | 0.12728 | 0.28879 | 0.54247 | 0.13737 | 0 |
| Original update 1600, gain 0.25 | 0.12733 | 0.26140 | 0.48693 | 0.13826 | 0 |
| Original update 2400, gain 0.25 | 0.12717 | 0.26763 | 0.50135 | 0.14029 | 0 |
| Refinement update 50 | 0.12807 | 0.28194 | 0.52571 | 0.13840 | 0 |
| Refinement update 100 | 0.13524 | 0.28297 | 0.49230 | 0.14889 | 1 |

The original 1600 checkpoint with gain 0.25 was the best screened checkpoint
before the guarded experiment below. An earlier 4096-environment,
100-update refinement started from it, reset the
optimizers, used gain 0.25 during PPO, held adapter L2 at 4 and penalized
raw corrections above 0.12. Mean absolute correction stayed near 0.016,
but the checkpoint at 100 fell on `mydata2/3_seg00.pkl`. Neither its updated
adapter alone nor its updated encoder alone fell on that clip; their combined
change did. This points to interaction between the two learned branches.
That version did not justify continuing to 30000 updates.

Zeroing the 1600 checkpoint's history latent worsened root and yaw error
on the same clips. The history encoder contributes useful information, so
removing it outright is not supported by this test.

## Guarded continuation and current result

The new `g1_twist_baseline_adapter_refine` configuration freezes the source
world model and history encoder, and anchors the new policy's action mean
to the original update-1600 actor on each PPO mini-batch. The anchor uses
gain 0.25, coefficient 10, policy learning rate 2e-6, adapter L2 coefficient
4 and the existing correction-tail penalty. The anchor loss and mean
absolute action difference are logged to W&B. W&B remains online and stores
its local files in each experiment directory.

The 4096-environment pilot ran from refinement update 0 through 200, with
separate online W&B runs for updates 0–100 and 100–200. The 50 and 100
checkpoints did not beat the source policy on the original four clips.
Update 200 did, so it was tested on four more motions from `val.yaml`.
Across all eight clips, the paired MuJoCo results are:

| Policy | Joint RMSE | Root error | Yaw error | Root velocity RMSE | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen TWIST | 0.13474 | 0.33977 | 0.65079 | 0.16214 | 0 |
| Original update 1600, gain 0.25 | 0.13396 | 0.32463 | 0.62358 | 0.16012 | 0 |
| Guarded refinement update 200 | 0.13471 | 0.29542 | 0.56732 | 0.15964 | 0 |

Update 200 reduces root and yaw error about 9% versus update 1600, with
about 0.6% higher joint RMSE, slightly lower root velocity RMSE and no
additional falls. Gains vary by clip; in particular, one easy clip worsens.
The eight-clip screening selector chose `guarded200`. The result supports
a staged continuation, not a claim of general improvement across the
entire validation split. Results are stored under
`reports/twist_baseline_adapter_guarded_20260923`.

## Code changes

`PPOTwistBaselineAdapter` fixes the WM target's one-step alignment: it uses
the action actually sent to the environment from the next observation's
history and records the pre-action state in autoregressive history. This
matters when commands are delayed or clipped. The refinement task also
supports a fixed adapter penalty, a squared tail penalty and the source
policy anchor. The exporter reads the
training gain stored in new checkpoints, with an explicit override for
older checkpoints. `tools/select_twist_adapter_checkpoint.py` chooses among
complete paired results, rejecting extra falls and material joint or
velocity regressions.

## Reproduce the screening result

Run the contract and causal checks before training or export:

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python \
tools/check_twist_baseline_adapter.py
```

The old failed pilot and its branch ablation remain under
`reports/twist_baseline_adapter_refinement_20260923`. The selected guarded
pilot checkpoint is in
`legged_gym/logs/g1_twist_baseline_adapter/anchored_guarded_1600_pilot_v1_200_20260923`.
The shorter preliminary run directory was removed; its online W&B run IDs
are `oqibv1m1` and `otc1sqku` in the `g1_mimic` project.

## Long-training instructions (W&B online)

Run from the repository root. The validated starting point is guarded
refinement `model_200.pt`. First extend to 1000 **refinement updates** with
4096 environments. This is already a continuation of the selected policy;
the command does not use `--no_wandb`, and `train.py` uses online W&B by
default. Use a new experiment ID if this one already exists.

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python -u \
legged_gym/legged_gym/scripts/train.py \
  --task g1_twist_baseline_adapter_refine \
  --proj_name g1_twist_baseline_adapter \
  --exptid anchored_guarded_long_1000_20260923 \
  --resumeid anchored_guarded_1600_pilot_v1_200_20260923 \
  --checkpoint 200 --num_envs 4096 --max_iterations 1000 --seed 42
```

Screen the saved update-1000 checkpoint against frozen TWIST, the original
1600 checkpoint and guarded update 200. The script exits nonzero unless the
new checkpoint passes the safety gates and improves the incumbent's
aggregate score by at least 0.5%. Inspect `selection.json` and per-clip
metrics before continuing.

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python \
tools/screen_twist_guarded_checkpoint.py \
  --candidate legged_gym/logs/g1_twist_baseline_adapter/anchored_guarded_long_1000_20260923/model_1000.pt \
  --output reports/twist_baseline_adapter_guarded_20260923/screen_1000
```

Only if `candidate` wins that screen and no clip falls, continue with the
following command. The original 1600 updates plus a maximum of 28400
refinement updates equals the requested 30000-update cap. At 4096
environments and 24 steps per update, the cap corresponds to
2,949,120,000 environment transitions. `--max_iterations` is absolute
within the refinement run, so resuming at update 1000 trains 27400 more
updates. W&B remains online. Screen later saved checkpoints, retain the
best validated checkpoint and stop the run if performance regresses or
falls; pass `--incumbent path/to/best/model_N.pt` to the screening script
when checking later checkpoints. Update count alone is not a selection
criterion.

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python -u \
legged_gym/legged_gym/scripts/train.py \
  --task g1_twist_baseline_adapter_refine \
  --proj_name g1_twist_baseline_adapter \
  --exptid anchored_guarded_long_30000_20260923 \
  --resumeid anchored_guarded_long_1000_20260923 \
  --checkpoint 1000 --num_envs 4096 --max_iterations 28400 --seed 42
```

Eight clips and one seed are a screening result. Validate a broader motion
set and additional seeds before claiming generalization beyond this pilot.
