# Frozen TWIST + dynamics WM + AnyAdapter

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

The 1600 checkpoint with gain 0.25 remains the best screened checkpoint.
The 4096-environment, 100-update refinement started from it, reset the
optimizers, used gain 0.25 during PPO, held adapter L2 at 4 and penalized
raw corrections above 0.12. Mean absolute correction stayed near 0.016,
but the checkpoint at 100 fell on `mydata2/3_seg00.pkl`. Neither its updated
adapter alone nor its updated encoder alone fell on that clip; their combined
change did. This points to interaction between the two learned branches.
It does not justify continuing this refinement to 30000 updates.

Zeroing the 1600 checkpoint's history latent worsened root and yaw error
on the same clips. The history encoder contributes useful information, so
removing it outright is not supported by this test.

## Code changes

`PPOTwistBaselineAdapter` fixes the WM target's one-step alignment: it uses
the action actually sent to the environment from the next observation's
history and records the pre-action state in autoregressive history. This
matters when commands are delayed or clipped. The refinement task also
supports a fixed adapter penalty and a squared tail penalty; these are
experimental controls, not a validated improvement. The exporter reads the
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

The completed pilot is in
`legged_gym/logs/g1_twist_baseline_adapter/anchored_refine_1600_v1_20260923`.
Its paired results, branch ablation and selection are in
`reports/twist_baseline_adapter_refinement_20260923`. To select from a new
paired evaluation:

```bash
python tools/select_twist_adapter_checkpoint.py \
  /path/to/paired_screen/results.json \
  --output /path/to/paired_screen/selection.json
```

The following command reproduces the failed 100-update refinement; it is
provided for diagnosis, not as a recommendation to repeat training. Use a
fresh `--exptid` if reproducing it. `--max_iterations` counts updates in
the new experiment. At 4096 environments and 24 steps per update, 30000
updates would be 2,949,120,000 environment transitions; counting the
original 1600 updates toward a total cap of 30000 leaves at most 28400 new
updates. A revised method should first beat the 1600 checkpoint on the
held-out screen at successive saved updates. Stop if a later update
regresses or falls.

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python -u \
legged_gym/legged_gym/scripts/train.py \
  --task g1_twist_baseline_adapter_refine \
  --proj_name g1_twist_baseline_adapter \
  --exptid YOUR_NEW_PILOT_ID \
  --warm-start-checkpoint \
  legged_gym/logs/g1_twist_baseline_adapter/anchored_opt_v2_long/model_1600.pt \
  --num_envs 4096 --max_iterations 100 --seed 42 --no_wandb
```

This four-clip result is a screen, not proof of overall generalization. A
broader validation split and more seeds are needed before claiming a gain
over frozen TWIST.
