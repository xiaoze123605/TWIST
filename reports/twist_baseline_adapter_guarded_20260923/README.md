# Guarded TWIST adapter continuation, 2026-09-23

The actor started from `anchored_opt_v2_long/model_1600.pt` at gain 0.25.
The refinement froze the source history encoder and world model, optimized
the adapter with an action-mean anchor to that source policy, and kept W&B
online. Its 4096-environment checkpoints were screened with seed 42 in
paired MuJoCo runs using raw references.

| Eight validation clips | Joint RMSE | Root error | Yaw error | Root velocity RMSE | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen TWIST | 0.13474 | 0.33977 | 0.65079 | 0.16214 | 0 |
| Original update 1600, gain 0.25 | 0.13396 | 0.32463 | 0.62358 | 0.16012 | 0 |
| Guarded refinement update 200 | 0.13471 | 0.29542 | 0.56732 | 0.15964 | 0 |

`pilot_4clip_results.json` includes refinement checkpoints 50, 100, 150
and 200. Update 200 was selected. `pilot_8clip_results.json` contains the
expanded screen; its selection also chose update 200. The eight clips are
a screening set, not the complete validation split. The win is driven
mostly by dynamic motions; easy motions are not uniformly better.

The original source checkpoint SHA-256 is
`82e39cd1b72e2a230ef557d3f9462d70eb68eeed450eb8e845b148ae14efa4d9`.
The guarded update-200 checkpoint SHA-256 is
`e9acae2f4500aaedbfd77d7a95641c70323bbd11f9144fa7262a0956124f7f9b`.
The local files are in `legged_gym/logs/g1_twist_baseline_adapter`.

Online W&B runs: [updates 0–100](https://wandb.ai/3216369125-zhejiang-university-of-technology/g1_mimic/runs/oqibv1m1)
and [updates 100–200](https://wandb.ai/3216369125-zhejiang-university-of-technology/g1_mimic/runs/otc1sqku).
Long training starts from the latter run's `model_200.pt`. See
`docs/twist_baseline_adapter.md` for the staged commands and screening gate.
