# Frozen TWIST + Motion-WM + AnyAdapter stability pilot (2026-09-25)

## Decision

**Do not launch the 30,000-update long run yet.** The audited 4096-environment pilots run to 100 updates and resume correctly, but fixed MuJoCo evaluation does not show sustained improvement over raw150. The 1e-6 actor rate improves the normalized 8-clip score by 0.37% at update 50 and 0.14% at 100. The 2e-6 controlled comparison improves by 0.23% at 50 and regresses by 0.64% at 100. Neither meets the prespecified 1% screening threshold, so the 12-clip and 54-second gates were not promoted to, and a long run is not justified.

## Implementation

- Task `g1_motion_wm_anyadapter_stable`: raw references, fixed motion sampling, frozen TWIST base / dynamics encoder / action WM, fixed action standard deviation, critic warmup for 20 updates, actor/critic Adam rates 1e-6 / 1e-4, PPO clipping 0.1, 3 epochs, KL rollback at 0.012, and a fixed 16,384-sample source-policy anchor bank.
- Task `g1_motion_wm_anyadapter_stable_fast`: same configuration, actor rate 2e-6 only.
- Checkpoints include the optimizer groups, replay bank, algorithm counter, and Python/NumPy/Torch/CUDA RNG state. Fresh runs must load the audited raw150 checkpoint; continuation must use a same-task checkpoint and an absolute update target.
- `tools/select_stable_motionwm.py` codifies the paired 12-clip and complete-motion promotion gate. Both tasks leave W&B online by default.

## Fixed evaluation (seed 42, `--reference-mode raw`)

The same eight clips were evaluated for 280 frames on the first clip and 400 frames on each other clip. All values below are means; lower is better. The normalized score is 0.4 joint + 0.3 root velocity + 0.3 yaw, with each term divided by frozen TWIST on the same clips.

| Checkpoint | Joint RMSE | Root velocity RMSE | Yaw error | Score versus raw150 | New falls |
|---|---:|---:|---:|---:|---:|
| raw150 source | 0.133694 | 0.161110 | 0.591097 | reference | 0 |
| stable 1e-6, 50 | 0.133779 | 0.160658 | 0.584585 | 0.37% better | 0 |
| stable 1e-6, 100 | 0.133577 | 0.160300 | 0.592273 | 0.14% better | 0 |
| stable 2e-6, 50 | 0.133717 | 0.160205 | 0.589794 | 0.23% better | 0 |
| stable 2e-6, 100 | 0.133839 | 0.160613 | 0.605502 | 0.64% worse | 0 |

The first fast50 evaluation accidentally used half the established durations. It is excluded. `eval_fast50_8clip_correct/results.json` is the valid replacement; every valid run was checked against the original frame counts. These clips are a screen, not an independent test set. Seed 42 in the fixed MuJoCo harness does not constitute independent-seed validation.

## Training and resume observations

Both corrected pilots completed 100 updates with 4096 environments, fixed action std 0.04907, and online W&B. At update 100, mean per-minibatch policy KL is 0.000741 for 1e-6 and 0.001020 for 2e-6, with no rejected updates. Mean training reward at update 100 is 49.33 for 1e-6 and 43.91 for 2e-6. Training reward and fixed evaluation disagree, so it cannot select the long-run policy. Peak Torch allocation is 10.44 GB and reservation 12.17 GB in both runs. Resume from stable update 100 to absolute 102 completed; `model_102.pt` contains counter 102, a full 16,384-sample bank, and two optimizer groups. Simulator trajectories restart on resume, so continuation is not bit-identical.

- 1e-6 online W&B: https://wandb.ai/3216369125-zhejiang-university-of-technology/g1_mimic/runs/r28ffoof
- 2e-6 online W&B: https://wandb.ai/3216369125-zhejiang-university-of-technology/g1_mimic/runs/kysakz7q
- Resume smoke W&B: https://wandb.ai/3216369125-zhejiang-university-of-technology/g1_mimic/runs/edk1u7l0

The next meaningful experiment should target the train/evaluation mismatch rather than extend the present settings. Keep raw150 as the incumbent and require a candidate to clear the paired gate at successive checkpoints before a 500-update endurance pilot, followed by broader held-out motion and independent-rollout validation before 30,000 updates.
