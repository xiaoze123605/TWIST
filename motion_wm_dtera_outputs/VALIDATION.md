# Motion-WM + DTERA verification

Branch: `feature/motion-wm-dtera`, based on `twist-wm/wm-demo-results`
(`origin` in this checkout points to the older TWIST repository).
Implementation commit: `377f757`.

No training or per-motion gate/gain tuning was performed.
DTERA: revision4 frozen-output-bias `model_4800.pt`, SHA-256
`caf4c74e0ec3e86b411c9ac954c9c0e8bb919369fcd5fd9dbbb232037f4ec1cd`.
Motion-WM: `full_stable_v2/best.pt`, SHA-256
`187c11b128cbf1a6c095edcf1e3e8291fef76b63c03639f1d5618285f83afd99`.
Original TWIST: `0529_twist_rlbcstu-36500-jit.pt`, SHA-256
`f7d34dd2bc0b278cd4da33811dc8218e65da306a83d5c62f6fd12c1dc8202ef0`.

## Checks completed

- 49 distinct targeted unit tests passed: 31 DTERA, 14 Motion-WM,
  3 deployment runtime, 1 combined warmup/reset integration regression.
- Six exports from one checkpoint: training demand-only, gate-off,
  demand+confidence, full gate, dynamics-only, tracking-only.
- Strict inference-state loading, JIT/eager parity, residual bounds, live
  branches and rejection of non-3695 probe dimensions passed for every export.
- Every DTERA rollout must log `[DTERA] Detected 3695-D dual-history policy`.
- Each recorded tracking-history last frame uses processed joint reference;
  two histories are checked at every step for 20x74 / 20x53 shape and shifts.
- Policy actions are reconstructed from JIT diagnostics and checked per frame.
- Each run starts new high/low processes; isolated Redis is created per batch,
  frame/reference keys are cleared between groups, and simulator/history state
  resets before inference. No unrelated Redis instance is cleared.
- Single-motion A/B and C/D first 24 frames have bit-identical references,
  robot states and actions. All seven single-motion WM runs have identical
  processed reference sequences over all 380 frames.
- All ten single-motion videos passed ffprobe: 640x480, 50 fps, 380 frames.
- Whitespace checks and Python compilation passed.
- All 55 runs completed: 10 single-motion runs and 45 multi-motion A–E runs,
  totaling 19,130 policy frames. All 55 initial MuJoCo qpos arrays are identical.
  Every scenario's A/B and C/D warmup states/actions match exactly, and B/D
  processed references match exactly over the complete sequence.

## Definitions and limitations

`single/SUMMARY.md` and `multi/SUMMARY.md` report full sequence and removal of
exactly the first 24 policy frames. Per-run JSON and CSV retain every scenario.
The D training configuration is demand-only, gains=1, delta scales=0.03.
The `full` gate ablation is separately named `D_full`. Gate-off sets gates to
one and does not disable residuals. Demand+confidence and full explicitly use
confidence strength=1; other learned weights/gains/scales remain the same.

Max tilt is the angle of the robot vertical axis relative to world vertical,
`acos(cos(roll)*cos(pitch))`; legacy summaries retain their older max-component
definition. Fall rate is an episode-level threshold proxy (height<0.35m or
tilt>1rad). It can flag intended lying/sitting poses and does not replace
manual identification of accidental falls. This limitation matters for the
preselected unseen lying-to-crouch motion.

Inference timing includes policy/runtime forward, but excludes Redis,
diagnostic calls and JSON serialization. Reference-pipeline time is separate.
These are CPU measurements, not a complete deployment latency benchmark.

The repository stores all summaries, command files, process logs, export logs,
and the ten single-motion MP4 videos. The 55 raw `frames.jsonl` files remain in
the local experiment directories as the audit source because they total about
907 MiB; they are reproducible from each run's `commands.json` and are not
committed to Git. DTERA and Motion-WM checkpoint binaries are likewise
identified by SHA-256 rather than duplicated in the repository.

Multi-motion selection is fixed before rollout: lexicographic files, first
eligible file in each of three source collections, duration 4–10 seconds,
excluding groups in configured DTERA twist_dataset.yaml and Motion-WM train/val
manifests. No pose/performance filter is used. This does not establish exclusion
from the frozen original TWIST pretraining corpus or unidentified source aliases.
The three seeds are 42, 43 and 44. Multi-motion validation repeats A–E;
the six DTERA ablation configurations are evaluated on the first single motion.

Single-motion D improves joint RMSE by 7.397% relative to C and 0.502% relative
to B after warmup, but worsens pitch/tilt and root velocity relative to B.
Tracking-only is slightly better than D in joint RMSE on this motion; no gain
change or branch replacement follows from that observation. Branch cosine is
0.397 on average and 6.2% of frames have opposing corrections, so persistent
global cancellation is not supported; pose over-correction remains a concern.

## Multi-motion outcome

All three fixed motions and seeds 42/43/44 are included. After warmup, mean
joint RMSE is A=0.278386, B=0.273090, C=0.280498, D=0.274109, E=0.275416.
Motion-WM improves the DTERA controller (D vs C) by 2.278% in aggregate joint
RMSE, with improvement in 8/9 pairs. Compared with Motion-WM + original TWIST
(D vs B), joint RMSE worsens by 0.373%, with improvement in only 2/9 pairs.
All five groups have a 33.3% fall-proxy rate, driven by the lying-to-crouch
motion. D vs B improves mean root-velocity RMSE slightly (0.299064 vs 0.299869)
and mean episode max tilt (0.935156 vs 0.940756), so the result is mixed rather
than a uniform degradation or improvement.

The Motion-WM + DTERA architecture is operational and remains the integrated
design. These frozen-checkpoint experiments do not establish a stable net
advantage over Motion-WM + original TWIST. No tuning or retraining followed.
