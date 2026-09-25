# Stable Motion-WM AnyAdapter pilot

The `g1_motion_wm_anyadapter_stable` task starts from the audited baseline-aligned
raw-reference update 150. It keeps the TWIST base, dynamics encoder, and action
WM frozen. The reference mode is raw. The motion curriculum is disabled so
training-motion sampling weights stay fixed; random starting times remain on.

This task has separate actor and critic Adam learning rates (1e-6 and 1e-4),
20 critic-only updates, PPO clip 0.1, three epochs, fixed action standard
deviations copied from the source checkpoint, and a 0.012 per-minibatch KL
rollback threshold. A fixed bank of up to 16384 training observations caches
base observations, frozen dynamics embeddings, and source-policy actions.
Twenty percent of the source-policy anchor penalty uses a random bank sample.
The bank, optimizer states, algorithm counter, and RNG state are checkpointed.
An environment reset still means resumed simulator trajectories are not
bit-for-bit identical to an uninterrupted run.

A separate `g1_motion_wm_anyadapter_stable_fast` task changes only the actor
learning rate to 2e-6 for a controlled comparison. Its checkpoints must
resume under that same task name.

The initial diagnostic must be capped at 100 updates with 4096 environments,
W&B online, and a new experiment ID. Use the exact raw150 checkpoint as
`--warm-start-checkpoint`; it must match the audited training split. The run
saves models at updates 25, 50, 75, and 100. Review `run_manifest.json` before
interpreting results: `fix_action_std=true`, `motion_curriculum=false`, and
`anchor_replay_size=16384` must all be present.

For a later continuation, use a new experiment ID and `--resumeid` pointing at
that task's source run, with `--checkpoint N` and an absolute
`--max_iterations` target. The same task and motion dataset are required. A
checkpoint includes both optimizer groups and the fixed anchor bank. Do not
use warm start for continuation, because it discards optimizer state.

Checkpoint promotion uses `tools/select_stable_motionwm.py`: a paired 12-clip
score of 0.4 joint + 0.3 root velocity + 0.3 yaw, each normalized to frozen
TWIST, must improve at least 1% over the incumbent raw150. Neither mean joint
nor velocity error may regress over 1%; no clip may regress over 5% for those
metrics; no new fall proxy is allowed. A complete long motion must remain
within 1% of the incumbent for joint/velocity and 3% for yaw. This is a
screening gate. Broader validation and fresh test motions are still needed
before declaring a policy better than the original TWIST baseline.

The 2026-09-25 100-update pilots failed the fixed screen: the 1e-6 run
improved the 8-clip normalized score by only 0.14%, while the 2e-6 run
regressed by 0.64%. Do not run 30000 updates until successive pilot
checkpoints pass the fixed screen and a later 500-update pilot continues
producing accepted policies. See `reports/stable_motionwm_20260925/README.md`.
