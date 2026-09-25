# Motion-WM + AnyAdapter optimization, 2026-09-24 to 2026-09-25

## Decision

Do not start a 30,000-iteration run from the current recipe. The best screened
candidate is the baseline-aligned raw-reference pilot at update 150, but its
advantage is small and the subsequent conservative continuation did not improve
the fixed eight-clip screen through update 200. Keep the original guarded update
200 as the comparison checkpoint. These are simulation results, not evidence of
real-robot performance.

Selected candidate:

- Training checkpoint: `legged_gym/logs/g1_motion_wm_anyadapter_baseline_pilot/baseline_reward_raw_pilot200_20260924/model_150.pt`
- JIT: `reports/clean_motionwm_optimization_20260924/baseline_reward_raw_150.pt`
- Original guarded comparator JIT: `reports/clean_motionwm_optimization_20260924/guarded200-jit.pt`

## Method

The paired MuJoCo screen uses raw high-level references and seed 42. The first
eight clips include a 5.6-second standing clip and seven 8-second clips. Four
additional 8-second clips form the 12-clip screen. Full `mocap1/1.pkl` is 54
seconds (2,700 frames). Each policy is run from reset on each clip; the values
below are arithmetic means across clips, except for the single full motion.
`joint_rmse` is a joint tracking error, `root_velocity_rmse` is root velocity
error, and `yaw_error` is heading error. The aggregate `root_error` from the
simulator combines height and orientation terms and must not be read as root
position error.

| Policy | 12-clip joint | 12-clip velocity | 12-clip yaw | Full joint | Full velocity | Full yaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen TWIST base | 0.13537 | 0.14802 | 0.52802 | 0.14568 | 0.13108 | 1.02826 |
| Guarded update 200 | 0.13531 | 0.14761 | 0.49525 | 0.14672 | 0.13197 | 0.82493 |
| Baseline-aligned raw update 150 | **0.13492** | **0.14761** | **0.47861** | **0.14541** | **0.13008** | 0.83452 |

The raw update 150 improves the 12-clip joint and yaw means over the guarded
checkpoint; its full-motion yaw is slightly worse than guarded update 200 but
better than frozen TWIST. The selection result is in
`baseline_raw150_12clip_full54_selection.json`. No new fall proxy appeared in
these paired screens.

## Experiments and code decisions

1. Swept adapter gains on clean updates 1000 and 4450. Gain 0.25 was safer
   than the training gain 1.0 in the short screen, but the older update 4450
   was rejected by a per-clip joint regression. See `gain_sweep8/results.json`.
2. Ran 200-update mixed/trainable, mixed/frozen, and raw/trainable pilots from
   clean update 1000, with stronger base protection. None produced sustained
   improvement on the fixed eight clips. W&B runs: `ba72cjhb`, `ai3jdi5k`,
   `zc2oy2y9`.
3. Added an optional action activity gate to the JIT exporter for diagnosis.
   It helped some short clips, but full-motion heading error was worse, so no
   gated JIT is recommended. The full-motion check is now an optional gate in
   `tools/select_twist_adapter_checkpoint.py`.
4. Audited the clean environment against the guarded TWIST baseline. Reward
   weights, motion curriculum, and physics randomization differed. Added a
   baseline-aligned Motion-WM task and tested mixed-reference and raw-reference
   pilots from guarded update 200. Both froze the pretrained dynamics encoder
   and action WM, used adapter gain 0.25, adapter regularization 2, policy
   anchor 5, and policy LR 2e-6. W&B runs: mixed `ihs3hx84`, raw `4xfmevv2`.
   Mixed update 100 failed the full-motion yaw gate. Raw update 150 passed it.
   These pilots change several settings as a package; their results do not
   isolate one causal factor.
5. Continued raw update 150 for 200 updates with LR 1e-6 and anchor 10 to that
   checkpoint. Update 50 changed eight-clip joint/velocity/yaw from
   0.13369/0.16111/0.59110 to 0.13397/0.16210/0.58455. Update 100 gave
   0.13422/0.16156/0.59414, worse on all three. Update 200 gave
   0.13453/0.16114/0.60744, also worse on all three. W&B run: `oxdry0cm`.
   See `conservative_continue100_eval8/results.json` and
   `conservative_continue200_eval8/results.json`.

The high-level `--reference-mode raw` is the tested deployment mode. The
pretrained dynamics-history encoder still conditions the adapter policy, but
the action WM is frozen during the baseline-aligned pilots and is not embedded
in the exported low-level JIT. `--reference-mode wm` requires the high-level
server and is a separate validation path.

The fixed screen is small, simulation-only, and deterministic in this setup.
Repeating it with another `--seed` did not create a different reset/trajectory,
so this report does not claim independent seed replication. Before long
training, require a candidate to improve broader held-out motions and preserve
full-motion heading and stability, then use periodic fixed evaluation and early
stopping instead of trusting training reward alone.
