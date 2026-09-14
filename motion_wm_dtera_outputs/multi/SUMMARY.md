# Motion-WM + DTERA fixed paired evaluation

All scenarios are included. Aggregate values are equal-weight episode means.
A=corrupt+TWIST; B=WM+TWIST; C=corrupt+DTERA; D=WM+DTERA(training demand_only); E=clean+DTERA.
D is also the demand-only ablation. Gate-off means gates equal one, not residuals disabled.
D_full enables demand, confidence and risk; branch-only modes retain training demand-only gates.
Units: joint/roll/pitch/yaw/tilt rad; height m; root velocity m/s; action rate policy-action units/s.
Fall rate is the fraction of episodes crossing height<0.35m or tilt>1rad; it is a threshold proxy, not a contact classifier.
Inference time excludes Redis, JSON logging and diagnostic calls; reference pipeline includes corruption and Motion-WM.

## full_sequence

| Mode | Episodes | Joint RMSE | Velocity RMSE | Height RMSE | Roll | Pitch | Yaw | Mean episode max tilt | Fall rate | Action rate | Policy ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.282910 | 0.314765 | 0.119987 | 0.859872 | 0.483665 | 0.970741 | 0.958883 | 0.333333 | 38.030095 | 0.249713 |
| B | 9 | 0.278107 | 0.305450 | 0.119397 | 0.857575 | 0.480210 | 1.147894 | 0.940826 | 0.333333 | 38.089290 | 0.253609 |
| C | 9 | 0.284739 | 0.309467 | 0.120769 | 0.838248 | 0.486095 | 0.929806 | 0.949516 | 0.333333 | 35.406745 | 0.669640 |
| D | 9 | 0.279017 | 0.304336 | 0.120132 | 0.814059 | 0.468186 | 0.970649 | 0.936076 | 0.333333 | 35.938310 | 0.682261 |
| E | 9 | 0.279970 | 0.302559 | 0.118008 | 0.861093 | 0.484889 | 0.953164 | 0.929128 | 0.333333 | 29.543760 | 0.676819 |

## after_warmup

| Mode | Episodes | Joint RMSE | Velocity RMSE | Height RMSE | Roll | Pitch | Yaw | Mean episode max tilt | Fall rate | Action rate | Policy ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.278386 | 0.309571 | 0.103985 | 0.889201 | 0.486073 | 1.013371 | 0.958686 | 0.333333 | 37.487223 | 0.234126 |
| B | 9 | 0.273090 | 0.299869 | 0.103638 | 0.886562 | 0.482319 | 1.197222 | 0.940756 | 0.333333 | 37.560204 | 0.238402 |
| C | 9 | 0.280498 | 0.304794 | 0.104871 | 0.866283 | 0.487791 | 0.969068 | 0.948596 | 0.333333 | 34.601668 | 0.643198 |
| D | 9 | 0.274109 | 0.299064 | 0.104202 | 0.840089 | 0.467948 | 1.012484 | 0.935156 | 0.333333 | 35.179823 | 0.658271 |
| E | 9 | 0.275416 | 0.297296 | 0.102767 | 0.890101 | 0.485300 | 0.995313 | 0.926527 | 0.333333 | 28.691265 | 0.654856 |

## Evidence

- `manifest.json`: fixed motions/seeds/checkpoint hashes and selection rule.
- `all_metrics.csv`: every scenario and both windows, including negative outcomes.
- `aggregate.json`: aggregate metrics and individual paired joint improvements.
- `branch_diagnostics.json`: correction norms, opposing-frame fraction and gate distributions after warmup.
- Per-run `commands.json`, `high.log`, `low.log`, `frames.jsonl`, `summary.json`.
- `frames.jsonl` contains clean/corrupt/WM/processed references, robot state, both histories, corrections, gates and actions.
- “Unseen” is relative to configured DTERA motion groups and Motion-WM train/val groups; original frozen TWIST pretraining provenance is not available.
