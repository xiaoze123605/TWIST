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
| A | 1 | 0.155253 | 0.188128 | 0.026908 | 0.041337 | 0.082695 | 0.676775 | 0.250125 | 0.000000 | 29.452168 | 0.218009 |
| B | 1 | 0.146377 | 0.178412 | 0.027296 | 0.043597 | 0.084742 | 0.565611 | 0.259391 | 0.000000 | 28.881670 | 0.247901 |
| C | 1 | 0.155305 | 0.189407 | 0.027272 | 0.041924 | 0.099212 | 0.615251 | 0.279594 | 0.000000 | 29.498633 | 0.694098 |
| D | 1 | 0.145498 | 0.180133 | 0.027537 | 0.045954 | 0.099599 | 0.548546 | 0.290316 | 0.000000 | 28.862647 | 0.682330 |
| E | 1 | 0.143453 | 0.175697 | 0.026745 | 0.041250 | 0.097344 | 0.571362 | 0.282191 | 0.000000 | 15.656923 | 0.665210 |
| D_gate_off | 1 | 0.150376 | 0.192889 | 0.027290 | 0.051401 | 0.130399 | 0.721634 | 0.325725 | 0.000000 | 30.100749 | 0.678244 |
| D_demand_confidence | 1 | 0.145677 | 0.179924 | 0.027524 | 0.045547 | 0.100546 | 0.536512 | 0.289189 | 0.000000 | 28.905789 | 0.769353 |
| D_full | 1 | 0.145631 | 0.180376 | 0.027385 | 0.045865 | 0.099751 | 0.543327 | 0.289147 | 0.000000 | 28.851006 | 0.770148 |
| D_dyn_only | 1 | 0.146368 | 0.178726 | 0.027373 | 0.044534 | 0.088700 | 0.537388 | 0.267046 | 0.000000 | 28.858167 | 0.581171 |
| D_err_only | 1 | 0.145447 | 0.180162 | 0.027344 | 0.044557 | 0.096199 | 0.564018 | 0.283741 | 0.000000 | 28.867873 | 0.663241 |

## after_warmup

| Mode | Episodes | Joint RMSE | Velocity RMSE | Height RMSE | Roll | Pitch | Yaw | Mean episode max tilt | Fall rate | Action rate | Policy ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.148330 | 0.171260 | 0.027008 | 0.033708 | 0.082388 | 0.698430 | 0.250125 | 0.000000 | 27.962609 | 0.205540 |
| B | 1 | 0.138362 | 0.159776 | 0.027420 | 0.036621 | 0.084579 | 0.583424 | 0.259391 | 0.000000 | 27.355461 | 0.236757 |
| C | 1 | 0.148664 | 0.172668 | 0.027387 | 0.034435 | 0.098801 | 0.634766 | 0.279594 | 0.000000 | 27.989163 | 0.672239 |
| D | 1 | 0.137667 | 0.161729 | 0.027669 | 0.039544 | 0.099216 | 0.565741 | 0.290316 | 0.000000 | 27.312405 | 0.662092 |
| E | 1 | 0.137161 | 0.157234 | 0.026799 | 0.033906 | 0.096794 | 0.589373 | 0.282191 | 0.000000 | 13.756148 | 0.645539 |
| D_gate_off | 1 | 0.143335 | 0.176644 | 0.027396 | 0.046042 | 0.131101 | 0.744803 | 0.325725 | 0.000000 | 28.624219 | 0.653886 |
| D_demand_confidence | 1 | 0.137864 | 0.161491 | 0.027657 | 0.039042 | 0.100250 | 0.553285 | 0.289189 | 0.000000 | 27.357900 | 0.746631 |
| D_full | 1 | 0.137812 | 0.162029 | 0.027509 | 0.039439 | 0.099402 | 0.560339 | 0.289147 | 0.000000 | 27.299555 | 0.741441 |
| D_dyn_only | 1 | 0.138398 | 0.160145 | 0.027502 | 0.037627 | 0.088540 | 0.554192 | 0.267046 | 0.000000 | 27.317161 | 0.566741 |
| D_err_only | 1 | 0.137589 | 0.161677 | 0.027465 | 0.037995 | 0.095846 | 0.581746 | 0.283741 | 0.000000 | 27.321781 | 0.644474 |

## Evidence

- `manifest.json`: fixed motions/seeds/checkpoint hashes and selection rule.
- `all_metrics.csv`: every scenario and both windows, including negative outcomes.
- `aggregate.json`: aggregate metrics and individual paired joint improvements.
- `branch_diagnostics.json`: correction norms, opposing-frame fraction and gate distributions after warmup.
- Per-run `commands.json`, `high.log`, `low.log`, `frames.jsonl`, `summary.json`.
- `frames.jsonl` contains clean/corrupt/WM/processed references, robot state, both histories, corrections, gates and actions.
- “Unseen” is relative to configured DTERA motion groups and Motion-WM train/val groups; original frozen TWIST pretraining provenance is not available.
