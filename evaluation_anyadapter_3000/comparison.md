# AnyAdapter OpenTrack model_3000 vs frozen TWIST base

Evaluation setup: synchronized high-level/low-level MuJoCo rollout, `accad/B3___walk1.pkl`, seed 42, 380 policy frames, no action EMA. Errors are measured against the clean motion target.

| Reference | Policy | Joint RMSE | Root error | Yaw RMSE | Tilt RMSE | Root velocity RMSE | Minimum pelvis height |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | frozen TWIST | 0.143694 | 0.295971 | 0.584098 | 0.065222 | 0.175784 | 0.741405 |
| raw | AnyAdapter-3000 | 0.205699 | 0.072326 | 0.123891 | 0.050329 | 0.187177 | 0.749891 |
| Motion-WM | frozen TWIST | 0.140757 | 0.272574 | 0.535382 | 0.069984 | 0.174740 | 0.740424 |
| Motion-WM | AnyAdapter-3000 | 0.207481 | 0.083434 | 0.149123 | 0.050400 | 0.182400 | 0.749519 |

## Relative change versus base

Lower is better for all error columns.

| Reference | Joint RMSE | Root error | Yaw RMSE | Tilt RMSE | Root velocity RMSE |
|---|---:|---:|---:|---:|---:|
| raw | +43.15% | -75.56% | -78.79% | -22.83% | +6.48% |
| Motion-WM | +47.40% | -69.39% | -72.15% | -27.98% | +4.38% |

## Conclusion

The checkpoint is not uniformly better than the base policy. It strongly improves global root/yaw/tilt tracking and slightly improves minimum pelvis height, but degrades joint-position and root-velocity tracking. The same tradeoff remains after the 79-frame history warm-up, so it is not explained by initialization alone.

The likely causes are unrestricted adapter drift (`adapter_reg_coef=0`, no action-delta scale or gate), a dynamics-only history without explicit joint tracking-error input, PPO reward tradeoffs favoring global stability/root objectives, and sim-to-sim sensitivity of the learned residual. Motion-WM also does not improve this already-clean reference: its processed reference differs only modestly from raw (mean absolute difference 0.0151), and the adapter's WM result is slightly worse than its raw result on joint/root errors.
