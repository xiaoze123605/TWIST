# DTERA model_4800 消融评估

## 结论

训练链路和双支路结构工作正常，但 `model_4800.pt` 没有证明相对冻结 base policy 的稳定净收益。残差不是“太保守”，而是不同 seed 下收益方向不一致；直接增大 `0.03` scale 很可能放大负面修正。

## 评估设置

- checkpoint: `model_4800.pt`
- checkpoint MD5: `c68f0ff5335c62a957962c8381be5589`
- task: `g1_stu_anyadapter_dtera`
- 512 environments, 2000 control steps, headless GPU PhysX
- controlled ablation: seed 42, no domain randomization, 7 modes
- robustness ablation: seeds 42/43/44, domain randomization, 3 key modes
- motion curriculum and observation noise disabled by the evaluator

## 七组 no-DR，seed 42

| Mode | Return | vs base | Fall rate | Root error (m) | Joint vel error | Applied residual L2 |
|---|---:|---:|---:|---:|---:|---:|
| base_only | 33.9913 | 0.00% | 15.55% | 1.2458 | 0.7334 | 0.0000 |
| dyn_only | 34.3759 | +1.13% | 16.53% | 1.2708 | 0.7210 | 0.0176 |
| err_only | 34.5546 | +1.66% | 13.61% | 1.1502 | 0.6982 | 0.0372 |
| full_gate_off | 32.4888 | -4.42% | 14.56% | 1.3498 | 0.7371 | 0.1162 |
| full_demand_only | 34.4650 | +1.39% | 16.41% | 1.2769 | 0.7306 | 0.0449 |
| full_demand_confidence | 33.5819 | -1.20% | 16.36% | 1.3111 | 0.7317 | 0.0443 |
| full_gate | 33.7715 | -0.65% | 16.14% | 1.2537 | 0.7197 | 0.0436 |

单 seed 中 `err_only` 最好，但该结果没有在后续 seed 中复现。`full_gate_off` 明显变差，说明 gate 有必要；confidence 和 full risk gate 也没有带来提升。

## DR 三 seed 稳健性结果

| Mode | Mean return | Paired delta | Fall rate | Root error (m) | Root error delta |
|---|---:|---:|---:|---:|---:|
| base_only | 34.4015 | 0.00% | 15.66% | 1.1969 | 0.00% |
| err_only | 33.9229 | -1.35% | 15.96% | 1.2665 | +5.81% |
| full_demand_only | 34.1579 | -0.65% | 15.19% | 1.2761 | +6.62% |

Seed-level return delta:

| Seed | err_only vs base | full_demand_only vs base |
|---:|---:|---:|
| 42 | +2.17% | +3.14% |
| 43 | -3.12% | -4.21% |
| 44 | -3.11% | -0.88% |

Paired return delta 的 95% CI：

- `err_only`: -0.479 return，95% CI `[-3.079, 2.122]`
- `full_demand_only`: -0.244 return，95% CI `[-3.405, 2.918]`

两个区间都跨 0，不能宣称显著优于 base。

## 诊断

1. 残差确实足够大：demand-only 的 applied residual L2 约 0.045，gate-off 达到 0.116；不存在“残差太小所以看不出区别”。
2. 当前主要问题是修正方向不稳定，而非幅度不足。相同 checkpoint 在不同 motion/DR seed 下从正收益变成负收益。
3. Dynamics branch 没有形成一致增益。full 相比 err-only 没有稳定改善 tracking，部分设置下反而增加 root/keybody error。
4. Demand gate 能限制损害，但没有识别出“残差真正有益”的时刻。confidence/risk gate 当前也未通过消融验收。

## 建议

- 不建议按当前配置继续训练到 30000 轮。
- 不建议增大两个支路的 `0.03` residual scale。
- 下一步应做固定 motion ID/固定起始帧的配对评估，定位哪些动作、哪些时刻残差改善或破坏 tracking。
- 在得到条件化收益证据前，不应把 full DTERA 作为优于原始 TWIST 的最终模型。
