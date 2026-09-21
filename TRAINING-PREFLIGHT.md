# WM + DTERA 训练前审计（2026-09-16）

结论：已排除一批真实的数据、时序、导出和评估问题，并通过完整训练清单的短 GPU 集成测试。可以开始受监控的训练实验，但不能保证训练后的策略优于 base，更不能据此直接实机部署。本次没有启动长训练，没有调整门控/增益/奖励权重，没有更新已有 checkpoint。

## 已修复的问题

| 问题 | 影响 | 修改 |
| --- | --- | --- |
| editable 安装指向旧 `/home/hank/TWIST/pose` | 本地修复可能根本未生效，训练和部署不一致 | 训练入口、smoke、导出器、高层服务优先加载当前 checkout |
| RSI 已随机初始化动作相位，runner 又随机 episode counter | 机器人姿态与 reference 突然错位 | WM 任务禁用第二次随机，runner 对错误调用直接报错 |
| RSI 可能从动作最后不足 24 帧的位置开始 | episode 在 WM warmup 完成前结束，WM 模式实际一直使用 corrupt reference | WM+DTERA 的 RSI 至少保留 26 个控制帧；原始 TWIST 任务不变 |
| 动作末尾未扣除 RSI 起始偏移 | reference 回绕产生不连续训练目标 | WM 任务按实际 motion 时间提前结束，不将真实失败误记为正常动作结束 |
| root quaternion 非单位化 | 离线 WM 与训练/高层参考预处理不同 | 统一归一化及符号连续处理，不改 PKL |
| 加载异常被静默跳过 | 实际训练集合与宣称集合不同 | 默认严格校验结构、有限值、DOF/body 顺序；报出失败文件 |
| 全集直接训练、跨 split 的相同文件副本 | 验证结果泄漏，浪费训练/评估 | 复用冻结 WM 的 group split，隔离跨 split 字节相同数据涉及的整个 group |
| gate/gain 是普通属性，state_dict 不保证导出一致 | 权重正确但部署行为不一致 | checkpoint 保存配置，导出核对配置，显式消融覆盖写入 JIT sidecar |
| 评估 resume 可混入旧模型结果 | 错误得出提升/退化结论 | 续跑前核对完整 manifest，禁用只重启低层的部分重试，必须显式提供 checkpoint |
| 实验目录可重用 | 覆盖或混合训练证据 | 新 WM 训练要求新 exptid，记录输入 SHA、环境/算法/策略配置 |

前一次提交已将最大的 local_body_pos 在 CPU 拼接后上传 GPU；这次确认真实训练实际使用该修复。不能用 allocator 环境变量代替释放其他进程占用或真实容量测试。

## 数据检查结果

原始 YAML：15,303 条目，15,302 个唯一文件。合并重复条目时累加权重，保留原采样质量；冻结 WM 原有的 8 个过短动作排除记录保留。

- 训练：12,248 motions；验证：1,519；测试：1,455。
- 发现 36 对跨集合字节相同文件；隔离涉及的 72 个 motion（按 group 隔离）。
- 282 个文件存在大于 0.05 的 quaternion norm 偏差；统一加载时修正，原始文件不变。
- 所有唯一文件完成结构/有限值检查；最终清单零结构错误。
- [清单及逐文件哈希审计](legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/audit.json)。同目录含 train.yaml、val.yaml、test.yaml，并指向当前 TWIST-WM checkout 的 motion 数据。旧 `/home/hank/TWIST` 数据副本与当前副本的 15,294 个审计文件已全量核对，零缺失、零 SHA-256 差异。

限制：没有穷举识别语义近重复动作；原始冻结 TWIST 的预训练数据来源不完整，因此这里只能声明对新 DTERA/已知 WM split 隔离，不能声明对 TWIST 完全未见。此前反复调试的 `mydata3/24_seg00.pkl` 只作工程诊断，不能作为最终泛化证据。因四元数预处理改变，正式对比必须用当前代码重跑所有 baseline。

重建命令（输出目录必须是新目录）：

```bash
/home/hank/anaconda3/envs/twist/bin/python tools/prepare_motion_wm_training.py \
  --data-root '/home/hank/TWIST（anyadapter）/track_dataset/twist_motion_dataset' \
  --out legged_gym/motion_data_configs/wm_dtera_prepared_new
```

## 实际验证

- Motion 系列单元测试 28 项通过；DTERA 系列 31 项通过。
- 2 环境、单 motion、3 次真实 PPO 更新：3695-D；WM 和 base 权重不变；两支路及两编码器均更新；观察和训练指标有限；跟踪历史确实使用 processed reference。
- 该 smoke 中 WM 完成 98 个 environment-frame 修复；证据：[smoke_proof.json](legged_gym/logs/g1_motion_wm_dtera_preflight/audit_20260916_single/smoke_proof.json)。这些更新不代表性能提升。
- 增加 RSI 尾部约束后再次完成 2 环境、3 次更新 smoke：98 个修复帧，冻结/梯度/历史断言继续通过。物理失败仍可在 warmup 前结束；这类失败不会被人为延长。
- 新 smoke checkpoint 导出 JIT：eager/traced 最大差异 0；训练配置匹配，warmup factor 0.003 保留。
- 最终组合预检使用当前 checkout 数据根、完整 12,248-motion train split、RSI 0.52 s 尾部约束和 64 环境，完成 3 次更新。三个 iteration 的 collection+learning 分别约 3.21/3.15/3.10 秒（不含初始化）；不是长期吞吐承诺。clean/corrupt/wm 三种模式每轮均获得样本，首轮 risk predictor 获得 10 个正样本并实际更新。
- 最终 64 环境 PyTorch 峰值 allocated 3,021,938,688 bytes，reserved 3,982,491,648 bytes；**不包含 PhysX 等非 PyTorch 显存**。运行时另有进程占约 12.5 GB 显存，未修改或停止该进程。
- [最终 64 环境 run manifest](legged_gym/logs/g1_motion_wm_dtera_preflight/audit_20260916_final64/run_manifest.json)、[结束证明](legged_gym/logs/g1_motion_wm_dtera_preflight/audit_20260916_final64/training_completion.json)。

## 尚不能靠修 bug 保证的效果

1. 当前满幅 action residual 理论上限为 `0.03×0.5 + 0.03×0.25 = 0.0225`；关节目标乘 action_scale 0.5 后为 0.01125 rad，约 0.64°。它适合小修正，不保证能补偿大幅步态错误。前 1000 轮还有 warmup。不要在导出端突然放大已训练残差；先观察 candidate/gated/applied、饱和率及验证误差，再决定独立的新训练预算实验。
2. demand_only 保持当前训练设定。confidence/risk 虽存在，但不是都参与当前执行门控；短 smoke 的 risk 正样本不足，不能声称 risk gate 已学会安全判断。full 模式必须单独验证。
3. 每环境 CPU corruption/数据传输仍可能限制大并行吞吐。64 环境通过不代表 1024 同样快或同样省显存；不为提速改变污染分布或减少历史长度。
4. 继承环境的 reset 含额外物理刷新步，完整 sim-to-sim 长序列时序一致性仍需后续对比验证；本次没有冒险重写 IsaacGym reset 生命周期。
5. 3 次更新无法证明长期数值稳定、步幅恢复或跌倒率改善。没有做实机安全认证。

## 固定验证计划与停止条件

训练命令及 JIT 导出见 [README-WM.md](README-WM.md)。只训练 train.yaml，验证选择 checkpoint，最后一次性报告 test.yaml；不要挑选表现最好的 motion。

预先固定的小验证集（均来自 val.yaml；按文件名顺序和长度选取，不按策略效果选择）：

| 类别 | motion |
| --- | --- |
| 综合 | accad/A5___pick_up_box.pkl |
| 综合 | bmlhandball/S01_Expert_Trial_upper_right_208.pkl |
| 综合 | bmlmovi/Subject_13_F_1.pkl |
| 走路 | accad/B11___walk_turn_left__135_.pkl |
| 走路 | kit/WalkInClockwiseCircle03_1.pkl |
| 走路 | omomo/Walk B22 - Side step left_poses.pkl |

固定 seeds 42/43/44，formal，速度/缩放 1.0。用 A/B/C/D/E 加 clean_base 分离 WM 收益、DTERA 收益与 clean 退化。先在 500/1000/1500 轮观察（500 尚未满 warmup），不按单个动作改 gain；报告完整序列和排除 24 帧两种结果。

每个动作的示例命令（显式替换 checkpoint 和动作、为各 checkpoint/motion 使用不同新目录）：

```bash
/home/hank/anaconda3/envs/twist/bin/python tools/run_motion_wm_dtera.py \
  --ckpt legged_gym/logs/g1_motion_wm_dtera_legs/legs_audited_seed42_v2/model_1500.pt \
  --export-preset motion_wm_v2 \
  --motion-file '/home/hank/TWIST（anyadapter）/track_dataset/twist_motion_dataset/accad/B11___walk_turn_left__135_.pkl' \
  --seeds 42 43 44 --mode-labels A,B,clean_base,C,D,E --no-video \
  --out motion_wm_dtera_outputs/legs_audited_1500_val_accad_walk
```

检查 joint/下肢 RMSE、步幅或髋膝幅度、root velocity、height、roll/pitch/yaw、max tilt、跌倒率、action rate、推理耗时；D 应分别与 B 和 A 比较，E 与 clean_base 比较。下肢与步幅详细曲线使用现有仿真诊断工具，不能用总 reward 替代。

立即停止并排查：NaN/Inf、分支无更新、base/WM 被更新、reference/history 错配、清单加载不完整。满 warmup 后若 D 在固定验证集不优于 B，或改善少数动作但明显增加跌倒，不继续盲目延长训练；先依据支路饱和、门控覆盖、抵消率定位，再做单因素新实验。性能阈值属于后续实验决策，不把本次 smoke 当作达标结果。
