# TWIST 本地数据与 SuperMimic 训练实现总结

日期：2026年11月9日。

已接通使用本地 TWIST 数据、从随机参数开始的 SuperMimic 训练流程，并通过真实 GPU 物理短测、独立进程恢复和 JIT 导出检查。本轮没有下载 BONES-SEED，也没有启动长训练。

## 本次实现

新代码位于 `twist_supermimic/`，使用原生 TWIST 的控制网络和动作编码器。输入为当前参考31维、当前机器人观测74维及10帧历史，共1155维，输出23个关节动作。整个控制网络参与更新。

训练循环包括：

1. 在 Isaac Gym GPU 仿真中执行当前策略，记录真实状态、请求动作、实际执行动作、PD目标、奖励、接触及终止信息。
2. 使用连续8步真实数据更新动力学模型。
3. 固定动力学模型参数，让策略在模型中执行32步，计算预测奖励和末端价值。
4. 沿预测轨迹计算梯度，更新控制策略。
5. 使用预测轨迹的收益更新价值网络，继续采集新策略的真实数据。

策略损失中的关键代码是：

```python
loss = -objective.sum() / self.cfg.pi_batch
loss.backward()
```

`objective` 包含预测轨迹奖励和末端收益估计。负号使减小损失对应增加收益；`backward()` 计算控制参数应该如何改变。动力学模型在这次策略更新中保持参数固定，动作到未来状态之间的导数仍然保留。

新训练没有教师策略、学生模仿损失或PPO更新，也没有加载heading300、raw48、外接纠偏器或AnyAdapter的参数。heading300仅用于独立基准评价。

动力学模型与价值网络只在训练时使用。导出的控制策略可以单独执行。JIT同时支持原生1155维输入，以及旧部署入口附加历史后的7001、7003维输入；控制网络统一读取前1155维。使用原有真实7003维观测检查后，两种旧输入与原生控制输出误差均为零。实机电机程序没有在本轮运行。

方法参考：[SuperMimic论文](https://arxiv.org/pdf/2610.28480)、[Uni-SuperMimic固定版本](https://github.com/Motphys/Uni-SuperMimic/tree/d78a5e2bc26ee26c3a94ae10418b7fb9018836ee)。

## 本地数据

| 项目 | 结果 |
| --- | ---: |
| 可用独立动作 | 15,301条 |
| 50Hz控制帧 | 7,980,092帧 |
| 训练动作 | 13,764条 |
| 开发动作 | 762条 |
| 最终测试动作 | 759条 |
| 另行保留的评价动作及同源片段 | 16条 |
| 最后一次完整整理耗时 | 21.06秒 |
| 紧凑动作库 | 约943MiB |

数据源为已有的 `twist_dataset.yaml`。原清单中的 `mocap1/6.pkl` 缺失，以及 `mydata2/15_seg00.pkl` 重复，均写入转换记录。保护动作及已知缺失文件保存在独立配置 `configs/twist_supermimic_local_motions.json` 中，数据准备不依赖历史训练报告。

划分单位为完整源动作，同一动作的切片保留在同一集合。当前数据没有完整表演者元数据，因此本次划分不代表表演者之间的独立划分。77、20及四条既有独立动作已经保留评价，数字编号使用真实文件映射。

采样一半来自全部训练动作，四分之一来自实际行走帧，四分之一来自实际下蹲、起身相关帧。全库采样同时使用均匀起点和按真实执行代价调整的困难片段起点。

读取紧凑文件后重新规范化姿态四元数。动作末尾继续保持最后姿态，并清零参考速度，供32步预测使用。

## 已完成的检查

全部检查使用现有真实动作、实际采集状态或已有真实观测记录。

| 检查 | 结果 |
| --- | --- |
| GPU物理与GPU优化 | Isaac Gym GPU Pipeline已启用 |
| 模型及策略更新 | 2048宽度世界模型、32步策略预测均完成实际更新 |
| 世界模型动作梯度 | 自动求导0.6259953，中心差分0.6259918 |
| 策略更新时保持世界模型固定 | 参数逐项检查通过 |
| 实际PD目标与缓存 | 与实际执行动作计算结果一致 |
| 独立进程恢复 | 恢复后完成新的模型和策略更新 |
| 完整训练入口 | 完成两轮，独立进程恢复后完成第三轮 |
| 定期评价与保存 | 三个检查点均保存完整开发动作评价 |
| JIT与检查点输出 | 最大误差0 |
| CPU独立导出 | 通过，无需创建训练仿真 |
| 旧7001、7003维真实输入 | 与原生策略输出误差均为0 |
| 运动学线速度 | 与自动求导的最大差约1.16×10⁻⁶ |
| 运动学角速度 | 与自动求导的最大差约2.06×10⁻⁶ |
| 姿态规范化与末帧保持 | 通过 |
| MuJoCo中的77号播放 | GPU推理、CPU物理，完整执行16帧接口检查 |
| heading300基准入口 | 使用真实历史和航向信息完成16帧GPU物理检查 |

主要证据：

- `artifacts/twist_supermimic_local/verification/verification.json`：梯度、真实动作、JIT及恢复检查。
- `artifacts/twist_supermimic_local/verification/interfaces.json`：运动学、输入维度和数据划分检查。
- `artifacts/twist_supermimic_local/runner_ready/runner_verified.json`：完整入口及恢复检查。
- `artifacts/twist_supermimic_local/runner_ready/evaluation/`：独立动作回合的评价和1、8、16、32步预测误差。
- `artifacts/twist_supermimic_local/mujoco_complete/result.json`：MuJoCo控制接口结果。

工程检查采用4个并行环境和小批量数据，世界模型保持正式配置的三层2048宽度，预测长度保持32步。该检查中PyTorch分配显存峰值约1.67GiB，保留显存峰值约1.80GiB；这些数值不包含完整PhysX占用，也不代表4096环境的显存需求。

工程模型只进行了少量参数更新，开发动作仍出现失稳。当前导出文件保留作接口检查，尚未得到经过验收的效果候选。完整动作稳定性、独立动作平均表现和20%航向保护需要通过正式训练后的真实评价判断。

## 实现中处理的问题

定期评价采用独立进程。Isaac Gym在同一进程创建第二个GPU仿真时会异常退出，进程隔离解决了训练与评价之间的冲突。评价前释放未使用的CUDA缓存，独立恢复检查也会先释放前一个检查占用的GPU对象。

真实轨迹的运动学和稳定性指标采用批量计算。训练奖励保持全身跟踪定义，评价单独计算躯干roll、pitch及其角速度误差，同时记录关节、速度、5—20Hz抖动、航向和接触情况。

数据转换使用局部的Pickle模块名称映射，避免在读取数据时提前修改NumPy全局模块。帧数计算处理了浮点舍入，防止少保留最后一帧。JSON文件明确使用UTF-8保存，支持中文路径和记录。

另一个项目目前正在使用GPU。本轮没有执行2048、4096环境的大规模性能校准。相应测量入口已经提供；正式训练总耗时和峰值显存需要在GPU空闲时测量。

## 命令

以下命令从仓库根目录执行。

### 数据准备与验证

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/prepare_twist_supermimic.py
```

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/verify_twist_supermimic.py
```

完整入口检查支持指定新的输出目录：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/check_twist_supermimic_runner.py \
  --output artifacts/twist_supermimic_local/runner_recheck
```

### GPU性能校准

GPU空闲时分别执行，比较`profile.json`中的中位轮次耗时与显存记录：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/profile_twist_supermimic.py \
  --num-envs 2048 --output artifacts/twist_supermimic_local/profile_2048
```

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/profile_twist_supermimic.py \
  --num-envs 4096 --output artifacts/twist_supermimic_local/profile_4096
```

测量同时记录PyTorch峰值和每0.5秒采集的整卡显存。整卡读数包括其他进程，选择配置时以约22,000MiB为目标上限。

### 正式训练与恢复

已准备配置：`configs/twist_supermimic_local.json`。默认4096环境、2048有效批量、512分批大小、前500轮模型热身、最多86,000轮及单次16小时上限。本轮没有运行以下长训练命令：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/train_twist_supermimic.py \
  --config configs/twist_supermimic_local.json
```

正式训练从随机参数开始。它产生检查点后，使用以下命令恢复：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/train_twist_supermimic.py \
  --resume artifacts/twist_supermimic_local/seed0/training_checkpoint.pt
```

训练目录保存检查点、开发集最佳检查点、世界模型、JIT、逐轮记录、当前进度和独立评价结果。`best_checkpoint.pt`表示开发集评分最低的训练检查点，最终候选验收状态单独记录。

### JIT导出及MuJoCo复现

本轮已实际执行CPU导出：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/train_twist_supermimic.py \
  --resume artifacts/twist_supermimic_local/verification/training_checkpoint.pt \
  --export-only --device cpu \
  --output artifacts/twist_supermimic_local/export_cpu
```

复现16帧控制接口检查：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/play_twist_supermimic.py \
  --model artifacts/twist_supermimic_local/export_cpu/policy_jit.pt \
  --motion mocap1__77 --device cuda:0 --frames 16 --headless
```

去掉`--headless`可打开窗口；去掉`--frames 16`可播放完整动作。当前模型仍是工程检查模型。

## 本轮结束状态

保留的本轮数据和验证文件约1.7GiB。已清理中断调试产生的临时目录和重复检查点，本轮仿真及验证进程均已结束。BONES-SEED保持未下载，长训练保持未启动。
