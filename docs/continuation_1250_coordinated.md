# 从 1250 继续全身协同训练

新训练和动作参考均使用原训练的 `model_1250.pt`。上下肢保持全身策略联合输出，全部 adapter 层可训练；TWIST 主干、历史编码器和动力学 WM 延续冻结设置。

## 启动

```bash
cd '/home/hank/TWIST（anyadapter）'
bash tools/run_1250_coordinated.sh coordinated_1250_seed42_20260926 \
  --source-checkpoint legged_gym/logs/recovery_4900/transanchor_norm_seed42_20260925/model_1250.pt \
  --seed 42 --num_envs 4096 --max_iterations 10000 --save-every 100
```

`max_iterations=10000` 表示新分支从 1250 开始追加训练 10000 次。新目录 `model_0.pt` 与参考 1250 权重完全一致，`model_100.pt` 表示追加 100 次更新。不要对原目录的 model_1250.pt 使用 `--resume-from`；新起点只继承权重，重建优化器、EMA 和回放池。

日志及模型：`legged_gym/logs/recovery_1250/coordinated_1250_seed42_20260926/`。`train_metrics.jsonl` 记录训练指标，`anchor_metrics.jsonl` 记录固定尺度锚定、尾部约束、六组回放数量、KL 和探索标准差。

## 这次针对性修改

- 参考动作锚定的尺度固定为 0.087，与探索标准差独立。默认不退火探索；actor LR 从 1e-6 缓慢降至 2e-7，锚定系数从 0.08 在 500 次更新内降至 0.04。
- raw/wm × 原地、转弯/过渡、常规运动，共六组历史优先级蓄水池。每组最多 4096 条，每个 rollout 每组最多插入 128 条；按非空组均衡采样，防止常见状态快速挤掉罕见状态。
- 过渡权重保留 1 的下限，最大为 3，不再用批均值归一化削弱常规状态锚定。最差 10% 样本增加系数 0.01 的软锚定代价；没有硬性动作锁定。
- 增加实际接触脚滑移（-0.25）、相对参考姿态的超额倾斜（-1.0）、转弯加权速度误差（-0.15）代价。参考本身的倾斜、抬腿和协同动作仍允许。
- 原地惩罚还要求参考腿部速度低，避免只根据 root 速度误罚踢腿、踏步、下蹲。
- wm 从 raw 历史预热开始，与当前部署 refiner 一致；全身 motion 参考不被固定或替换。
- checkpoint 原子保存 live 权重、匹配的优化器、EMA、回放池、计数和 Torch 随机数状态。EMA 单独导出评估。恢复会检查参考哈希、代码和配置；模拟器物理状态不恢复，环境重新初始化。

这些是待验证的优化配置，短训练只能验证代码和恢复流程，不能证明策略表现已经改善。

## 中断与恢复

Ctrl+C 或向训练 PID 发送 SIGTERM 后，等待当前更新完成并保存。启动器不自动重启。

例如保存到了新分支 model_1800.pt：

```bash
bash tools/run_1250_coordinated.sh coordinated_1250_seed42_20260926 \
  --source-checkpoint legged_gym/logs/recovery_4900/transanchor_norm_seed42_20260925/model_1250.pt \
  --seed 42 --num_envs 4096 --max_iterations 10000 --save-every 100 \
  --resume-from legged_gym/logs/recovery_1250/coordinated_1250_seed42_20260926/model_1800.pt
```

这会训练剩余 8200 次，不是重新追加 10000 次。不读取原训练目录的 `bank_state_latest.pt`。

## 按历史最好模型选择，允许曲折改善

建议每 200–500 次更新评估保存模型，同时保留 live 与 EMA 候选。一次退化不自动中断训练，也不要求优于紧邻 checkpoint。最终晋级仍检查相对 1250 的综合表现、局部退化和新增摔倒；之后用长动作确认集及其他 seed 验证。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/evaluate_1250_candidate.py \
  --checkpoint legged_gym/logs/recovery_1250/coordinated_1250_seed42_20260926/model_500.pt \
  --output reports/recovery_1250/coordinated_1250_seed42_20260926_500_ema \
  --ema
```

不加 `--ema` 则评估 live 权重，使用另一个 output 目录。基准为 `reports/recovery_1250/reference_1250_dev/` 中已完成的 1250 固定协议评估。`comparison_to_1250.json` 的 accepted 仅为候选筛选门槛，不会停止训练。仿真进程异常每案例最多重试三次，不将不完整案例计入结果。
