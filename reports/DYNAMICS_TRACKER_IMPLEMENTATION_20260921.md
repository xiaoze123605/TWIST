# 完整可训练的动力学条件跟踪器

这次实现对应 [设计评审](ANYADAPTER_REDESIGN_REVIEW_20260921.md) 的新策略路线，**不是原论文 AnyAdapter 的逐层 adapter 复现**。保留历史辨识、多步预测监督、分阶段训练；整个 actor 从头训练。当前实现和短测试证明代码链路可运行，不证明性能已经超过 TWIST。

## 分支与 baseline 边界

- 原代码备份：`4806b2b`，已推送 `origin/feature/motion-wm-anyadapter-opentrack`。
- 新实现分支：`feature/dynamics-conditioned-tracker`。
- 原 TWIST / AnyAdapter / DTERA 的模型、PPO、环境、配置、实机入口不变。共享文件只新增任务和 runner 注册。
- 旧代码中的历史/动作语义问题没有通过修改共享代码修复，以免改变既有实验；新任务独立修正。
- 新权重、旧 JIT 和旧 observation normalizer 不能混用。旧 baseline 可继续按原命令运行；新策略是否退化必须另做评测。

## 状态、历史、动作的统一定义

控制频率约 50 Hz。实际控制周期、关节顺序、限位、PD 增益、力矩限制和动作模式都写入 checkpoint 与导出模型的 `deployment.json`。

\[
s_t=[0.25\omega_t^{body},\;g_t^{body},\;q_t,\;0.05\dot q_t]\in\mathbb R^{52}.
\]

保留所有踝关节速度。actor 使用带噪传感器，WM 使用独立保存的干净监督状态。root 线速度、高度和接触只进入 critic；actor 不使用仿真特权状态。参考 31 维包含高度、roll/pitch、相对 heading、参考局部线速度/yaw 角速度、23 维关节位置。参考采样使用当前时刻（`tar_obs_steps=[0]`），不要求实时动作源提供未来帧。增加用相邻两次参考做后向差分得到的关节速度，首次置零、限幅 ±20 rad/s；没有未来速度输入。

\[
h_t=[(s_{t-79},u_{t-79},v_{t-79}),\ldots,(s_{t-1},u_{t-1},v_{t-1})],
\quad o_t=[s_t,r_t,\dot q_t^{ref},u_{t-1},h_t].
\]

每帧历史为 `52 + 23 + 1 = 76` 维，最后一维是有效位；reset 后全部无效。总 actor 输入 `129 + 79 × 76 = 6133`，critic 输入 135。

PPO 保存 pre-tanh Gaussian 样本 `a_t`，不拿限幅后的目标反算 Gaussian log probability。参考中心模式：

\[
u_t=\operatorname{clip}(q_t^{ref}+\alpha\odot\tanh(a_t),q_{min},q_{max}).
\]

直接模式：

\[
u_t=\frac{q_{min}+q_{max}}2+\frac{q_{max}-q_{min}}2\odot\tanh(a_t).
\]

直接模式的最后一层 bias 初始化为默认站姿对应的 pre-tanh 值，避免从关节区间中点起步。两种模式都没有旧 base actor，也没有在旧策略动作上叠加 residual。参考中心模式仍有参考前馈，应与“完全直接输出”区分。熵项对应 tanh 后、最终关节限幅前的分布。

环境显式区分 `raw_sample`、`sent_target`、`applied_target`。历史和 WM 输入都是 **sent target**；执行器延迟由被辨识的闭环动力学负责，actor 不偷看内部 applied target。PD：

\[
\tau=\operatorname{clip}(K_p(u^{applied}-q)-K_d\dot q,-\tau_{max},\tau_{max}).
\]

没有部署端额外 EMA。动作平滑惩罚作用于实际发送 target 的变化。

## 动力学学习与阶段切换

\[
z_t=E_\phi(h_t),\qquad \hat s_{t+1}=F_\psi(s_t,u_t,z_t).
\]

预测器直接预测各状态分量的增量，并把 gravity 重新归一化；不强迫用末端速度积分 q。损失是 gyro、q、dq 三组平均 L1 加 gravity 方向误差。每步把 **预测前的状态** 与当步命令写入历史，再进行下一步自回归预测。终止转移本身参与监督，reset 后的转移从该序列移除，避免跨 episode 污染。

默认预测长度从 1 步逐渐增至 20 步（200 updates）；短测试直接用 20 步以覆盖完整实现。WM batch 默认 64，每个环境每个 WM epoch 抽一个序列。没有大型 replay buffer。

- Stage A：名义物理参数，传感器噪声，latent 对 actor 置零。完整 actor/critic 学跟踪，encoder/decoder 同时学预测。
- Stage B：载入 Stage A 权重，启用 latent，完整 actor 继续 PPO。第一次启用时将 actor 首层连接 latent 的列置零，所以切换瞬间的确定性输出保持一致，随后这些连接可训练。
- PPO 的梯度不进入 encoder；encoder/decoder 只接受 WM 损失。每轮先完成 PPO，再更新 WM，保证同一轮 PPO 内 encoder 固定。
- Stage B 保留约 30% 名义环境，其他环境使用温和摩擦、质量、CoM、PD 增益扰动，0–1 控制步延迟及推力。名义子集不加这些扰动。当前没有地形课程、单扰动/混合扰动细分或参考 Motion-WM 联合训练。
- time-limit 用 reset 前的 terminal critic observation bootstrap；物理失败与 motion clip 结束不 bootstrap。新 reset 不额外推进全部环境一次物理仿真。

## 运行方式

在仓库根目录、现有 `twist` 环境运行。若 Isaac Gym 找不到 Python 动态库，先设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`。

```bash
conda activate twist
python tools/train_dynamics_tracker.py \
  --task g1_dynamics_tracker --headless --num_envs 256 --max_iterations 3000 \
  --output legged_gym/logs/dynamics_tracker/nominal_seed42 --seed 42

python tools/train_dynamics_tracker.py \
  --task g1_dynamics_tracker_robust --headless --num_envs 256 --max_iterations 3000 \
  --warm-start legged_gym/logs/dynamics_tracker/nominal_seed42/model_3000.pt \
  --output legged_gym/logs/dynamics_tracker/robust_seed42 --seed 42
```

这是启动示例，3000 iterations 不是已经验证的收敛预算。默认数据使用已有审计过的 `train.yaml`。`--output` 必须是新目录，避免覆盖既有实验；每次保存配置、输入文件 SHA256、训练指标和 checkpoint。`--resume-checkpoint` 恢复 optimizer 和学习迭代；`--warm-start` 使用新 optimizer/新课程时钟。恢复不保存物理仿真状态，不保证逐 bit 接续轨迹。

完全直接输出的两个任务是 `g1_dynamics_tracker_direct` 和 `g1_dynamics_tracker_direct_robust`。它们必须使用自己的 Stage A 权重，不能直接载入 reference 模式 checkpoint。

短测试示例：

```bash
python tools/train_dynamics_tracker.py \
  --task g1_dynamics_tracker --headless --num_envs 4 --max_iterations 3 \
  --motion_file track_dataset/twist_motion_dataset/accad/B3___walk1.pkl \
  --smoke-test --output /tmp/new_tracker_smoke
```

评估与导出：

```bash
python tools/evaluate_dynamics_tracker.py \
  --task g1_dynamics_tracker_robust --headless --num_envs 16 --steps 1000 \
  --motion_file path/to/held_out_motion.pkl \
  --model path/to/model.pt --output path/to/evaluation.json

python tools/export_dynamics_tracker.py \
  --checkpoint path/to/model.pt --output path/to/policy.pt
```

评估输出关节/高度 RMSE、力矩饱和比例、物理失败、motion 完成、timeout 和 reset 数，并逐步核验 scripted 与训练模型的确定性目标一致。每个环境固定 clip、从零相位启动；失败会重启，RMSE 是所有存活 frame 的汇总，**不能单独作为完整动作成功率或 baseline 优越性证据**。完整 benchmark 仍需共同动作集合、同预算、多个训练与扰动 seeds，以及每个 clip 成功/失败的分层统计。

## 部署接口

`rsl_rl.modules.dynamics_tracker_runtime` 不依赖 Isaac Gym。`measured_state_and_reference` 接收按导出关节顺序排列的 q/dq、机身 gyro、`wxyz` 四元数和世界 yaw 的参考，构造与仿真一致的 state/relative reference。

```python
runtime = DynamicsTrackerRuntime('policy.pt', joint_names, device='cpu')
runtime.reset(current_sent_target)
state, relative_ref = measured_state_and_reference(q, dq, gyro, quat_wxyz, reference)
target = runtime.propose(state, relative_ref)  # 绝对关节目标，rad
# 由外部控制器按导出 dt / PD 增益发送 target。
runtime.commit(actual_sent_target)
```

辅助函数输入带 batch 维；runtime 处理单机器人。`commit` 必须记录实际发送的目标，而非执行器内部延迟后的未知值；如外部投影或停止改变发送目标，也必须反映到历史中。中断后恢复应 reset 历史。

当前交付了训练、离线仿真评估、TorchScript 导出和传感器推理接口；**没有改写或运行现有 DDS 实机控制入口**。新的 6133 维策略不能直接放进旧 AnyAdapter 2635/7001 维入口。实机接线还需将这个接口接到当前控制器，核对导出的关节顺序、控制周期与 PD 参数，并通过仿真迁移和硬件小幅验证。

## 验证记录

见同目录 `dynamics_tracker_validation_20260921/`。短训练权重仅用于实现检查，不作为可部署策略。尚未开展完整数据重训、多 seed 对照、MuJoCo 迁移或实机效果评估，因此现在不能保证比 baseline 更好。
