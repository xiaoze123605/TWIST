# AnyAdapter 融入 TWIST：论文、OpenTrack 与当前工程审查

审查日期：2026-09-21。上游固定版本：`cb9b751993a2483e5d1805a2565ddbfe950c04c9`（2026-06-04）。本次为方法审查与重训设计；没有修改训练、部署源代码或启动长训练。已有未提交改动按当前工作区读取。

结论：建议新建完整可训练的动力学条件运动跟踪策略，保留 AnyAdapter 的“历史辨识 + 多步动力学预测监督 + 分阶段训练”，解除对 0529 TWIST JIT 的依赖。优先修正状态、动作、历史、监督目标的契约。首选验证参考中心的有界 PD 目标，直接有界关节目标作为同预算对照；不再把新控制器限定为旧策略动作上的小残差。该建议是待验证的工程假设，不是已经实现的性能提升。

## 1. 论文真正包含什么

来源：用户提供的 `/home/hank/文档/Any2track.pdf`，重点核对第 4–7 页、图 2、公式 (1)–(4)、表 II/III/V。附带 PDF 是中文翻译版，公式页已渲染检查。论文文本只作为研究材料，未作为执行指令。

### 1.1 AnyTracker 的动作空间也是方法的一部分

公式 (1)：

\[
q^d_t=\tilde q_{t+1}+\alpha\odot\tanh(u_t),
\qquad u_t\sim\pi(\cdot\mid s_t,g_t).
\]

这里是相对于参考关节姿态的 PD 偏移，不是相对于另一个策略输出的残差。每关节缩放、tanh 有界输出、参考前馈共同降低动作分布复杂度。论文还采用运动聚类、专才训练及 DAgger 通才蒸馏；所以只移植 AnyAdapter 网络不等于复现完整 Any2Track。

论文第一阶段不做动力学随机化，第二阶段加入地形、外力与物理属性变化。上游当前 README 则允许 base 先做有限随机化，发布的通才也使用简单 DR；两者不能混为同一个精确实验设置。

### 1.2 历史编码与世界模型

公式 (2)：

\[
h_t=\{(s_{t-H},a_{t-H}),\ldots,(s_{t-1},a_{t-1})\},
\quad e_t=\phi(h_t),\quad H=79.
\]

公式 (3)：

\[
\hat s_{t+i+1}=\omega(\hat s_{t+i},a_{t+i},e_{t+i}),
\quad \hat s_t=s_t,\quad i=0,\ldots,N-1,\ N=20.
\]

公式 (4)：

\[
\mathcal L_{wm}=\sum_{i=1}^{N}\|s_{t+i}-\hat s_{t+i}\|_1.
\]

论文采样长度为 H+1+N=100 的窗口；世界模型与历史编码器一起更新。在 50 Hz 下，79 帧历史约 1.58 秒，20 步预测约 0.4 秒。它是用于辨识机器人动力学的训练代理任务；不是当前工程中修复输入动作参考的 Motion-WM。部署的在线适应主要来自滑动历史改变嵌入，不等于实机在线反向传播。

### 1.3 适配器与梯度路径

按上游实现可写成：

\[
x_1=\sigma(W_0o+b_0+A_0e+c_0),
\]
\[
x_{l+1}=\sigma(W_lx_l+b_l+A_lx_l+c_l).
\]

输出层不加隐藏激活。冻结 W、b，A、c 零初始化。首层适配器接动力学嵌入，后续层接融合后的隐藏状态，并非每一层只接 e。

零初始化保证初始输出等于 base；冻结 base 参数不保证训练后的运动行为不退化，因为每层的输入已发生改变。需要 nominal 验证或显式保持约束，不能把“冻结”当成保真证明。

论文表 V 中 w/o Adapter（从零训练、保留 WM）并非完全无效：无扰动 SR 为 89.6%，完整方法为 89.8%；物性扰动 SR 同为 80.6%。完整方法多数跟踪误差更优。这支持把完整可训练策略作为候选，也说明不能宣称去掉 adapter 必然更好。

## 2. 官方实现核对

上游源码位置：

- [网络与结构化动力学](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/brax_networks.py)
- [PPO 与多步世界模型损失](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/model_based_ppo_losses.py)
- [交替优化训练循环](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/train_model_based_ppo.py)
- [环境观测、动作及配置](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/envs/g1_tracking_adapter/train/g1_env_tracking_general.py)
- [部署说明](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/deploy/README_deployment.md)

已核对的关键行为：

| 部件 | 当前上游实现 | 移植含义 |
|---|---|---|
| 历史编码器 | 64/64 通道，kernel 9/6，stride 5/3，128 维嵌入 | 当前本地卷积结构基本对应 |
| Actor | 零初始化逐层加法适配器 | 不是只在动作末端叠加补偿 |
| 策略分布 | NormalTanhDistribution | PPO 保留未压缩采样及正确概率语义 |
| PD 目标 | reference q + action × scale；所读环境配置 scale=1.0 | 与论文逐关节 alpha 的文字设计不完全相同 |
| 世界状态 | gyro、重力向量、29 关节位置/速度、root height，共 65 维 | privileged decoder 可以用额外监督，actor 不应获得不可测真值 |
| 世界模型 | 输出 33 维增量，速度积分位置、gyro 更新重力向量、预测高度变化 | 本地 51 维 roll/pitch 版本属于改造 |
| 损失 | gyro/q/dq/height 为分组 L1，重力为方向点积损失 | “全部 L1”不精确等于当前上游 |
| 优化 | 先更新 WM+encoder，再更新策略/critic；默认策略不训练 encoder；监督控制损失默认 0 | 保留明确的梯度归属 |

上游也有值得隔离核验的疑点。环境历史每帧为 64 维 proprio + 29 维 motor target，共 93 维；但 `compute_world_model_loss` 的自回归历史更新按 `predicted.shape[-1]-1=64` 移位并追加 64 维预测，不是完整 93 维帧。总向量长度没变，帧内语义却会错位。并且 WM 使用 `data.action`（tanh 后的参考偏移），环境执行的是 reference 加偏移，当前 reference 没有作为 WM 显式输入。以上是对该 commit 的静态数据流判断，未运行官方 JAX 训练确认实际性能影响；应先做契约测试，不能把这些细节当成应照抄的算法定义。

## 3. 当前工程的具体问题与优先级

### P0：自回归历史时间错位（已做最小数值复现）

文件：`rsl_rl/rsl_rl/algorithms/ppo_anyadapter_opentrack.py:66`，父类 `ppo_any2track.py` 同样存在。

在线环境使用 `(state_t, executed_action_t)`，但 WM rollout 追加 `(predicted_state_t+1, action_t)`。正确的同契约更新应为：

\[
\hat h_{k+1}=\operatorname{shift}(\hat h_k)\oplus(\hat s_k,a_k),
\quad \hat s_{k+1}=\omega(\hat s_k,a_k,\phi(\hat h_k)).
\]

最小复现：输入 state=10、action=3，假设完美模型预测 11，代码第二步 history 最后一帧为 `[11,3]`；按在线契约应是 `[10,3]`。这不是参数调优问题。修复时先保存当前 state，再推进预测 state，并区分 reset 和终止掩码。

### P0：世界模型动作与历史动作不一致（静态确认）

`humanoid_char.py:120–125` 在延迟、裁剪之后把 `self.actions` 写进历史；`rollout_storage.py:297` 给 WM 的却是 PPO storage.actions。`ppo.py` 保存的是 actor 原始采样，runner 和 `PPOAnyAdapter.process_env_step` 没有另存执行动作。

无延迟且未裁剪时两者相等，因此不是每一步都出错；触发延迟/裁剪后契约就分裂。延迟目前在 `5000*24` 环境 step 后开启，评估影响时需核查具体运行计数与配置，不能据此把更早 checkpoint 的退化全部归因于延迟。

建议分别保存 `policy_sample`、`command_target`、模拟器 `applied_target`。PPO log_prob 用原采样；WM 用定义明确的控制输入。实机可观测的是最终发送命令，未必知道电机内部实际延迟后的 setpoint。首选让 WM 预测“发送命令→下一状态”的总闭环动力学，历史同样记录发送命令；或者仅将 applied_target 模型作为仿真特权辅助。不能在 actor 历史中偷用真实硬件不可测的 applied action。

### P0：踝速度清零与结构化积分冲突（代数核验）

`g1_mimic_distill.py:417` 将四个踝速度置零；`selected_state()` 从该 actor 观测截取 WM target；`actor_critic_twist_any2track.py:225` 又规定：

\[
\hat q_{t+1}=q_t+\Delta t\,\hat{\dot q}_{t+1}.
\]

于是运动中的踝关节，真实位置变化与“速度恒零”监督不能同时满足。例如 20 ms 内 q 改变 0.02 rad，积分需要缩放速度 0.05，但 target 速度是 0。

不要再从旧 base 的观测切片生成动力学真值。分离 noisy actor observation、encoder sensor history 与 clean simulator targets。旧 base 必须清零时可以只对旧 base 清零；新策略重新评估是否使用滤波后的真实踝速度。对没有可用速度的通道，预测位置增量或使用传感器观测模型，不同时强制错误的积分和零速度标签。

### P0：7001 维仿真路径没有对应的现成实机入口

sim server 已自动识别 7001，使用 79 帧及 action 后提交。`server_low_level_g1_real_anyadapter.py` 仍固定 20 帧、2635 输入，未开启新的提交契约；`server_low_level_g1_real_v2.py` 明确检查 1155 输入。这里是所检查入口的兼容性结论，不能把当前 7001 导出直接当成实机部署已打通。

### P1：所谓 OpenTrack 分支仍受旧策略表达限制

`actor_critic_twist_anyadapter_opentrack.py` 继承 `TwistAny2TrackActorCritic`。底层按固定 state_dict key 重建 0529 网络，冻结 motion encoder、backbone、LayerNorm、原 normalizer；`freeze_base=False` 在父类中被丢弃。因此改一个配置不能解除冻结。

其动作依然是 `q_default + 0.5*a`，没有迁移参考中心的 CAS；当前 Gaussian + clip 的语义也不是上游的 tanh Gaussian。逐层 adapter 已不是早期动作残差，不宜继续用“残差幅度不够”概括全部现象。

### P1：方向改善与关节退化存在，但根因未唯一确定

`evaluation_anyadapter_3000/comparison.md`：单动作、seed42、380 帧，raw joint RMSE 从 0.143694 增到 0.205699（约 +43%），yaw 从 0.584098 降到 0.123891。它证明该样本有质量交换，不证明所有动作都退化，也不证明缺少 tracking-error 分支就是原因。

当前 V6 reward 继承配置将 root velocity 增至 1.5，joint tracking 仍为 0.6，并增加原地运动约束。不同奖励尺度、饱和程度、运动类别会影响贡献，不能单凭系数断言根因。应记录各奖励分布及分组误差。

`sim_analysis/ema_reference_ablation_4900_20260921/README.md` 的单动作结果表明，EMA 平滑命令的同时可明显恶化 yaw/root，并使下蹲更浅。它支持把滤波放入训练模型，而不是只在部署追加。该报告的髋 yaw proxy 不等于实际足端朝向，不能单凭它排除 joint mapping、参考可行性与执行器误差。

## 4. 推荐的新结构：完整可训练的动力学条件 Tracker

### 4.1 保留两个职责清楚的世界模型

```
动作源 → 可选 Motion-WM → 因果参考 g_t → reference/task encoder ─┐
当前传感器 s_t + 短历史 ──────────────────────────────────────┤
传感器/发送命令长历史 → dynamics encoder → z_t ───────────────┤
                                                           ↓
                                              完整可训练 actor
                                                           ↓
                                    有界 PD 目标 → 执行器模型 → 机器人

训练时：history encoder + dynamics decoder → 多步真实状态预测损失
```

Motion-WM 处理参考噪声与延迟，Dynamics-WM 处理机器人响应辨识。不能因为名字相近就共享一个 latent 或混用预测目标。第一轮控制器比较先用 clean reference，之后再加入当前 clean/corrupt/wm 三模式；评价始终同时保留 clean motion 指标。

### 4.2 观测与条件化

建议传感器状态以 projected gravity、gyro、q、经过一致滤波的 dq、上次发送 target 为基础。真实 root 线速度、高度、接触可以作为 critic/decoder 特权监督；进入 actor 前必须有部署可用的估计器及相应噪声/延迟建模。

task 分支显式提供 `q_ref-q`、`dq_ref-dq`、相对姿态/目标根部速度、可用参考脚高等。在线动作源没有未来真值时，不允许训练 student 偷看未来；短期参考可以由过去样本因果估计，文件播放的未来 preview 作为单独实验。

默认从简单的拼接条件化开始：

\[
z_t=\phi(h_t),\quad
u_t=\pi_\theta(o_t,g_t,\operatorname{sg}(z_t)).
\]

如果证明策略忽略 z，再比较逐层 FiLM：

\[
x_{l+1}=\sigma((1+\gamma_l(z_t))\odot\operatorname{LN}(W_lx_l)+\beta_l(z_t)).
\]

所有 actor 参数可训练；FiLM 的 gamma/beta 初始接近零，只是优化初始化，不是旧 base 动作残差。不要一开始同时加 ensemble、risk gate、多个误差分支，以免无法识别收益来源。

### 4.3 两种动作方式

首选 A（参考可靠时）：

\[
q_t^{cmd}=\operatorname{project}_{\mathcal Q}
(\bar q^{ref}_{t+1}+\alpha\odot\tanh u_t).
\]

这是独立策略的参考中心动作空间，\(\bar q^{ref}\) 必须是部署真正可得、已经过延迟/噪声处理的参考。其优点是让 actor 学动力学补偿而非从默认站姿重建整段动作；缺点是坏参考会直接前馈进命令，且 alpha 太小会限制恢复能力。alpha 应按各关节可行范围与名义跟踪实验确定，不能直接照抄 29 DOF 数值。

对照 B（完全没有参考加法）：

\[
q_t^{cmd}=q^{mid}+r\odot\tanh u_t,
\quad q^{mid}=(q_{max}+q_{min})/2,\quad r=(q_{max}-q_{min})/2.
\]

q_ref 仅作为网络输入。它没有旧策略残差，也没有参考残差；代价是更复杂的动作学习和可能的较差样本效率。两者都保留统一的最终限位、执行器模型与命令日志。

若使用 tanh Gaussian，存储 pre-tanh 样本，正确处理变量变换概率/熵；不能简单把输出 tanh 后继续拿原 Gaussian 对该值计算 log_prob。WM 输入建议是最终发送的关节目标，使相同控制输入具有稳定物理语义；采用参考偏移作 WM 输入时必须同时输入对应参考。

## 5. 动力学学习与训练流程

### 5.1 监督目标与损失

encoder 使用真实可获得的 noisy history；decoder 可预测干净 gyro、重力方向、q/dq，并辅助预测 root velocity/height/contact。特权目标不会自动导致泄漏，前提是推理 encoder/actor 不需要这些真值。

建议损失（新设计，不是论文原式）：

\[
\mathcal L_{dyn}=\sum_{k=1}^N\lambda_k m_k
\left[\sum_j w_j\left\|{\hat y_{t+k,j}-y_{t+k,j}\over\sigma_j+\epsilon}\right\|_1
+w_g(1-\hat g_{t+k}^{T}g_{t+k})\right].
\]

重力向量先归一化。尺度 sigma 从训练数据估计并记录；清楚区分分量 sum/mean，不直接移植上游 500/500 权重。预测跨度先 1，再 5、10、20；始终用闭环预测回填，不能用未来真值偷偷替换预测历史。

终止时最好保存 terminal observation，并让 mask 在 episode 内传播；或在 reset 后明确作为新序列起点，不能把 reset 状态当成物理后继。历史不足时用一致 padding 加 validity mask；79 帧不足不应伪装成 79 帧真实经验。

建立以 episode_id 和时间戳索引的紧凑 sequence buffer，存状态/命令/后继，不为每个时间步重复存整段历史。保留多个最近 rollout，而 PPO 仍使用 on-policy 数据。当前 64 env / 16 WM minibatch 相当于每 batch 仅 4 个序列、每轮每 env 一条 20 步窗口；噪声与覆盖值得测量，不能从 iteration 数推断充分训练。

### 5.2 分阶段，而不是强 DR 下所有网络同时从零学

1. **协议与名义技能**：先完成 P0；新 actor 在 nominal/小 DR 下训练，clean reference，z 用零或固定输入。按走路转弯、下蹲、踢腿、上肢、接触转换分层验证。若统一训练出现类别冲突，再训练专才并 DAgger 蒸馏；不是无条件先增加大量教师。
2. **动力学预训练**：用能运动的策略收集 nominal + 渐增扰动数据，训练 encoder/decoder，检查多步验证误差和 z 的实际贡献。对尚未接触到的摩擦/地形，历史无法提前辨识，不能把 z 当成全知环境参数。
3. **完整策略鲁棒训练**：开启 z，逐步扩展质量/CoM/摩擦、执行器增益/延迟、外推力与地形。整个 actor 可更新，encoder 初期只由 WM 更新。保留 nominal 采样，避免只学保守站稳；可从 30% nominal / 50% 单扰动 / 20% 混合扰动试起，这是建议起点，不是论文数值。
4. **按证据联合微调**：若保持 stop-gradient 无法利用 z，再比较小权重 PPO 梯度进入 encoder。交替更新时控制 encoder 漂移，PPO old log_prob 必须保留采样时数值，监测更新 encoder 后的行为 KL；可用按轮冻结的 target encoder 降低非平稳性。
5. **参考鲁棒性与跨仿真**：再加入参考污染/Motion-WM，冻结数据 split；IsaacGym 与 MuJoCo 同控制契约对照，最后实机小范围验证。

保真约束可用当前新名义策略作 teacher，在 nominal 样本上弱 BC/KL 或定期蒸馏，不锁死旧 0529 base；强扰动状态不强制模仿可能失效的名义动作。这样比永久冻结整套旧架构更有调整余地。

## 6. 运动表现和 sim-to-real 需要一起重设计

### 6.1 奖励与目标

对足端外八、膝部轨迹和下蹲深度，先核对重定向参考的 FK、关节顺序和可行性，再增加相对参考的足端旋转、膝/足位置、骨盆高度、分组关节误差指标。不能固定惩罚髋 yaw 非零，因为合法转弯和蹲姿可能需要它。

足端姿态可用 SO(3) geodesic 或足前向轴误差；接触、滑移奖励由参考接触相位门控。通用 air-time 目标和原地速度阈值要按运动类别审查，避免鼓励所有动作使用同一种步态。

平滑项作用于最终命令与真实响应：记录 PD target 的一阶/二阶差分、真实关节加速度、扭矩与饱和率。强惩罚 raw action 可能损害高速动作；参考中心动作也不能只惩罚 offset，忽略 reference 前馈的抖动。部署 EMA 如要启用，必须作为训练执行器模型的一部分。

### 6.2 执行器与部署一致性

逐项对齐关节顺序、23 vs 29 DOF、锁定/外控手腕、IMU frame、quaternion 顺序、角速度尺度、控制周期、默认角、Kp/Kd、扭矩限幅和参考时间戳。当前 g1.yaml 踝 pitch default=-0.18，训练为 -0.2；先判断是否有意校准，不能直接覆盖硬件配置。

仿真使用 PD torque clip；所检查的旧 `g1_wrapper.send_robot_action` 直接发送 q/Kp/Kd，不能因为类里存在 torque_limits 就认为已实现同一限幅。上游部署提供 torque projection，可参考其思想：

\[
\tau^*=K_p(q^{cmd}-q)-K_d\dot q,
\quad \bar\tau=\operatorname{clip}(\tau^*,-\tau_{lim},\tau_{lim}),
\]
\[
q^{sent}=q+K_p^{-1}(\bar\tau+K_d\dot q).
\]

这只是基于测量与已知 PD 的投影模型，不能假设等于硬件内部真实力矩。若实现，训练与部署均建模，并按实际发送命令回填历史。

随机化应围绕实际测量分布展开，尤其是动作/传感器延迟、控制 jitter、摩擦、负载、PD 增益误差、编码器偏置、速度滤波。质量/摩擦等慢变量保持 episode 内稳定；外力与包丢失采用具有持续时间的过程。每帧完全独立乱跳的动力学并不适合历史辨识。

## 7. 文件级实施建议

| 区域 | 建议变更 |
|---|---|
| `g1_mimic_distill.py` / 新 env | 独立 sensor state、clean dynamics target、task reference；消除踝速度监督冲突 |
| `anyadapter_history_mixin.py` | 定义统一的状态/发送命令契约、时间戳和 reset validity |
| `humanoid_char.py` | 明确 sample→target→延迟/执行器→physics；导出各阶段日志 |
| `rollout_storage.py` | PPO 样本与 WM command 分开；compact 序列数据、terminal state |
| `ppo_any2track.py` / `ppo_anyadapter_opentrack.py` | 修复预测历史时序，独立 clean target、mask 与动作输入 |
| 新 actor 模块 | 原生可训练 tracker，参考中心/直接目标可切换；无 base JIT 必需项 |
| 新 PPO / runner 分支 | 明确 actor/critic 与 encoder/decoder 的 optimizer；避免继承残差专用参数收集 |
| 新 config | 显式列出全部观测、奖励、DR、归一化及 fresh-training 参数，不继承旧 continuation 超参 |
| 导出与 real/sim runtime | 一个版本化 schema：joint names、scales、dt、history order、action mode、limits、filter、normalizer；加载时校验 |

新建独立 task，例如 `g1_dynamics_conditioned_track`。现有 base、DTERA 和 layerwise adapter 留作对照。仅设 `freeze_base=False`、改几个 reward 或把继续训练的 5e-6 学习率用于新 actor，都不构成完整重训方案。

## 8. 实验设计与验收

第一阶段用最少的对照定位收益，保持相同数据 split、训练环境步数、训练 seeds 和评估扰动 seeds：

| 实验 | 用途 |
|---|---|
| 当前 0529 base / 当前 adapter checkpoint | 历史基线，注明预训练预算不同 |
| 修正 P0 后的 frozen layerwise adapter | 分离实现缺陷与方法上限 |
| 新 actor + DR，无 dynamics latent | 判断仅重训/动作空间是否已经带来收益 |
| 新 actor + 历史 latent，仅 PPO 学 encoder | 判断 WM 监督是否必要 |
| 新 actor + WM latent + 参考中心输出 | 主方案 |
| 新 actor + WM latent + 直接关节目标 | 检验完全无参考残差方案 |

先在 clean reference 完成上述比较，再对胜出结构做 raw/corrupt/wm 对照；不要同时改变所有因素。至少报告 3 个训练 seeds（资源允许时）与多扰动 seeds，并区分它们。

验证维度：名义、单独质量/摩擦、负载、持续力、冲量、延迟、地形、组合扰动；按运动类别统计 SR/跌倒率、完整序列 joint RMSE、腿部/手臂误差、MPJPE、root/yaw、足端朝向、参考接触一致性、滑移、扭矩饱和、实际关节加速度及推理时间分位数。失败样本单独计数，不能只对成功片段算漂亮误差。

latent 因果检验：正常 z、z=0、跨环境打乱 z、冻结旧 z；若跟踪表现几乎不变，不能因为 WM loss 降低就声称学到了适应。扰动发生后看恢复时间；长/短历史（例如 20/40/79 帧）的差别用 held-out 突变与慢变量验证。

建议预先声明 acceptance margin：例如名义 joint/足端误差相对同动作空间的无适应策略不恶化超过 5%，扰动成功率/恢复时间有可重复改善，且不增加扭矩饱和或实际高频振动。这是可调整的工程门槛，不是论文保证；统计区间与最差动作同样报告。

本次证据边界：完成论文公式原页核验、固定上游 commit 的代码审查、本地训练/导出/部署路径审查、自回归历史最小数值复现及踝部监督代数核验；未运行新策略训练、官方 JAX 训练或新的物理 rollout。现有历史评估作为已有记录引用，不能替代后续对照实验。
