# TWIST / Motion-WM / AnyAdapter：架构与方法诊断（2026-10-13）

**当前最有证据的判断：进度卡在基础跟踪与误差恢复能力、动力学模型的状态/动作语义、以及训练和部署的接触动力学一致性。现有优化主要限制策略“改变多少”，没有直接解决策略“如何把偏离的机器人带回正确轨迹”。继续增加动作锚定、降低学习率或延长同配置训练，缺乏新的依据。**

本次检查训练环境、PPO/储存、各类Adapter、两个世界模型、JIT导出、参考处理、MuJoCo控制、实机接口与评估链路，并核对近期和早期实验记录。不是所有文件的逐行审计。新增10次完整77仿真与确定性接口探针；未训练、未修改生产控制/网络/奖励、未替换checkpoint。所有写入位于本仓库的新诊断目录。原来的用户改动保留。

证据分三层：**实测事实**来自本次运行或重新核查的现存产物；**支持的解释**来自结构与干预共同指向的机制；**待验证假设**需要独立训练/多动作试验。本次不能声称找到了所有问题的唯一原因，也没有已经修复效果的候选。

## 1. 本次新增实验：不改变策略权重也会出现显著退化

源策略为heading300，7003→23，Adapter gain=0.5，EMA=0，投影margin=0.02，knee_min=0。固定已记录的raw/WM参考；MuJoCo 3.2.3、Torch 2.4.1、CPU、seed42、50Hz控制。每次均64.14秒、3207控制周期，共641.4模拟秒、32070周期。

采用现有`RealTimePolicyController`，以已有`RecordedReference`回放不可变参考。仅移除墙钟等待，加速离线播放；原物理步长与控制周期保持，只有明确的physics_2ms干预例外。没有Redis连接、实时输入或机器人连接。raw/WM两个null的qpos、qvel、joint、RPY、动作和PD目标与既有基线逐元素相同；其余每次clean/processed参考都逐元素相同。

|唯一干预|模式|Yaw RMSE(°)|关节RMSE(rad)|根速度RMSE(m/s)|末10秒相对null的Yaw差RMS(°)|
|---|---|---:|---:|---:|---:|
|无：精确null|raw|67.777|0.138451|0.109049|0|
|初始根Yaw +10°|raw|71.098|0.138137|0.105904|27.868|
|初始根Yaw −10°|raw|71.928|0.138828|0.107131|23.768|
|无：精确null|WM|32.080|0.139306|0.103864|0|
|初始根Yaw +10°|WM|66.893|0.139162|0.111414|124.528|
|初始根Yaw −10°|WM|54.191|0.139901|0.103133|77.303|
|仅足底改成训练URDF的两圆柱|raw|100.077|0.143594|0.178835|—|
|仅足底改成训练URDF的两圆柱|WM|34.398|0.142176|0.173392|—|
|仅物理dt=2ms、decimation=10|raw|27.237|0.154930|0.116956|—|
|仅物理dt=2ms、decimation=10|WM|53.768|0.153429|0.117028|—|

10组均未触发相同的跌倒代理（height<0.35m或|roll/pitch|>1rad）。不能以此宣布跟踪或稳定性合格。表中末10秒差值衡量轨迹分离，不是对参考的跟踪误差；后者由Yaw RMSE给出。

曲线采用[-180°,180°]的周期误差，跨边界时断线，避免把角度包裹误画成瞬时大旋转。

**实测事实：**WM中±10°初始偏差使同一策略的整段Yaw误差分别增大约109%和69%；后半程没有可靠返回原轨迹。更改接触几何或积分步长也会明显改变结果，关节误差变化与Yaw变化并不一致。

**支持的解释：**当前基线的长时表现对初始状态和接触动力学有明显依赖；首次Actor更新的微小变化是在这一敏感系统上施加扰动。它不是仅靠Adam、critic预热或权重保存一致性就能解释的问题。

**边界：**这些是一个动作、一个checkpoint、一个确定性seed的干预。没有估计完整吸引域、Lyapunov指数或所有扰动的收缩性。脚型或dt的替换没有产生可晋级模型，也没有复现整个PhysX训练环境。

![初始朝向干预：参考Yaw误差](../reports/architecture_diagnosis_20261013/initial_yaw.png)

![接触几何与步长干预](../reports/architecture_diagnosis_20261013/model_contract.png)

## 2. 最需要修正的实现问题：动力学世界模型的数据契约不自洽

### 2.1 四个踝速度被屏蔽，却仍被当作物理积分标签

现有训练[`g1_mimic_distill.py`](../legged_gym/legged_gym/envs/g1/g1_mimic_distill.py)的`compute_observations`第415–417行将四个踝速度置零；仿真[`server_low_level_g1_sim.py`](../deploy_real/server_low_level_g1_sim.py)第655–661行也如此。`ANYADAPTER_STATE_INDICES=31:82`又直接从这份TWIST观测抽取51维动力学状态。因此不仅冻结TWIST看不到踝速度，Adapter历史和世界模型目标也继承了零速度。

在完整3207个真实消费输入上检查：

|量|raw|WM|
|---|---:|---:|
|真实四踝速度RMS(rad/s)|0.9303|0.8847|
|真实四踝速度最大绝对值(rad/s)|9.4862|9.0993|
|Actor消费的四踝速度最大绝对值|**0**|**0**|
|相邻20ms踝角变化RMS(rad)|0.01956|0.01846|

[`Any2TrackWorldModel.forward`](../rsl_rl/rsl_rl/modules/actor_critic_twist_any2track.py)第218–226行强制：

```text
q_next_pred = q_current + scaled_velocity_next_pred × 0.02 / 0.05
```

同一输出既要解释真实的q变化，又要拟合置零的速度标签。使用实际OpenTrack分支的L1系数，单步单踝的目标可写成（省略共同归一化常数）：

```text
L(v) = |0.4 v − Δq_real| + 0.5 |v|
```

当Δq_real≠0时，两项无法同时为零。对独立单步输出，最优是v=0、预测角度不变。确定性数值探针取Δq=0.02rad：v=0时loss=0.02；精确匹配角度需v=0.05，loss反而为0.025。

**已确认的是目标冲突及信息丢失，不是“它单独造成全部抖动”。**多步自回归和共享网络还会增加耦合，不能把单步oracle说成整个训练问题的全局最优。当前续训已冻结history encoder/world model并把WM loss设为0；此缺陷也不是正在运行一个错误的动力学预测器实时控制机器人。部署只使用其历史编码器，预测器本身不参与动作前向。

建议：保持冻结TWIST的1155维输入契约，在Adapter/世界模型专用状态中增加未屏蔽的踝速度。真实速度可滤波、裁剪，但不能把被屏蔽的策略输入当成物理真值。世界模型的监督应来自独立的真实状态；决定采用端点速度近似积分、区间平均速度积分，或独立预测Δq，需要写清定义并验证。新状态和编码器需要独立重训/蒸馏，不能直接给旧JIT填入新的含义。

### 2.2 世界模型使用指令动作，环境历史使用延迟后的执行动作

[`HumanoidChar.step`](../legged_gym/legged_gym/envs/base/humanoid_char.py)第102–125行可随机选择当前或前一条动作，再裁剪，历史提交的是`self.actions`。`G1MimicDistill`的counter初始化使当前学生路径很早便满足动作延迟开启条件。当前训练manifest也开启`action_delay`。

另一方面，[`RolloutStorage.world_model_sequence_generator`](../rsl_rl/rsl_rl/storage/rollout_storage.py)第295–299行传出的是`storage.actions`，即PPO采样指令；[`PPOAnyAdapterOpenTrack._autoregressive_world_model_loss`](../rsl_rl/rsl_rl/algorithms/ppo_anyadapter_opentrack.py)把它直接传给预测器，并把同一指令写入自回归预测历史。没有读取真实执行动作。

本次通过实际generator和实际loss函数做了确定性探针：采样指令+0.5，next observation已提交的执行动作−0.25，预测器接收到的仍是+0.5。两步探针均有效、未涉及reset。这确认了代码路径的不一致，但不提供历史正式训练中的发生率。

这类模型可选择建模“含执行器延迟的指令→状态”，或“真实执行目标→状态”。前者必须明确动作队列/延迟状态或概率动力学，并维持历史语义；当前实现把两种定义混用。另有PD目标投影，指令也不完全等于最终关节目标。

建议：PPO保留原采样动作、log_prob和优势，不改变其概率语义；为WM单独保存`executed_action`或投影后的`executed_pd_target`，按相同joint order转换并按同一定义维护自回归历史。若采用指令模型，则另建带执行器队列的状态。不要直接覆盖`storage.actions`，那会损坏PPO目标。

## 3. 架构上的主要限制：相对误差没有进入核心控制决策

当前1155维主干输入由当前105维与10帧历史组成。实际反馈有角速度、roll/pitch、关节角、部分关节速度、last action；缺少直接的实际root height、root linear velocity、actual global yaw和足部接触。79×74历史依然只含这些selected state与动作，没有参考或额外全局位姿。

不能据此说策略完全不能从关节/历史推断高度或支撑；这些信息可以部分隐式推断。但不同全局Yaw、瞬时线速度/高度与接触状态，可能有相同或近似相同的可用输入。不能依赖有限过去角速度历史辨认任意长期朝向偏差。

目前heading300增加的两维sin/cos误差只进入[`heading23500_actor.py`](../tools/heading23500_actor.py)的单个线性头：

```text
action = core(base, z) + 0.2 tanh(Wz z + Ws sin(e_yaw) + Wc [1−cos(e_yaw)] + b)
```

**核心Adapter没有接收实际heading误差。**改变context时，扣除heading项后的core差仅1.2e-7/2.4e-7，是浮点减法误差；结构上core不依赖这两维。头没有直接输入当前参考姿态、目标关节速度、当前接触，虽然z能间接携带实际运动阶段信息。

完整轨迹上heading头preactivation最大绝对值只有0.0092，tanh基本在线性区域。头动作RMS约3.48e-4至4.01e-4；对Yaw误差的动作导数最大约9.64e-4 action/rad。因此当前误差反馈主要是小幅、近似线性的加性修正，缺少明显的“同一Yaw误差在不同目标/支撑下采用不同纠偏动作”的直接条件化。

**实测支持的判断：**它能改善77名义WM轨迹，却没有在本次±10°扰动上形成可靠长期纠偏。静态导数小本身不能证明反馈无效或增益应加大：既有head-off实验也说明小修正确实影响轨迹。直接放大这条旁路可能进一步激发接触/历史放大。

建议把相对朝向、姿态/速度跟踪误差、目标运动阶段和可信支撑估计输入到一个有非线性交互的跟踪控制模块/核心Adapter，并给critic同一组任务误差。保留旧策略输出的初始化一致性，但需要重新定义训练任务和恢复状态分布。不要再把“critic加两维失败”推断成“所有显式误差观测无用”；此前并未把这些误差送进核心actor。

实际root height/velocity与contact的新增信息必须有部署可获得的估计、噪声和延迟模型；不可把MuJoCo真值未经蒸馏直接当作实机actor输入。

## 4. 方法上的主要限制：借用了Adapter，却没有重建它依赖的基础任务

Any2Track先训练可用的AnyTracker，再做动力学适配；它使用目标关节位置/速度等目标信息，并以参考为中心预测PD残差。这与当前冻结TWIST、以固定nominal pose加统一0.5动作缩放、再在多层注入修正的体系有不同任务定义。相同卷积布局或“world model先、PPO后”的更新顺序不能证明方法完整等价。[Any2Track论文](https://arxiv.org/html/2509.13833v3)、[官方OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack)。

本地官方代码只读对照也确认：Adapter配置包含dif_joint_pos/dif_joint_vel、reference feet/root height；world state含root height和未屏蔽的joint velocity；执行目标是reference pose+action。相关位置为`/home/hank/OpenTrack/track_mj/envs/g1_tracking_adapter/train/g1_env_tracking_general_dr.py`与`g1_env_tracking_general.py`。本次没有写入该仓库。

TWIST原方法用有未来信息的teacher指导单帧student，采用RL+BC；它本就强调局部跟踪与实时响应，并明确分析缺未来参考造成的迟疑。当前把一个已有student长期冻结，再仅以原student/已有Adapter行为作保护，并不会自动恢复这种teacher提供的前瞻控制能力。[TWIST论文](https://arxiv.org/html/2505.02833v1)。

历史30000次训练的最优23500相对1250只有小幅收益，完整77上没有全面优于原TWIST；髋Yaw等偏差在早期4900便存在。后续强锚定这些已有动作，只能保持已有控制习惯，不能提供更正确的物理动作标签。它可以保护已掌握的能力，也可能限制修复继承偏差；后者需要独立实验，不能从anchor loss小直接认定。

更有针对性的路线是先训练少量专科teacher，分别覆盖深蹲/起身、行走/转向和稳定保持。它们必须解决本次统一环境中的真实任务误差和扰动恢复，再通过student访问状态上的DAgger/RL+BC聚合，而不是只在teacher的理想轨迹做BC。已有良好TWIST行为可作保护参考；错误状态不能仅把旧动作当最优监督。

参考中心动作空间`q_target=q_ref+scale_per_joint*residual`是值得独立比较的方向，但不能把它直接套到旧checkpoint上。TWIST已编码的PD补偿需要保留或蒸馏，新空间需要对应训练。它的意义是把精确姿态、预载/平衡补偿与动力学适配分工得更清楚，不是强迫PD目标等于参考角度。

## 5. Motion-WM未解决物理可执行性，也没有把未来信息送给actor

Motion-WM是参考序列GRU：25×31→11×31，只依赖外部参考历史；不是包含机器人状态、接触和动作的物理动力学模型。runtime第307–308行只使用`prediction[0,0]`，训练reference pipeline也只使用当前预测。未来十帧被用于离线训练，却没有被低层actor消费。

本次77中，WM输出相对clean输入产生joint RMSE=0.03613rad、height RMSE=0.01137m、Yaw RMSE=2.3876°。两个策略间该参考仍逐帧相同，因而不能说机器人反馈把WM目标带偏。但processed参考相对raw的改变，确实会把闭环引入另一条接触路径。

既有`motion_world_model/PHASE1_RESULTS.md`也区分了污染输入收益与单独clean输入效果：full-corruption离线恢复提升，不代表clean当前帧、每种污染或机器人闭环都改善。当前续训配置关闭artificial_corruption，raw与WM各半；不能将污染分布上的MSE收益直接转成该配置的物理收益。

建议先用clean/raw把基础闭环纠偏学好，再单独评估参考修复器。若希望WM提供前瞻能力，应让新teacher/跟踪模块实际消费预测未来的紧凑特征，按预测误差和时序一致性训练；未知未来不能在部署用真实未来代替。对修复参考新增FK/脚接触、关节速度、角速度及高度的一致性审计，而不只是31维平均MSE。是否存在77的物理不可行段，本次没有证明。[Retargeting Matters](https://arxiv.org/abs/2510.02252)说明参考重定向伪影值得单独处理；这不是本仓库所有误差的因果证据。

## 6. 接触几何与时间契约：优化的是另一套闭环

训练manifest指向PhysX、2ms×10、23关节URDF；评估使用MuJoCo、1ms×20、25关节XML。PD、动作缩放、主要armature一致，并不意味着接触动力学一致。

训练足底URDF每脚两根横向圆柱，radius=0.02m、length=0.15m；评估XML每脚四个radius=0.005m的球。接触位置、支持区域、底面高度和载荷分配不同。新增脚型实验只替换这些几何，保留其余模型；编译后mass、inertia、body poses、joint axes/ranges、armature、damping与actuator参数全部逐元素相同。

脚型替换使WM速度RMSE从0.10386增至0.17339m/s，约+66.9%；因此不能推荐仅把仿真XML换成圆柱就算修复。步长改为2ms时raw Yaw显著改善，WM Yaw却变差，关节误差与振荡同时增加。也不能以raw Yaw变小选择更有利的模拟器设置。

建议先确定一套目标机器人资产和接触/执行规范，固定它用于下一轮主要训练与评估。MuJoCo是当前效果问题的实际验收环境，优先建立同资产的MuJoCo训练/专科teacher，降低跨引擎因素；保留PhysX作为独立泛化评估，或在训练显式加入可测的接触形状/摩擦/延迟变化。迁移训练框架是工程方案，不是已证明一定提升。

实机链路还含TargetSafetyFilter、动作渐入、reference age以及零位差异。这些要进入执行契约，本次CPU仿真未验证实机效果；它们也不能解释已经在纯同步仿真复现的全部退化。

## 7. 为什么大量小改动没能积累成进步

1. **保护局部函数，没有保护恢复动力学。**10月13日既有hard26数据表明，原策略和新策略实际访问状态上，完整动作偏差均≤8.03e-5，WM全长Yaw仍为77.93°。这已经反驳“只要池足够大、动作足够接近，闭环就稳定”。
2. **接触系统放大路径变化。**把物理状态、控制器缓存和历史看成一个闭环状态，小动作差既可改变接触载荷，也会影响后续状态/动作历史；策略会在不同历史上继续产生不同动作。函数值偏差约束没有限制反馈导数或接触切换后的恢复能力。本次未直接估计各项增益。
3. **奖励看局部平均，验收看长时累积。**当前rollout 24×20ms=0.48秒，gamma=.99、lambda=.95，GAE直接残差权重的e-fold时间约0.326秒。13.4秒后的奖励折扣约0.00119。价值函数仍可bootstrap远期收益，不能说PPO只看0.48秒；但长时转向残差与稳定恢复的信用分配，比即时姿态/速度收益困难得多。单独加长rollout或提高gamma不是已验证修复。
4. **多个新模块不等于有更正确的监督。**DTERA的uncertainty衡量模型预测分歧，risk通常监督约0.2秒内的失败；均不等于“这个残差会减少64秒累计Yaw误差”。已有4800多seed报告也未显示其稳定优于base。继续增加门控不能代替学习正确的恢复动作。
5. **反复在77选优会产生开发集偏倚。**完整77很适合定位问题，但大量checkpoint/超参数在它上面筛选后，名义32°仍只是开发结果。本次±10°、脚型与dt测试说明需要独立长动作与扰动检验；不能用bootstrap同一条轨迹代替独立seed/场景。

原`episode_length_s=10`不能作为“训练最长只有10秒”的证据：headless `HumanoidMimic._init_buffers`会用最大motion长度覆盖这一值，本次已核查。训练高heading/转向也已有覆盖；不能再声称关键状态完全没有采到。

## 8. 修改顺序与可验证的交付条件

|优先级|修改|目的|进入下一步的证据|
|---|---|---|---|
|P0|为WM建立独立真实state target和执行action/PD target通道；恢复Adapter专用踝速度|消除已确认的目标冲突与动作语义混用|真实延迟/投影回放上q、dq、action、joint order、reset/terminal逐项一致；WM留出数据上逐关节/接触阶段预测有效|
|P1|统一目标资产/控制/接触规范，建立同环境训练与评估|让梯度优化实际验收的动力学|旧模型固定回放可复现；新teacher在目标资产完成同一raw任务和扰动恢复|
|P2|以显式相对误差、目标/支撑条件化训练专科teacher，再蒸馏给student|获得比旧student更好的动作监督和纠偏能力|raw和WM都改善实际姿态/速度/Yaw，扰动后误差回落，多动作没有交换损害|
|P3|独立比较参考中心动作空间与未来特征输入|减少整个姿态重建负担，使用真正的前瞻信息|分别单因素对照；零干预初始化保持；未知未来用预测而非oracle|
|P4|在P0–P3基础上加局部策略敏感性约束与轨迹接受/回退|限制后续改进破坏已有能力|有非零且可接受的学习；不能只报告拒绝率、低KL或低MAE|

P0包含不同接口缺陷，应先分别验证，再按同一物理契约整合。它们修复模型正确性，不承诺单独恢复77。P1与P2才是我认为最有希望解除当前性能平台的方法方向。

**下一轮最小实验建议：**先固定目标MuJoCo资产，围绕一个转向→停止/稳定支撑任务建立专科teacher；用部署可用的relative heading、yaw-rate error、未屏蔽关节速度和目标姿态作为核心条件，以实际机器人偏离后的恢复曲线作为任务指标。扰动采用预声明的正/负方向和时刻，不广泛扫参数。对照旧heading300的相同任务，并保持其他主要条件。若这一任务都无法形成恢复能力，先不扩到全动作大规模训练。

后续再分别加入深蹲协调teacher、其他参考和动力学变化。对没有可靠实机估计的高度/线速度，先给teacher特权信息并做带估计噪声的student蒸馏；actor不能依赖无法部署的真值。已有随机Yaw重置分支在随后赋值`root_quat`时会被覆盖，不能只打开flag便认为已经做了相对heading扰动；要实际旋转参考reset姿态，并核对消费误差。

对抖动，可参考对策略输入敏感性的Lipschitz/梯度约束，优先验证接触附近的实际高频与恢复能力；约束本身也不保证整个混合接触闭环稳定。已有EMA带来延迟与速度代价，不能只以输出更平滑选优。[Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies](https://arxiv.org/abs/2410.11825)。

显式anchor相对位姿误差与目标/状态联合输入有相关先例；借鉴的是观测与控制任务的定义，不是要求采用完整diffusion系统。[BeyondMimic](https://arxiv.org/html/2508.08241v1)。

完整双模式轨迹保护门建议保留，但它的作用是避免污染后续优化起点，不是解决根因。所有候选都拒绝时，必须报告没有有效学习，不能把“没有退化”当作方法成功。

## 9. 哪些方向目前缺乏依据

既有实测不支持把77的问题主要归于腿/腰电机饱和：深蹲全部20个子步中腿/腰未饱和。参考本身有躯干前倾，机器人平均更直立，不能添加全局Pitch→0损失。膝/髋目标没有被投影裁掉，取消投影对膝问题缺乏直接依据且Yaw退化。

现有历史配对/JIT/WM参考对齐已核验；本次null也精确。不能再以“可能错帧”解释全部问题。已确认的WM动作监督语义问题，与历史本身是否按(s_t,a_t)正确配对是两件事。

仍待验证：每个接口缺陷分别对继承encoder和最终策略的贡献；实际参考可执行性；物理状态与缓存历史分别贡献多少；哪些关节/接触是主要扰动通道；新teacher、新动作空间或新训练引擎是否带来多动作收益。本文没有把这些写成已经证明的根因。

## 10. 产物、复现与验证

所有新增产物位于[`reports/architecture_diagnosis_20261013`](../reports/architecture_diagnosis_20261013)：

- `summary.json`、`comparison.csv`、`phase_metrics.csv`：10次全长仿真及重新计算指标。
- `probes/static_audit.json`：输入、heading反馈、踝速度、WM参考变化与source hash。
- `transition_contract.json`：实际WM训练函数的指令/执行动作探针及单踝oracle。
- `probes/simulation_results.json`：6次初始朝向实验及两个精确null。
- `model_contract_retry/results.json`：4次脚型/dt单因素实验；独立XML保存在该目录。
- `validation.json`：资产不变量、完整帧数/子步数量、28项通过的回归测试。
- `initial_yaw.png`、`model_contract.png`及每次`frames.jsonl`/metrics/log：可检查的曲线与原数据。

28项回归测试：仿真评估系统15、Adapter runtime 3、Motion-WM runtime 4、training reference 6。新增探针实际调用原函数并验证数值断言；10次仿真指标从trace重新计算，与在线汇总误差<1e-12。2ms例明确检查每控制周期10个子步，其他为20。代码编译和空白检查在交付收据中核查。

第一次脚型脚本因相对路径处理错误在任何仿真前停止，原`model_contract.log`与初始XML保留；有效结果仅取`model_contract_retry`四个完整运行。此失败不计入10次物理实验。

CPU环境可复现初始朝向实验；输出须新目录：

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hank/anaconda3/envs/twist/bin/python \
  reports/architecture_diagnosis_20261013/run_probes.py \
  --stage all --output reports/architecture_diagnosis_20261013/repeat_new
```

`run_model_contract.py`对已有输出拒绝覆盖；再次运行需在独立副本中指定新输出位置。新仿真不是训练/实机验收；本次没有GPU训练，因为本沙盒内CUDA不可用，但所有计划内CPU诊断均完成。
