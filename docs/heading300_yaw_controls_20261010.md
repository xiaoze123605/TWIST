# heading300续训Yaw退化：四组合与原分布控制（2026-10-10）

结论：四组合无训练消融已经完成，核心与heading头在结构上可独立替换，但闭环性能有显著交互。原始采样512环境控制在第30次更新（25次critic预热、5次策略更新）也出现WM Yaw明显退化，因此停止，未运行40/50。保留heading300为主要性能基线。下一轮唯一优先方向是独立增强现有完整教师动作锚定，不冻结头、不同时改学习率或采样。本阶段没有启动该修改方案。

## 数据、控制条件及独立保存

- heading300训练源：`legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt`。
- 加权采样model50：`legged_gym/logs/recovery_heading300_sampling/sample_only_alpha014_p1_env512_seed42_20261009/model_50.pt`。
- 四组合、完整仿真报告及分析：`reports/heading300_yaw_controls_20261010/`。
- 原分布控制：`legged_gym/logs/recovery_heading300_yaw_controls/original_env512_seed42_20261010/`。

全部对照保留TWIST主干、历史编码器、WM、7003维输入/23维输出、Adapter增益0.5、PD、投影margin0.02/knee_min0、动作EMA0及seed42。完整77为64.14秒、3207个记录帧，raw与WM分别比较。相位与64点Welch 5–20Hz沿用已有计算。所有指标仍以clean参考计算；Yaw为wrapped角度RMSE。

原分布控制与上次加权实验的环境配置逐字段相同，唯一差别是采样YAML。训练配置除类名、保存节奏外相同：512环境、24步rollout、16 minibatches、3 epochs、完整冻结教师、actor LR 5e-7→2e-7/2000更新、heading LR 5e-6、critic LR 1e-4、25次critic warmup、锚定0.08→0.04/500更新、固定scale0.087、tail系数0.01、回放容量/采样及固定探索不变。新增诊断不消耗随机数，不修改梯度；3项测试确认读取梯度及autograd.grad探针不改变原backward。

原23500依然由评估系统作为无投影历史条件单独保存，不参与本阶段因果对照或heading300保护判断。没有修改奖励、网络、WM、PD、实机代码或既有训练入口。

## 一、无训练四组合消融

结构检查：核心为`layerwise_actor.adapter.0`至`.4`，分别添加到五个冻结TWIST层；heading为`.5`，130→23的独立Linear，输入是冻结历史latent128与heading上下文2。heading不直接读取核心Adapter中间特征。两checkpoint的共享TWIST/历史编码器/WM/normalizer/std张量逐项相等、键和形状一致；critic差异不参与部署动作计算。

构造四组时仅替换对应weight/bias。输入batch=1/2/17/64，显式“选定核心输出＋选定heading输出”与组合forward最大误差0，JIT误差0；A/D另与各自原始JIT比较最大差0。原权重未覆盖。组合的`.pth`标注diagnostic_only，没有恢复优化器；不得拿组合诊断状态直接续训。

|组合|核心/heading|raw Yaw RMSE，°|WM Yaw RMSE，°|raw 膝RMSE，°|WM 膝RMSE，°|raw 行走HF，rad|WM 行走HF，rad|跌倒|
|---|---|---:|---:|---:|---:|---:|---:|---|
|A|旧/旧|67.777|32.080|17.371|16.416|0.0103384|0.0091667|无|
|B|新/旧|78.674|49.808|17.386|16.419|0.0099958|0.0097037|无|
|C|旧/新|80.957|74.471|17.372|16.013|0.0108068|0.0096917|无|
|D|新/新|81.936|41.754|17.381|16.449|0.0102394|0.0106533|无|

A与heading300逐项复现一致。A是未改变基线，不因“没有改善”就判为策略失效；B/C/D均未通过Yaw保护。

固定这两个checkpoint、相同仿真条件下，WM中核心替换在旧头背景使Yaw增加17.727°（B−A），在新头背景反而减少32.717°（D−C）；头替换在旧核心背景增加42.391°（C−A），在新核心背景减少8.053°（D−B）。RMSE交互对比D−B−C+A为−50.444°，raw为−9.918°。这是该checkpoint对、该动作的非线性交互，不是普适的模块责任比例。

可以确定：两个模块变化都能在受控替换下破坏旧策略Yaw；新头在新核心背景下有补偿作用。**不能直接推荐冻结旧头**，B的WM Yaw49.808°比D41.754°更差；也不能仅因C最差就宣称heading头是唯一根因。结构可替换与闭环组合稳定是不同检查，后者必须实测。

## 二、原始采样同批量控制与死机恢复

从heading300重新初始化独立原分布控制，模型0与教师/源权重精确一致；7003→23、critic1318、gain0.5确认。单独3次critic-only冒烟通过，策略权重保持不变。正式控制原计划最多50次，30/40/50暂停评估后才继续。

2026-10-10电脑意外死机发生在model30完成保存、其自动评估刚启动之后。检查结果：model0/10/20/30均可完整加载，iter与recovery count一致；没有有效原评估结果。训练代码及manifest哈希完整，但部分日志尾部为零字节；critic归档1/25/26有效，30归档为0字节，目标梯度26/27记录有效，其他完整训练/梯度记录止于26。不能根据这些丢失文件声称27–30的KL/损失/梯度具体值，也不把死机推断为训练导致的数值异常。

在新目录`original30_after_crash`补做model30完整77，重新导出JIT，eager/JIT误差0，heading300对照与此前严格一致。没有恢复训练、重置环境后追加更新或覆盖中断文件。结果触发保护，记录STOPPED.json，正式结束控制；40/50未运行。受损原文件保留，完整JSON行另存`recovered_diagnostics`；评估未完成不能视为通过。

|分支|总更新/策略更新|模式|关节RMSE，rad|根速度RMSE，m/s|Yaw RMSE，°|膝RMSE，°|行走HF，rad|跌倒|
|---|---|---|---:|---:|---:|---:|---:|---|
|heading300|0/0|raw|0.138451|0.109049|67.777|17.371|0.0103384|无|
|原分布控制|30/5|raw|0.137289|0.104983|64.737|17.302|0.0101076|无|
|加权采样|50/25|raw|0.139032|0.105678|81.936|17.381|0.0102394|无|
|heading300|0/0|WM|0.139306|0.103864|32.080|16.416|0.0091667|无|
|原分布控制|30/5|WM|0.139279|0.109598|60.504|15.937|0.0098460|无|
|加权采样|50/25|WM|0.139846|0.105355|41.754|16.449|0.0106533|无|

原分布控制：raw Yaw−4.485%、速度−3.729%、膝−0.397%、HF−2.232%；WM Yaw+88.602%、速度+5.520%、膝−2.920%、HF+7.410%。膝与协调有局部改善，却不能抵消Yaw和WM速度保护失败。

两条训练分支共享512环境等条件，但原分布只完成5次策略更新、加权完成25次，**没有原分布50的同更新剂量终点**。只能确定原分布也出现退化，增加蹲起权重不是退化出现的必要条件；不能据60.5°与41.8°直接宣布加权采样更好。没有4096环境控制，无法单独辨认批量变小、优化器重新初始化、critic适配与数据分布的影响。

## 三、机制证据

### 分动作阶段Yaw（WM）

沿用原相位掩码，增加同一wrapped Yaw RMSE分组，不更换已有主指标。

|策略|深蹲|下降|起身|行走|转向|
|---|---:|---:|---:|---:|---:|
|A heading300|20.980°|16.121°|22.941°|36.935°|38.770°|
|B 新核心/旧头|51.836°|45.699°|53.226°|52.284°|55.574°|
|C 旧核心/新头|74.344°|76.219°|70.376°|78.443°|82.093°|
|D 加权50|43.477°|37.623°|45.073°|43.521°|47.233°|
|原分布30|61.151°|63.357°|57.634°|65.349°|67.937°|

退化分布在多个阶段，不能解释成单个蹲起帧或行走接触点的偶发误差。raw阶段数据在phase_yaw.csv，不与WM混为一类。

### 同一组状态上的动作漂移

取两分支回放中raw/WM×三运动组的共同探针集6144个状态，使用相同base/冻结latent/context分别计算各策略，避免用不同状态分布直接比漂移。此处是静态动作探针，不是闭环因果替换。

|策略|核心漂移RMS|heading漂移RMS|总动作漂移RMS|总动作MAE|下肢漂移RMS|
|---|---:|---:|---:|---:|---:|
|heading300|0|0|0|0|0|
|加权50|0.0015022|0.0001846|0.0015527|0.0011388|0.0011162|
|原分布30|0.0010986|0.0001131|0.0011425|0.0008406|0.0008726|

原heading输出RMS=0.0004421，两个续训头的漂移分别约42%与26%这一量级。原分布的平均漂移更小，WM Yaw反而更差，明确说明小平均动作误差不能代替闭环Yaw验证。核心与头均变化，不能用其中一个RMS较大就认定它是主要原因。

### 学习率、KL、梯度与锚定

预热1–25没有actor梯度；第26次开始策略学习，core/head/critic均参与既有全局0.5裁剪。有效记录第26次：actor LR约4.999e-7、heading5e-6、critic1e-4；最大rollout KL=0.0005075、无跳过minibatch，动作MAE0.0002718、anchor loss0.0009077、有效anchor系数0.078。第26次48个minibatch裁剪前全局梯度范数平均13.628、最大32.233，核心平均13.583、heading0.881、critic0.579。不能说本次actor学习被critic梯度主导。

第27次首个minibatch的目标梯度探针：核心PPO norm17.237、加权锚定norm0.3875（2.248%），heading PPO1.0538、锚定0.01567（1.487%）；梯度cosine分别−0.0162、−0.0843。锚定包含当前/回放完整动作与原tail项，无新损失。第26次首个minibatch锚定梯度接近0是初始化动作一致的正常现象，不是遗漏教师。

这说明在保留下来的首批证据中，锚定梯度量级远小于PPO，而低KL依然不能守住WM朝向。它支持优先验证锚定强度，但不是“提高系数必然解决”的因果证明。27–30完整梯度/训练统计因死机丢失，不能补造；加权50旧实验没有目标梯度探针，不能声称两采样分布梯度已严格成对比较。

### critic按heading误差分组

共同探针使用原控制的1/25/26归档：相同critic观测、实际奖励、done/timeout与最后观测，分别评估heading300、加权50、原分布30的critic。TD使用原算法的一步及timeout自举语义；GAE目标在同一归档内固定，但依赖归档时的critic自举，不是真实完整episode回报。

第26次探针中，WM的低heading≤30°组2674帧、高heading≥60°组2049帧：

|critic|低heading TD RMS|高heading TD RMS|低heading共同GAE残差RMS|高heading共同GAE残差RMS|
|---|---:|---:|---:|---:|
|heading300|0.323887|0.276666|0.601311|0.795250|
|加权50|0.296942|0.277616|0.614815|0.639276|
|原分布30|0.308177|0.291114|0.390623|0.466258|

高heading共同GAE残差通常更大，但TD并非一致随heading增大，且原分布30的critic相对共同GAE目标拟合更好，策略Yaw却严重退化。critic缺显式heading上下文的结构风险仍存在；本数据**不能证明它是主要原因**。分组受动作、速度、接触状态及回报结构混杂；没有在其余1318维输入相同而heading不同的干预下证明观测混叠，也不据此改变critic结构。

## 四、下一轮唯一优先修改：增强现有完整动作锚定

选择该方向，而不是立即冻结头、降低头LR或改采样：

1. 原分布也退化，先改采样不是针对本次共同失败机制的最直接验证。
2. B/C都明显退化，且新头对新核心有补偿作用；只保护一个模块可能打破已有补偿，当前没有证据支持单独冻结或减缓头一定有效。
3. 两条续训的低平均动作漂移/低KL都未守住Yaw，首批目标梯度显示现有软锚定较弱。先强化既有完整教师行为约束，是同时保护核心与头、且不改网络/奖励的最小独立测试。

**仅把anchor起止系数同步提高5倍：0.08/0.04→0.4/0.2。** 原tail0.01、scale0.087、回放、各学习率、所有环境/奖励/控制参数都保持。倍率5是首试选择，尚未验证；因tail保持原值，不能把目标总梯度声称恰好放大5倍，也不根据一个minibatch精确反算最优系数。

下一轮仍从heading300、原始采样、512环境、seed42起步；最多30更新（25预热＋5策略），保存25/30。先验收完整77 raw/WM的Yaw/速度≤105%、无新增跌倒，达到后才考虑继续评估动作覆盖。若再次明显退化即停止；本阶段未启动下面命令。

```bash
bash tools/run_heading300_original_control.sh anchor_x5_original_env512_seed42_next \
  --seed 42 --num_envs 512 --max_iterations 30 --save-every 5 \
  --anchor-coef-start 0.4 --anchor-coef 0.2

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hank/anaconda3/envs/twist/bin/python tools/evaluate_sim.py \
  --checkpoint anchor30=legged_gym/logs/recovery_heading300_yaw_controls/anchor_x5_original_env512_seed42_next/model_30.pt \
  --expected-update 30 --mode standard --cases mocap1__77 --workers 1 \
  --output reports/heading300_yaw_controls_20261010/anchor_x5_original30_next_eval
```

这里刻意将训练预算锁在30，之后手动调用已有评估入口；不使用当前按30/40/50自动继续的50预算入口，避免未验证就扩到40/50。该推荐不是已通过验收的修复。

## 验证、限制与交付

本阶段真正完成：四组合数值/共享状态/JIT验证；10组独立完整77无训练仿真（A与基线复用同一JIT/同条件结果，其余包含历史对照）；原分布控制30更新及恢复后6组完整77仿真；3项只读诊断单元测试；初始化、critic-only冒烟、配置同条件及源权重哈希检查。源heading300和model50均保留。

未完成或未识别：原分布40/50（因30保护失败而未运行）、同剂量50终点、4096环境批量对照、其他seed/其他动作本阶段验证、critic因果干预、死机丢失的27–30完整训练数据。没有从未完成评估得出通过结论，没有实机测试、没有250次长训。本阶段无法判断平台期。

新增工具：`heading300_cross_ablation.py`、`train_heading300_original_control.py`、`run_heading300_original_control.sh`、`heading300_training_diagnostics.py`、`analyze_heading300_yaw_controls.py`及对应诊断测试；既有物理仿真和诊断计算直接复用。

- [四组合完整报告与曲线](../reports/heading300_yaw_controls_20261010/cross_eval/report.md)
- [组合结构/数值验证](../reports/heading300_yaw_controls_20261010/cross_policies/validation.json)
- [死机后model30完整评估](../reports/heading300_yaw_controls_20261010/original30_after_crash/report.md)
- [四组合CSV](../reports/heading300_yaw_controls_20261010/mechanism_summary/cross_ablation.csv)
- [原分布控制CSV](../reports/heading300_yaw_controls_20261010/mechanism_summary/original_control.csv)
- [相位Yaw CSV](../reports/heading300_yaw_controls_20261010/mechanism_summary/phase_yaw.csv)
- [共同状态动作漂移](../reports/heading300_yaw_controls_20261010/mechanism_summary/common_state_action_drift.csv)
- [共同状态critic残差](../reports/heading300_yaw_controls_20261010/mechanism_summary/common_state_critic_residuals.csv)
- [完整机制分析JSON](../reports/heading300_yaw_controls_20261010/mechanism_summary/summary.json)
- [恢复凭证](../reports/heading300_yaw_controls_20261010/crash_recovery_receipt.json)
- [下一轮唯一配置（未启动）](../reports/heading300_yaw_controls_20261010/next_minimal_config.json)

无训练分析重现：`python tools/analyze_heading300_yaw_controls.py --output <新的分析目录>`。保持原来源路径和旧数据不变，输出目录已存在即拒绝覆盖。
