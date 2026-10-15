# heading300第一次Actor更新精确诊断（2026-10-12）

本阶段仅诊断：未启动训练，未修改模型原权重、网络、奖励、critic、优化器或超参数，未做实机测试。当前最优参考仍为heading300。

## 六个必须回答的问题

1. **model25保持heading300吗？保持。** 除critic八个张量因warmup更新外，所有Actor/冻结模块权重逐元素相同；完整77 raw/WM的关节、qpos/qvel、Yaw、策略动作、PD目标和7003维输入全程逐元素相同。不是仅凭随机输入或JIT成功推断。
2. **model26修改了什么？** 第一次Actor更新包含48个minibatch step，改变核心Adapter五层的10个张量、heading头2个张量及critic8个张量；其余状态完全不变，std也未变。核心和heading都未被单独冻结。逐层变化量见`verification/verification.json`。
3. **raw/WM为何不同？** 相同输入上，两种模式的直接动作变化RMS都约2.45e-4，并非WM直接输出漂移显著更大；两种模式都产生接触/状态/历史轨迹分离。raw行走/转向误差有所改善而下蹲更差，综合RMSE略降；WM多阶段误差恶化。raw是得失抵消，不能称没有退化。
4. **存在机器人状态→WM参考的放大吗？没有这条路径的证据，而且当前实现没有该反馈输入。** 高层Motion-WM只吃clean参考的25帧历史；所有模型的处理后参考、原始历史及归一化输入全程相同。存在策略—机器人—状态/动作历史的闭环放大；应区别于Motion-WM参考反馈。AnyAdapter的冻结历史编码器参与actor，训练用world_model decoder不在部署actor forward中。
5. **锚定遗漏了敏感状态吗？不能认定完全遗漏。** 历史WM rollout中高heading误差、转向及接触切换组合已有样本，已有anchor池中也存在明显pointwise动作漂移；证据更支持“平均/软尾部损失未提供逐状态功能保护”，而非简单缺少WM或大heading样本。具体未保存的接触标签、历史分组PPO梯度不能补造。
6. **下一轮唯一修改？** 优先验证在现有固定完整教师anchor输入池上，对每个Actor minibatch候选步施加逐状态、逐关节最大动作偏差约束，并联合回退核心Adapter与heading头的参数位移。保留现有软锚定，其系数和所有其他条件不变。不先加采样、不先改critic、不冻结上下肢。该方案只是待验证假设，本阶段未实施训练。

## 1. 一致性、条件和只读记录

- 模型：Adam恢复分支`model_25.pt`、`model_26.pt`；源`heading23500_gain05_pilot100_20261008/model_300.pt`。源hash及逐张量差异保存在verification。
- 随机128个7003维输入：model25/source误差0；三个模型JIT/eager误差均≤1e-6。六条实际轨迹的同batch JIT/eager最大误差为0；离线batch256与在线batch1最大差异≤3.1e-06，属于浮点batch差异，不能把它当更新漂移。
- 全部使用CPU、seed42、完整77（64.14s/3207帧）、raw/WM、同步ready/frame协议、动作EMA0、Adapter增益0.5、PD步长1ms、20倍decimation、投影margin0.02/knee_min0。历史23500本阶段不加入，避免无投影条件干扰。
- 新记录通过显式`--first-update-diagnostics`启用。只复制已消费的Actor输入到npy侧文件，以及读取WM历史；不额外更新任何历史、不抽随机数、不影响策略动作。启用记录后的heading300轨迹与上一轮未启用记录的基线逐元素相同。
- 验证全部79帧AnyAdapter状态历史与过去state/action配对，以及10帧TWIST观测历史；错位误差均0。初始历史填充属于既有初始化语义，不被当作真实负时间帧。WM第24帧ready，第0–23帧回退原参考，25帧窗口每次重新运行GRU，无跨窗口携带hidden state。
- Motion-WM实际checkpoint：`/home/hank/TWIST（anyadapter）/legged_gym/logs/motion_world_model/full_stable_v2/best.pt`，SHA256 `187c11b128cbf1a6c095edcf1e3e8291fef76b63c03639f1d5618285f83afd99`；高层路径及31维归一化输入保存在analysis。

## 2. 完整77量化结果

|模型|模式|关节RMSE(rad)|根速度RMSE(m/s)|Yaw RMSE(°)|
|---|---|---:|---:|---:|
|heading_300|raw|0.138451357|0.109048942|67.777002|
|heading_300|wm|0.139305507|0.103864178|32.080325|
|model25|raw|0.138451357|0.109048942|67.777002|
|model25|wm|0.139305507|0.103864178|32.080325|
|model26|raw|0.137909181|0.110908009|66.211415|
|model26|wm|0.139745312|0.102400613|51.853080|

三模型均无跌倒代理触发。model26 raw关节/行走Yaw可改善，同时下蹲Yaw恶化；WM速度略改善但Yaw明显恶化，不能宣布整体更优。

|阶段|raw model25→26 Yaw(°)|WM model25→26 Yaw(°)|
|---|---:|---:|
|descent|74.757 → 79.945|16.121 → 52.781|
|squat|72.740 → 78.053|20.980 → 59.047|
|rise|68.244 → 73.210|22.941 → 60.206|
|walking|72.581 → 68.503|36.935 → 52.514|
|turning|75.003 → 70.941|38.770 → 56.383|

阶段沿用已有phase_masks，可能重叠。后半程下蹲朝向差异可能继承此前转向产生的累计偏航，不能将下蹲阶段高误差直接认定为下蹲触发。

## 3. 分离时间及同输入分解

Yaw分离阈值1/5/10°要求连续≥10帧（0.2s）；这只是透明声明的诊断事件阈值，不替换原Yaw RMSE定义。

|事件|raw|WM|
|---|---:|---:|
|policy_direct_gt1e_6|0.00s|0.00s|
|actual_joint_gt1e_6|0.02s|0.02s|
|contact_first_mismatch|7.86s|8.62s|
|yaw_gap_gt1deg|12.80s|18.52s|
|actual_joint_gt0_01rad|16.04s|20.20s|
|yaw_gap_gt5deg|20.72s|23.50s|
|yaw_gap_gt10deg|20.86s|24.34s|

WM第一次接触状态不同发生在8.62s下蹲下降附近，持续1°Yaw分离在18.52s转向，5°在23.50s行走/转向。接触先分离是时序证据，不是单独接触事件的因果证明。raw的分离更早，因此不能说WM一定具有更大的放大增益。

严格区分：直接效应=`π26(o25)−π25(o25)`；状态/历史部分=`π26(o26)−π26(o25)`。二者相加等于闭环动作差。后者只是固定策略的函数分解，不是控制单一关节/接触的因果效应。

|模式|同输入动作RMS|同输入core RMS|同输入heading RMS|状态/历史部分RMS|
|---|---:|---:|---:|---:|
|raw|0.0002471|0.0002376|0.0000273|0.1157396|
|wm|0.0002446|0.0002373|0.0000264|0.1010714|

0秒输入、机器人状态、参考完全一致，最大动作差0.003724，最大项左肩Pitch；腿部还有右踝Pitch约−0.001216、右髋Yaw约−0.001086的动作差。它们不是足以单独归因的证据。23.50s同输入最大差仅0.000444，固定策略状态/历史部分最大差已约0.3211；不能把此时闭环大动作差归因于权重直接输出漂移。

## 4. 受控短分叉：复制完整前缀，不作qpos-only恢复

复用现有RealTimePolicyController。固定记录下的同一WM参考，四组先完整重放0–18s的model25，再运行18–26s的8s分叉。所有前缀qpos/qvel、关节、Yaw、动作、PD逐元素相同；null分支整个26s也与原轨迹完全相同，因此继承了同一物理/控制历史而不是假设求解器warmstart可忽略。模块替换仅用于独立诊断策略，原文件未覆盖。

|18s后策略|该8s Yaw RMSE(°)|相对null最大Yaw分离(°)|
|---|---:|---:|
|null|12.549|0.000|
|full26|10.597|38.944|
|core26|9.830|24.073|
|head26|12.818|4.410|

核心Adapter变化主导这一受控局部的Yaw轨迹分离；heading变化也有作用，两者组合非线性。但full26/core26在此局部RMSE反而较null低，说明从0秒开始的整段退化与进入转向前的状态/历史有关，不能把这一局部结论推广为“新core始终有害”或“只冻结heading即可解决”。

## 5. 锚定覆盖及梯度证据

- 历史真实第一次更新rollout含12288帧，其中WM 5994帧。WM heading≥30°占54.95%；转向×单支撑×接触切换附近占6.57%。不能认定这些类别完全没采到。
- 历史model25 anchor bank有19200条输入，六个raw/WM×阶段池各3200，原sample对非空池均匀采样。它保存了真实base、冻结latent、heading context、完整教师目标和权重；没有保存真实接触标签和时间顺序，不能补出接触切换统计。各池model26的pointwise动作漂移最大约0.0041–0.0046，而RMS仅约0.00026–0.00028：已经进入anchor池的状态也允许产生尾部偏差。
- 历史首个minibatch的PPO/core梯度范数7.221813，完整锚定/core 0.000407908，比值0.00565%；heading比值0.00351%。锚定包括原on-policy、replay和tail，数值近零而非严格零。理想数学上，策略=教师时二次偏差损失的一阶梯度为零；浮点batch差异产生少量数值梯度。
- 新独立固定策略审计：512环境、8×24步、98304帧、源配置/动作分布不变，optimizer_steps=0；原生GAE和timeout bootstrap。记录真实接触、clean阶段及动作概率，按raw/WM、heading、转向、支撑、切换、深蹲/行走分组。它不是历史rollout的重建。
- 新审计WM全组model26条件梯度中on-policy+tail anchor/PPO比值core约10.80%、head约9.03%；转向×单支撑×切换组core约20.81%、head约18.10%，该组权重质量占比还高于样本占比。没有发现简单地把此组权重压低的证据。条件梯度最多2048个确定性均匀抽样状态，稀疏组不作因果结论。
- 历史未保存完整Actor observation、采样action、优势对应的minibatch indices，无法准确重建历史每个分组的PPO梯度。已明确输出缺失项，只有保存的首minibatch全局梯度属于历史实测。独立审计的分组梯度仅含on-policy+tail；replay缺少支撑标签，不能声称重建了按支撑分组的完整replay梯度。

## 6. 下一轮唯一最小方案（尚未训练）

保留现有完整教师、现有anchor输入池和所有训练设置，只新增Actor候选步的逐状态功能偏差保护。固定使用model25已有19200条anchor输入，确保原核心Adapter+heading教师目标正确；检查所有23个动作，不固定任何关节。候选Actor位移对core/head联合回退，避免单模块破坏已有补偿关系。
- 首轮只到26次更新（25次critic warmup＋1次Actor更新），保存0/25/26，分别完整77 raw/WM评估。
- 初始工程假设：池内最大完整教师动作偏差≤1e-4（动作单位，约为实测池内尾部的1/40）。超过阈值则对候选Actor参数位移回退；Adam公式/LR、critic更新、reward、sampler、WM、PD、projection、EMA均不改。checkpoint必须记录实际接受位移及优化器状态，不能把被拒绝步伪装成成功学习。
- 1e-4尚未证明足以保护闭环，也可能导致学习停滞。只检验这一约束，不同时扩充数据池或改采样。保护门失败就停止；通过保护门仍不等于下蹲/抖动10%改善，更不等于多动作验收。
- 选择约束而不是先改采样的理由：高heading/接触转向已有训练覆盖，同输入漂移并非只发生在WM；已有池上的尾部偏差与近零首次锚定梯度是更直接证据。若后续固定池保护通过但闭环仍退化，再单独检查池外动态覆盖。

配置详见`next_minimal_config.json`，只是下一轮方案，没有启动训练。

## 7. 产物与复现

- 统一评估：`evaluation/manifest.json`、`summary.json`、CSV、report.md、六条完整trace及`frames.jsonl.actor_inputs.npy`。
- 精确权重和随机JIT检查：`verification/verification.json`；实际JIT：`analysis/actual_jit_consistency.json`。
- 逐帧：`analysis/raw_frames.csv`、`wm_frames.csv`；关键状态：`analysis/key_states.csv`；同输入及闭环分组：`analysis/simulation_groups.csv`。
- 组件：`analysis/*_components.npz`（base/core/adapter/head/full/latent）；反事实：`*_counterfactual.npz`；WM归一化输入：`*_wm_inputs.npz`。
- 曲线：`analysis/plots/{raw,wm}_timeline.png`、`{raw,wm}_first_divergence_joints.png`、`controlled_fork.png`，以及原评估的全部23关节、PD、COM和频谱图。
- 历史coverage/anchor-bank与独立分组梯度：`coverage/*.csv`、`results.json`；独立审计输入：`readonly_audit/audit_states.pt`及receipt。
- 受控分叉：`fork/results.json`及四条逐帧trace；源checkpoint保留。

实际评估命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/hank/anaconda3/envs/twist/bin/python tools/evaluate_sim.py \
  --checkpoint model25=legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_25.pt \
  --checkpoint model26=legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_26.pt \
  --mode standard --cases mocap1__77 --workers 1 \
  --first-update-diagnostics --no-historical-control \
  --output reports/heading300_first_actor_update_20261012/evaluation
```

输出已存在，复现必须使用新的独立目录，不能覆盖。

本轮未检验其他动作、多个训练seed、实机，以及其他分叉时刻/更长干预的泛化。没有将数值一致、局部分叉改善或统计关联宣称为整体优化成功。

## 补充：根因证据排序和接触/执行目标

第一层（已测）：Actor首次更新的核心Adapter功能变化，通过机器人—状态/动作历史闭环形成路径依赖；不是外部Motion-WM参考被机器人反馈改写。冻结模块/JIT/同步异常已由完整轨迹对照排除。局部分叉支持core的影响大于heading，但并不证明每个关节或每个转向状态都变差。
第二层（保护不足的直接证据）：完整教师均值/软尾部锚定在首minibatch几乎没有一阶抵消，且已有anchor池状态仍允许约0.004的逐点尾部漂移。该证据支持下一轮检验功能约束，尚不证明约束一定改善Yaw或下蹲。
第三层（待验证）：具体足部碰撞时刻、摩擦/支撑切换、上肢反作用及历史反馈分别贡献多少。不能只按闭环动作差、参数L2、力矩饱和或contact标签关联直接归因。
在0秒，最大动作差对应左肩Pitch，投影后的真实PD目标仍差0.001862rad；右髋Yaw和右踝Pitch的执行目标分别差约−0.000543/−0.000608rad。它们确实进入执行目标，并非全被投影抹掉，但没有单关节干预证明哪一个是唯一触发源。
WM在2.00s已经出现>50N的足部力瞬时差异；8.62s第一次支撑标签不同：左足垂直力model25约1.09N、model26约191.73N，接触点4→7，足部位置最大差仅约1.56mm。因此这次差异不仅是跨越5N统计阈值，也包含真实接触载荷变化。尚不能据此把第一次50N差异定义为唯一原因。
raw也在1.96s出现>50N足部力差异，进一步反驳“raw没有同等闭环敏感性”的简单解释。当前只能说净Yaw误差的方向不同，不能用一个模式的平均动作MAE预测另一个模式的结果。
新增执行/力曲线：`analysis/plots/{raw,wm}_contact_and_targets.png`；实际Yaw及参考曲线：`{raw,wm}_yaw_and_reference.png`；关键力、投影和PD差异：`analysis/contact_key_states.json`。
回归检查：原仿真自动评估系统15项单元测试通过；新增六条完整轨迹、四条同前缀分叉及只读审计均完成。未启动任何新训练。
