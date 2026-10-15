# heading300逐状态完整教师硬约束实验（2026-10-12）

**结论：硬约束严格生效且Actor进行了真实非零更新，但WM闭环保护失败，已停止实验；heading300仍为最优参考。不开展后续50次更新。**

本实验从原heading300训练checkpoint重新初始化，恢复同一Adam历史；只增加每个实际Actor minibatch更新的完整教师动作硬约束。完整77评估覆盖64.14秒、3207帧，raw/WM独立运行。训练与CPU仿真串行；未修改实机程序，未进行实机测试。

## 同条件结果

|模型|模式|Yaw RMSE (°)|Joint RMSE (rad)|Root velocity RMSE|跌倒代理|
|---|---|---:|---:|---:|---:|
|heading_300|raw|67.777|0.138451|0.109049|否|
|heading_300|wm|32.0803|0.139306|0.103864|否|
|unconstrained26|raw|66.2114|0.137909|0.110908|否|
|unconstrained26|wm|51.8531|0.139745|0.102401|否|
|hard26|raw|58.4354|0.13831|0.104185|否|
|hard26|wm|77.9312|0.139694|0.110316|否|

新model26 raw Yaw改善，但WM Yaw 77.931°超过33.68°，相对32.080°基线退化约142.9%，且比无约束model26的51.853°更差。WM根速度RMSE退化6.21%，超过5%上限。raw根速度改善4.46%；所有模型均无新增跌倒代理事件。不能据raw改善宣布整体更优。

## 蹲姿与行走

|模型|模式|左膝RMSE(°)|右膝RMSE(°)|髋膝踝RMSE(rad)|高度RMSE(m)|相对躯干Pitch RMSE(°)|
|---|---|---:|---:|---:|---:|---:|
|heading_300|raw|17.4441|17.2971|0.209479|0.0425813|6.38633|
|heading_300|wm|17.0547|15.7515|0.203955|0.0345296|5.92419|
|unconstrained26|raw|17.5652|17.2681|0.210669|0.0422106|6.46861|
|unconstrained26|wm|17.2233|15.6602|0.204345|0.0345252|5.91284|
|hard26|raw|17.522|17.2707|0.212202|0.0420005|6.5005|
|hard26|wm|16.3457|15.6672|0.200602|0.0359119|6.05702|

|模型|模式|行走5–20Hz RMS(rad)|踝Pitch 5–20Hz RMS(rad)|接触切换动作变化率RMS|关节速度RMSE|
|---|---|---:|---:|---:|---:|
|heading_300|raw|0.0103384|0.0169249|6.87611|1.63697|
|heading_300|wm|0.00916672|0.0148629|6.17113|1.60802|
|unconstrained26|raw|0.010552|0.0173589|7.0699|1.63863|
|unconstrained26|wm|0.010074|0.0162353|6.281|1.6146|
|hard26|raw|0.0103721|0.0172423|6.57291|1.63409|
|hard26|wm|0.00988005|0.0164575|6.68622|1.64402|

指标沿用sim_eval_metrics.py的clean参考阶段掩码及64点Welch频谱。左右膝分别计算，使用同一深蹲掩码；总体膝误差仍沿用原左右合并定义。不同模型的闭环轨迹不同，阶段曲线叠加只作描述，不作为同输入因果差分。所有完整逐帧记录、PD投影、力矩饱和与详细曲线保存在evaluation下。

## 实施与初始化证明

- 7003维Actor输入、23维输出、Adapter增益0.5；完整教师包括原核心Adapter及原heading修正，teacher_max_abs_error=0。
- 固定池来自旧Adam恢复分支model25，19,200条1155维base+128维冻结历史编码latent+2维heading context。history encoder完全未变，因此使用原存储latent不会更改对应Actor计算；不以抽样替代全池检查。
- 初始化model0所有state_dict张量与原heading300一致；model25只改变8个critic张量。model0/model25完整77的qpos/qvel/joint/rpy/PD目标/动作与基线逐项一致。
- model26只改变10个核心Adapter张量、2个heading张量及8个critic张量。主干、history encoder、WM、std与其他非训练模块完全未变。
- model0/model25/model26导出JIT与各自eager最大误差均0；model0与model25导出文件SHA相同。评估器按模型SHA和条件签名复用其结果，保留两者独立标签和模型文件，没有把相同模型当作独立随机重复。
- 训练环境配置与无约束Adam分支完全相同；原custom训练数值均一致，仅保存间隔1→25。training配置仅algorithm class名称及save_interval不同（config_match.json）。

## 联合Adam回退语义

原PPO采用一个Adam、三个参数组：core、critic、heading。每个候选更新前保存全部参数和Adam state；同一已裁剪梯度提出候选。有限比例1、0.5、0.25、0.125、0.0625、0.03125同时缩放core/head学习步长，critic学习率保持原值；每次重试完整恢复原参数/状态后重新执行Adam。所有重试的critic参数、动量和step必须与第一次候选逐位相同。接受时Actor/critic各前进一步；全拒绝时Actor参数和Adam状态复原、critic保留一次更新。

检查器对CPU和训练CUDA后端都执行完整池前向，只读、停止梯度、检查RNG未变，取两后端最大偏差。统计P95/P99/均值使用训练后端全23维误差；阈值无容差。原始保存teacher targets与相同完整教师CPU重算的最大浮点差3.8147e-6，因此约束分别采用相同后端和chunk的冻结教师输出；最终策略相对原始存档targets的最大偏差也核验为9.87947e-5，没有放宽阈值。

## 学习、拒绝率与动作漂移

- 48次Actor尝试接受6次、拒绝42次，接受率12.5%；接受比例[0.5, 0.125, 0.0625, 0.03125, 0.0625, 0.0625]。所有279次候选全池检查均记录；已保留状态最大偏差始终≤1e-4，最高9.973347e-5。
- 最终全池最大偏差9.8824501e-05，均值9.32311832e-06，P95 3.05771828e-05，P99 4.86373901e-05；核心输出RMS漂移1.34700376e-05，heading输出RMS漂移9.81270432e-07。
- 累计核心参数L2变化6.60168687e-05，heading参数L2变化5.73632231e-05；共497,726个参数元素实际变化。Adam Actor step=13200→13206，critic=14400→15648。48次事务核验全部通过。
- PPO surrogate loss均值-3.02002785e-05，最大KL 2.3345153e-07。梯度、损失、接受比例、实际学习率、回退次数、参数变化及优化/check耗时详见minibatch_updates.csv和原始hard_constraint_steps.jsonl。
- 这证明真正执行了非零梯度更新，但不证明任务性能获得有效改善。87.5%的尝试被拒绝，模型接近硬约束边界，存在后续学习停滞风险；本轮同时发生WM保护失败，不属于“Yaw稳定且学习成功”的情况A，也不是“完全不学习所以稳定”的情况B。

## 为什么不能宣称约束解决了退化

1. 控制条件相同、clean和WM处理后参考逐帧完全一致，model0/model25的完整闭环结果又与heading300逐项一致；初始化、warmup与不同参考不能解释本轮差异。
2. 全池硬约束在每个实际Actor步骤严格通过，且完整基线77共同观测3207帧上的策略最大动作漂移仅8.028746e-5（raw/WM均低于1e-4）；但WM闭环Yaw明显失败。这直接否定了“本固定池上1e-4约束足以保护完整77”的本轮假设。
3. WM新轨迹相对基线首次连续0.2秒超过5°差异发生在13.40秒，raw为20.72秒；这是描述性轨迹分离时间，而非因果起点。微小功能变化仍可形成显著动力学/接触/历史反馈差异。
4. 当前证据不能把失败直接归因为anchor池漏掉了基线77状态：基线77共同状态输出也低于阈值。新闭环轨迹上的全部Actor输入未在本轮开启额外sidecar记录，因此其逐状态偏差及局部动力学增益尚未测定；不能声称已证明新轨迹也全局满足1e-4，不能把池覆盖假说写成已确认根因。
5. 更小的全局动作漂移并不自动带来更小的Yaw误差；受扰方向、接触切换与历史状态有关。本轮只有单seed单动作，不能据此断言所有更强锚定都会失败。

## 下一步唯一方向

**不启动50次训练，也不立即更换阈值。优先做一次只读的短时闭环分叉灵敏度诊断**：以本轮WM约13.4秒首次明显分离之前的共同物理状态为起点，完整保存/恢复MuJoCo状态、controller/观测/动作历史、参考时间和RNG；在相同状态及同一WM参考下分别运行heading300与本轮hard26，记录前1–2秒的完整动作、接触、Yaw和所有7003维实际Actor输入。先证明baseline分叉恢复与原轨迹一致，再测新轨迹上的完整教师偏差，区分“离开受保护状态后动作偏差越界”和“仍在小偏差内但接触/动力学敏感”。

这一步不训练，不修改网络、奖励、WM、PD或阈值。只有找到敏感状态/动作方向和放大环节，才有依据设计下一轮保护约束；目前不能以继续堆更新轮数或默认扩anchor池代替诊断。next_minimal_config.json标记为未执行诊断计划，training_updates=0。

## 测试、耗时和保存位置

- CPU 6项、CUDA 6项测试通过：共享Adam缩步与参考一次更新逐位匹配、全拒绝状态恢复、异常事务恢复、core/head完整输出抵消、warmup及真实19,200池/RNG检查。
- 独立3-update smoke通过；只执行critic更新，Actor/Adam Actor状态逐位保持。随后只执行一次独立26-update正式实验。
- 正式runner总训练时间170.92秒，其中第26次142.53秒；Actor优化事务合计141.38秒，全池检查140.63秒，约束检查占绝大部分。单次池检查约0.44–0.45秒；最慢拒绝步骤需要6候选+恢复后复核。初始化和JIT导出/仿真耗时不计入runner训练时间。
- 自动完整评估10个模型/模式结果（因model0/model25 JIT相同，实际8个仿真案例）；raw/WM全部3207帧成功，无quick截断。异常结果不会作为通过；保护失败已写入正式目录STOPPED.json。
- 正式模型目录：`legged_gym/logs/recovery_heading300_yaw_controls/hard1e4_adam_env512_seed42_20261012/`，保留model_0.pt、model_25.pt、model_26.pt。
- 报告/CSV/JSON/曲线目录：`reports/heading300_hard_constraint_20261012/`。完整自动报告：`evaluation/report.md`；Yaw/膝/Pitch曲线：`plots/raw_time_series.png`、`plots/wm_time_series.png`；约束/接受率：`plots/constraint_steps.png`。
- 未验证：其他动作、多seed、实机、下一轮50更新、新轨迹逐状态教师偏差、分叉机制。保持原heading300及所有旧checkpoint。

## 修改文件

新增`tools/heading300_hard_constraint.py`、`tools/train_heading300_hard_constraint.py`、`tools/run_heading300_hard_constraint.sh`、`tools/analyze_heading300_hard_constraint.py`和`tests/test_heading300_hard_constraint.py`。原训练入口、模型结构、奖励、物理执行和实机代码本轮未修改。新增本报告及独立实验产物。

## 已执行命令（复现说明，勿覆盖现有目录）

```bash
bash tools/run_heading300_hard_constraint.sh hard1e4_adam_env512_seed42_20261012 \
  --num_envs 512 --seed 42 --max_iterations 26 --save-every 25
```

评估使用`tools/evaluate_sim.py --mode standard --cases mocap1__77 --workers 1 --no-historical-control`，同时传入hard0/hard25/hard26及旧unconstrained26四个`--checkpoint label=path`，heading300由系统作为固定主基线加入。执行时CUDA_VISIBLE_DEVICES为空，训练先完全结束再串行评估。
