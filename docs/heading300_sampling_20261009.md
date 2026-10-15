# heading_300：完整教师与深蹲采样独立实验（2026-10-09）

本实验从 `legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt` 起步。原模型、原23500及其JIT均保留。实验在 `legged_gym/logs/recovery_heading300_sampling/`，审计与汇总在 `reports/heading300_sampling_20261009/`。

## 改动与不变项

独立脚本 `tools/train_heading300_sampling.py` 复用原heading训练实现。冻结教师的目标改为 `source.layerwise_actor(base, z) + source.heading_delta(z, context)`，核心Adapter自身使用增益0.5，heading项只加一次，不再遗漏已经学到的heading修正。回放缓存也保存完整目标。

网络结构、模型初始化权重、奖励、WM配置及权重、PD、目标投影（margin=0.02，knee_min=0）和部署动作EMA（0）不变。每次启动比对原manifest中的完整环境配置，只允许采样配置、seed和环境数量字段不同。原训练脚本未改动，避免破坏其恢复配置检查。新分支使用独立优化器、回放缓存和从0开始的专项更新计数。

GPU上已有约14 GB其他进程占用，未经干预保留这些进程。本实验使用512个环境，原heading实验为4096，rollout仍为24步、16 minibatches、3 epochs。这改变了批量大小，因此本实验不是严格排除批量影响的采样因果消融。不得把所有性能变化都归因于采样。

## 采样审计

最终配置：`reports/heading300_sampling_20261009/sampling_alpha014_p1.yaml`。

设原概率为p，每个动作中的参考可信双支撑深蹲比例为s，专项分布为归一化的p×s。新概率为0.86×p+0.14×专项分布。保留全部12,248个原动作，每个动作均保留至少86%的原概率；动作内部时间采样不变，不固定上下肢。77仍为未加入训练的回归动作。

|检查|有效深蹲|参考可信双支撑深蹲|再要求实际双足接触|正常行走|转向|
|---|---:|---:|---:|---:|---:|
|512环境×1024步，0优化器更新|11.146%|7.257%|4.788%|49.893%|29.915%|
|专项训练前50次更新|10.501%|6.830%|4.436%|51.327%|31.066%|

深蹲及行走阶段阈值与原诊断一致。参考可信双支撑定义为双足reference stance confidence均>0.5；实际接触额外要求两足竖直力均>5 N。这两个比例不混为一谈。上表前两列达到10%–12%与5%–8%的目标；更严格的实际接触占比尚未达到5%，不能声称所有深蹲都执行稳定。

初始方案alpha=0.11、power=2和第二方案alpha=0.16、power=1的零更新审计参考可信双支撑比例分别为8.53%和8.45%，因此降低到最终alpha=0.14；这些审计没有进行策略训练。

## 初始化与冒烟验证

- actor全部state_dict张量与源checkpoint逐项相等。
- 策略7003维输入、23维输出、critic 1318维输入、Adapter增益0.5。
- 实际环境观测上策略与完整冻结教师max_abs_error=0；所有教师参数requires_grad=False。
- 在初始环境观测上，若遗漏heading，max_abs_error=0.00053452，确认修复项不是空分支。
- 初始JIT与原heading_300 JIT在seed42、batch=1/2/17/64的输入上最大差为0，7001维错误输入被拒绝。导出eager/JIT误差为0。
- 3更新冒烟及model_3→第4更新恢复通过；计数一致，warmup时策略保持不变，critic发生更新，无非有限损失。
- 完整教师2项测试、现有自动评估15项测试通过。

验证凭证：`jit_initialization_parity.json`、`smoke_receipt.json`、`experiment_receipt.json`，以及各run的`initialization.json`和`sampling_audit.json`。审计过程中修复了新采样YAML在ASCII默认locale下的读取问题（使用ASCII转义）与新增检查的启动缩进错误；均发生在训练前，修复后重新通过检查。

## 训练与自动评估

```bash
bash tools/run_heading300_sampling.sh sample_only_alpha014_p1_env512_seed42_20261009 \
  --sampling-config reports/heading300_sampling_20261009/sampling_alpha014_p1.yaml \
  --seed 42 --num_envs 512 --max_iterations 250 --save-every 50 \
  --auto-eval --eval-interval 50
```

每50次更新完成原子保存后调用既有AutoEvalHook，CPU导出JIT，并完整评估77（64.14秒，3207帧）的raw和WM，同时重跑heading_300与历史23500。训练在评估期间暂停，评估CUDA不可见、单worker、全局锁防并发、每个checkpoint独立目录。未启用`--auto-eval`时不会触发评估。未完成、异常或无法启动评估都停止实验，不能视作通过。

heading_300与候选保持seed42、raw/WM、参考内容、PD、投影、EMA和控制参数一致；历史23500使用原增益1、7001维输入、无目标投影，单独标注，不参与验收判断。Yaw误差超过heading_300的105%或新增跌倒即停止，不追加训练轮数。其他速度/深蹲协调/高频验收项完整报告，未通过不能宣布整体更优。

所有关节、Pitch、COM、力矩、动作变化率及频谱计算沿用`sim_eval_metrics.py`。Yaw为RMSE，膝误差为左右膝合并RMSE；关节速度参考沿用clean角度的numpy.gradient估计；行走高频为连续片段≥64帧的64点Welch，5–20 Hz。

## 实际结果：第50次更新停止

专项run：`sample_only_alpha014_p1_env512_seed42_20261009`。完成50次更新、614,400个训练transition，其中前25次为原有critic warmup，后25次更新策略。最大rollout KL=0.001036，没有跳过minibatch，没有非有限损失或冻结模块变化。第50次更新完整评估确认Yaw退化，保护逻辑写入`STOPPED.json`并退出，**未运行100/150/200/250更新**。允许曲折进步不等于在明显朝向退化后继续追加预算。

完整77的WM结果如下；RMSE与频谱定义未改变。

|指标|heading_300|专项model_50|原23500历史条件|
|---|---:|---:|---:|
|23关节RMSE，rad|0.139306|0.139846|0.180957|
|关节速度RMSE，rad/s（参考微分估计）|1.608025|1.622242|1.675120|
|根部线速度RMSE，m/s|0.103864|0.105355|0.108094|
|Yaw RMSE，度|32.080|41.754|72.364|
|深蹲左右膝合并RMSE，度|16.416|16.449|17.386|
|深蹲髋膝踝协调RMSE，rad|0.203955|0.204268|0.263878|
|深蹲参考相对躯干Pitch RMSE，度|5.9242|5.9164|4.1785|
|深蹲实际平均Pitch，度（参考16.729）|11.8593|11.8894|14.3660|
|深蹲高度RMSE，m|0.034530|0.034701|0.049805|
|行走下肢5–20Hz RMS，rad|0.0091667|0.0106533|0.0099466|
|行走踝Pitch 5–20Hz RMS，rad|0.0148629|0.0170763|0.0158881|
|接触切换附近下肢动作变化率RMS，action/s|6.1711|6.6892|6.6217|
|双支撑深蹲静态COM平均裕量，cm|5.0151|5.0227|4.7885|
|全程投影帧比例|28.188%|28.968%|0%（历史无投影）|
|全23关节力矩饱和比例|2.371%|2.353%|2.462%|
|跌倒代理指标|无|无|无|

WM：Yaw +30.16%，膝误差+0.203%，协调误差+0.153%，下蹲高度误差+0.496%，行走高频+16.22%，踝Pitch高频+14.89%，接触切换动作变化率+8.40%。下蹲Pitch误差仅减少0.0078°（0.13%），不足以认定有实质姿态改善。实际平均Pitch并未变得更直立，也不应把参考要求的前倾强行去掉。原23500的Pitch误差更小而膝/协调误差更大，说明不能仅凭Pitch选择模型；其执行条件不同，不把差值解释成纯模型效果。

|77 raw指标|heading_300|专项model_50|变化|
|---|---:|---:|---:|
|关节RMSE，rad|0.138451|0.139032|+0.419%|
|关节速度RMSE，rad/s|1.636966|1.633045|-0.240%|
|根部线速度RMSE，m/s|0.109049|0.105678|-3.092%|
|Yaw RMSE，度|67.777|81.936|+20.89%|
|深蹲膝RMSE，度|17.3707|17.3806|+0.057%|
|深蹲Pitch误差，度|6.3863|6.3117|-1.168%|
|行走5–20Hz RMS，rad|0.0103384|0.0102394|-0.957%|

raw有局部速度/Pitch/高频改善，WM却有多项退化，两种模式均超过Yaw门槛。不能宣布model_50整体更优。

### 9动作回归与验收

停止训练后，另用CPU对8个标准开发动作进行回归，合计9动作×2参考模式×3模型=54个仿真结果。77为全64.14秒；其余动作沿用协议的3.38–8秒开发长度。已保存逐帧记录、独立报告、完整相位CSV与所有关键曲线。

|验收项|结果|
|---|---|
|无新增跌倒|通过；9动作两种模式中候选均无跌倒代理触发|
|逐动作根部速度误差退化≤5%|通过；最差为mocap1/20 WM，+4.736%|
|逐动作Yaw误差退化≤5%|未通过|
|raw/WM深蹲协调均改善|未通过；9动作聚合raw -0.072%，WM +0.735%|
|raw/WM行走高频均改善|未通过；9动作聚合raw -0.161%，WM +6.360%|
|77下蹲姿态实质改善|未证明；膝/蹲深略差、Pitch变化极小|

Yaw未通过的不止77：站立摆臂raw 1.114°→1.210°（+8.61%），WM 1.412°→1.514°（+7.21%）；mydata2/3 raw 86.991°→96.493°（+10.92%），WM 83.078°→88.540°（+6.57%）。摆臂动作的绝对变化约0.1°，需与77约10–14°的绝对退化区别，但严格相对门槛仍未通过。

完整模型权重及heading_300/23500 JIT的SHA256在评估前后均保持不变，训练GPU在停止后释放；其他已有进程未终止。本次没有实机测试。

### 原因排序与证据边界

1. **确定：完整教师修复正确，但现有动作锚定并不保证闭环Yaw不退化。** 第0次策略和教师严格一致；保存回放中的完整教师目标CPU复算最大差3.58×10⁻⁶（CPU/GPU数值差）。第50次回放的完整动作漂移RMS约0.001571 action单位，源动作MAE约0.001097，KL低且无数值异常，却出现显著轨迹Yaw变化。既有诊断已证明很小的heading修正可以影响长时间闭环朝向。不能用低训练KL或低动作平均误差替代仿真Yaw验收。
2. **确定有参数漂移，主要责任尚未独立验证。** 核心Adapter输出漂移RMS=0.001519，heading输出漂移RMS=0.0001895；原heading输出RMS=0.0004510，后者的变化约为其原有量级的42%。核心也明显改变，不能仅据此把退化归于heading头。77 WM下蹲/行走/转向阶段Yaw RMSE分别20.98°→43.48°、36.94°→43.52°、38.77°→47.23°，并非只有行走某个瞬时跳变。
3. **确定：覆盖增加没有在本轮转化为膝/蹲深改善。** 实际训练有效深蹲10.50%、参考可信双支撑6.83%，比旧审计明显增加；但策略仅更新25次，约束强且平均输出变化小，不能据一次提前停止否定采样本身，也不能据覆盖达标就宣称姿态改善。静态支撑裕量基本相同、全关节饱和比例未增加，现有证据不支持靠改PD、取消投影来补救本次失败。
4. **必须保留的混杂因素：批量变化与critic风险。** 4096→512环境增加了梯度方差和训练数据量差异，尚无同批量原分布控制，不能把退化完全归因于加权采样。critic仍为1318维，没有显式实际heading误差上下文，这一已知风险未被本轮改变；没有按heading误差分组的价值残差证据，不宣称它就是本次退化根因。

### 最优checkpoint与后续判断

**保留heading_300为最优且可继续作为参考的checkpoint。本轮没有可替代它的新模型。**

- 推荐训练checkpoint：`legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt`。
- 推荐JIT：`reports/recovery_1250/fix_stage_20261008/heading_300.pt`。
- 本轮候选：`legged_gym/logs/recovery_heading300_sampling/sample_only_alpha014_p1_env512_seed42_20261009/model_50.pt`，未通过验收，仅保留用于诊断。
- 候选JIT在对应`evaluations/update_00000050_.../exports/checkpoint/policy.pt`；7003→23，增益0.5，eager/JIT误差0。不建议拿它替代heading_300部署。

**不值得按当前配置直接续到250或更长。** 下一轮仍可保留本次达标的采样配置，但先做小而可归因的检查：同样512环境、完整教师、原动作分布的短控制，区分批量/继续优化与采样影响；记录按heading误差分组的价值残差。随后独立测试保留已有heading参数、仅训练核心Adapter，或单独降低heading学习率（例如5e-6→1e-6），不得两项一起改；这不固定任何腿或上肢关节，也不改变网络结构。核心变化仍可能影响朝向，所以任何方案都必须通过闭环Yaw保护，不能假定冻结heading参数就足够。

下一次可在10/25/50更新加密评估以定位失效起点，仍维持原奖励、WM、PD、投影和EMA。只有先守住Yaw后才值得继续观察采样是否改善蹲深/协调；若覆盖达标且朝向保持而膝仍无改善，再回到已提出的独立参考相对支撑约束方案，需要新的授权及消融，不能在本轮顺手混入。上述后续训练均未启动。

未验证：不同seed、4096环境严格同批量控制、完整16动作长时协议、实机效果。50次更新且提前停止不足以判断平台期，不能解释为训练已经收敛。

## 交付文件与产物

本阶段新增：`tools/heading300_teacher.py`、`tools/build_heading300_sampling.py`、`tools/train_heading300_sampling.py`、`tools/run_heading300_sampling.sh`、`tools/summarize_heading300_sampling.py`、`tests/test_heading300_teacher.py`及本报告。既有仿真计算与自动评估模块直接复用。本阶段未编辑实机代码、原网络/奖励实现或原训练入口。

- [每50更新汇总CSV（实际仅50）](../reports/heading300_sampling_20261009/final_summary/metrics.csv)
- [专项汇总JSON](../reports/heading300_sampling_20261009/final_summary/summary.json)
- [专项曲线](../reports/heading300_sampling_20261009/final_summary/checkpoint_comparison.png)
- [9动作完整对比报告](../reports/heading300_sampling_20261009/full_comparison/report.md)
- [9动作逐阶段CSV](../reports/heading300_sampling_20261009/full_comparison/phases.csv)
- [额外8动作报告及曲线](../reports/heading300_sampling_20261009/regression_step50/report.md)
- [采样/恢复/JIT/原资产验证目录](../reports/heading300_sampling_20261009/)
- [回放漂移分析](../reports/heading300_sampling_20261009/replay_drift_step50.json)
- [按原相位划分的Yaw分析](../reports/heading300_sampling_20261009/phase_yaw_step50.json)

9动作报告由两个已完成报告通过已有`compare_models`/`write_report`合并，核验模型哈希一致、每个对照条件及参考摘要匹配。复现汇总：`python reports/heading300_sampling_20261009/merge_comparison.py`（要求新输出目录，现有结果不覆盖）。
