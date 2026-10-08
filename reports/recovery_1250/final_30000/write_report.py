import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;project=r.parents[2];old=project/'reports/recovery_1250/best_search_23500'
analysis=json.loads((r/'motion77_analysis.json').read_text());rows=analysis['results']
selected=project/'legged_gym/logs/recovery_1250/coordinated_1250_seed42_20260926/model_23500.pt';policy=old/'23500_live/policy.pt'
receipt=dict(selected='23500_live',checkpoint=str(selected),policy=str(policy),ema=False,checkpoint_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),scope='46 evaluated variants;26 new variants covering every saved step24000..30000 in both live/EMA. Earlier checkpoints sampled, not exhaustive.',baseline='original TWIST36500 frozen backbone without Adapter',comparison4900='pilot_gpu_seed42_200/model_4900.pt',training_complete_updates=30000,simulation=dict(motion='mocap1/77.pkl',duration=64.14,frames=3207,modes=['raw','wm'],seed=42,headless=True,synchronized=True),alternatives={'10000_live':'best short development gain;failed long confirmation','24000_ema':'better on77;long bootstrap confidence gate not passed','28500_ema':'long case velocity regression12.11%','30000_live':'development gain0.955%;not preferred'},limits=['Single training seed and deterministic rollouts; no new hardware trial','Existing gates do not constrain per-case yaw','Baseline comparison here is on77, not a full multi-motion baseline ranking'])
(r/'selection.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
text='''# 30000步训练最终评估与优化方案

训练已完成30000次更新。结合此前结果共评估46组原始/EMA权重，本轮完整补评24000至30000每500步保存的26组；不是遍历所有早期checkpoint。综合推荐仍为23500原始权重，不能使用其EMA。

## 选模依据

相对原1250，正数表示综合误差降低。权重：关节40%、根部速度30%、朝向30%。

|模型|开发集改善|长时改善|判断|
|---|---:|---:|---|
|10000原始|2.57%|-0.90%|短时最好，长时失败|
|23500原始|1.89%|2.18%|通过现有两阶段条件，综合推荐|
|24000 EMA|1.88%|1.63%|77更好，但长时bootstrap上界1.01936，未通过确认|
|28500 EMA|2.05%|1.14%|长时单动作速度退化12.11%|
|30000原始|0.96%|未补测|终点并非最好|
|30000 EMA|-1.87%|未补测|开发集退化|

23500确认上界0.99974也接近边界，不能把门槛通过解读为多种子或实机上的确定优势。现有保护条件未约束逐动作yaw退化。新增27500 EMA在mydata2__3_seg00/wm出现摔倒；已有23500 EMA摔倒可复现。

训练9600–9999与29600–29999窗口：奖励64.05→64.77、回合长度1046→1057、支撑脚滑移2.612→2.606。Actor LR在10000后固定2e-7，最后动作相对1250的MAE约0.00205。收益处于小幅波动平台，没有证据说明继续同配置延长训练能修复结构性问题。

## mocap1/77高层+低层仿真

实际启动server_high_level_motion_lib.py和server_low_level_g1_sim.py，使用隔离Redis端口、CPU、同步参考、headless模式，raw/wm分别跑完整64.14秒（每次3207帧）。baseline按项目既有定义是原始TWIST36500、不带Adapter；4900是pilot_gpu_seed42_200模型。额外测1250、24000 EMA和28500 EMA。所有策略的clean参考逐帧完全一致。原始日志、进程输出、指标、帧轨迹和哈希均保留。

|模型|模式|关节RMSE(rad)|速度RMSE(m/s)|朝向RMSE(deg)|摔倒|
|---|---|---:|---:|---:|---|
'''
for label in ['baseline','ckpt4900','reference1250','ckpt23500','ckpt24000ema','ckpt28500ema']:
 for mode in ['raw','wm']:
  x=next(x for x in rows if x['label']==label and x['mode']==mode);text+=f"|{label}|{mode}|{x['joint_rmse']:.5f}|{x['root_velocity_rmse']:.5f}|{x['yaw_rmse_deg']:.2f}|{x['fall_proxy']}|\n"
text+='''
23500相对4900：raw综合改善0.15%，wm改善5.60%；相对原始TWIST：raw综合退化15.51%，wm退化0.46%。这里是单个77动作上的固定权重结果，不代表所有动作上的总体排名。

23500的下肢RMSE raw0.190、wm0.192rad；baseline约0.136、0.133rad。右髋yaw RMSE约0.393/0.400rad，而baseline约0.202/0.203rad。23500右髋yaw平均有符号误差约+0.186/+0.190rad。4900已具有相近偏差，因此继承错误是明确现象；把它全部归因于某一损失或单个输入仍需要消融。

23500在前10秒yaw RMSE raw约14.6°、wm14.2°，40–64.14秒增至约129.3°、103.1°。轨迹图显示后半段累计偏航明显。23500相对baseline用关节跟踪损失换取部分速度和wm朝向收益，尚未全面超过原始TWIST。没有摔倒只说明未触发本协议的骨盆/倾角判据。

## 优化方案：分开验证五个假设

### E0：先定位继承的Adapter偏差

保持全部23个关节可协调控制。增加仅用于消融的Adapter残差强度alpha，测试0、0.25、0.5、0.75、1，观察髋yaw/肩yaw误差与稳定性变化；alpha0必须数值验证恢复原始TWIST。不同层的残差缩放不等同于动作线性插值，不直接在实机尝试未经验证的混合动作。

选模加入原始TWIST对照：至少下肢/上肢/腰分项，以及髋yaw尾部误差。不要仅用1250教师锚定限制所有状态。对成对仿真中baseline明显更好的训练片段，加入baseline行为保持或蒸馏；对Adapter在稳定性/扰动恢复上更好的片段保留它的行为。教师按完整状态/片段选择，避免按上下肢硬拼策略破坏协调。当前强锚定限制偏离1250，是可能阻碍修正继承偏差的因素，需单独验证。

### E1：补齐长期朝向误差信息

当前base观测有参考yaw、角速度、roll/pitch，却没有显式实际heading error，79帧历史只有约1.58秒。增加Adapter专用的sin/cos相对heading error，使用对齐后的机器人IMU/状态估计和参考启动朝向。额外通道零初始化保持初始23500输出，保留冻结TWIST的1155维契约，重新训练/导出扩展观测策略；不能把新增维度直接塞入旧JIT。

配套周期yaw损失、yaw-rate跟踪、转向结束后的残余角速度。训练和部署必须使用同一种heading估计和起始对齐，验证估计自身漂移，不能训练用真值而部署用不可比估计。

### E2：条件化接触与漂移目标

在参考下肢近静止且支撑接触可信时约束脚滑移、足部空间漂移和根部非预期移动。迈步、蹲起、转身时允许参考规定的腿腰运动及必要平衡补偿。用平滑条件权重并记录覆盖率，避免阈值抖动与误罚摆动脚。不冻结上肢或下肢。

损失分别记录joint/leg/hip-yaw、root velocity、yaw、slip的均值和尾部。现有tail_anchor只惩罚偏离教师的动作，不等于约束物理误差尾部；需加入后者。先看梯度量级与奖励有效覆盖，再调系数，避免一口气提高所有惩罚。

### E3：匹配实机执行链路

将真实TargetSafetyFilter、动作缩放、渐入/回退和PD参数加入仿真。默认实机踝零位-0.18rad与训练-0.20rad的差异要明确归一化/动作零位，站立过渡姿态单独定义。已有部署代码先提交raw动作历史，再过滤实际目标；必须统一训练与部署历史语义，不能只改last_action一个变量。

参考包增加采集/发送时间和帧号，按实测分布加入参考延迟、丢帧与时序抖动；旧日志redis_age0/frame_id-1不能证明参考新鲜。先固定参考回放，再实时输入。保留安全限幅，不通过取消限幅换取仿真分数。

### E4：训练实验与验收

建议另开优化分支：23500原始权重初始化，1250与原始TWIST都保留为对照。更换奖励/观测时重新初始化或适应价值函数/优化器，不绕过原恢复配置校验。以下是起始实验预算，不是保证最优的超参数：

|实验|变更|初始预算|评估|
|---|---|---|---|
|A|仅部署执行链路匹配|1000–2000更新|每250步导出live/EMA|
|B|A+heading context/周期yaw目标|2000更新|同动作和初始条件|
|C|B+baseline行为保护+条件接触目标|2000更新|检查髋yaw与腿部误差|
|D|C+实测时序扰动|2000更新|seed42/43及异步参考测试|

可从actor LR5e-7开始单独验证，沿用相同环境数；不再直接投入30000步。一次只改变一类主要因素，允许中途退化，按保存模型外部选择。77用于问题诊断与回归，同时新增未用于选模/调参的转向、摆臂和行走长动作进行最终确认。

建议阶段性目标：下肢误差较23500至少下降10%，长时yaw误差下降20%，raw/wm均无新增摔倒且逐动作速度退化不超过5%；最终再要求下肢误差接近baseline（例如不高于5%）。这些为拟议验收目标，尚未通过实验验证。原有逐动作yaw、脚滑移与漂移保护缺失应补齐；比例和绝对误差同时看，避免小基数误判。

## 复现指定动作

运行已保存的驱动会实际启动并回收高层、低层和专用Redis进程；output需使用新目录：

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
/home/hank/anaconda3/envs/twist/bin/python \\
  reports/recovery_1250/final_30000/run_motion77.py \\
  --output reports/recovery_1250/final_30000/motion77_repeat \\
  --policy baseline=legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt \\
  --policy ckpt4900=legged_gym/logs/g1_motion_wm_anyadapter/pilot_gpu_seed42_200/traced/pilot_gpu_seed42_200-4900-anyadapter-opentrack-jit.pt \\
  --policy ckpt23500=reports/recovery_1250/best_search_23500/23500_live/policy.pt
```

本次只新增评估和分析文件，没有修改训练/实机控制代码，也没有启动实机。优化方案尚未实施。单训练seed、确定性仿真及有限动作覆盖不能替代真实硬件验证。
'''
(r/'assessment_and_plan.md').write_text(text)
print('Saved',r/'assessment_and_plan.md')
