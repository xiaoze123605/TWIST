"""Render the independently saved first-update diagnosis, including limits."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def report(root):
    v=json.loads((root/'verification/verification.json').read_text())
    a=json.loads((root/'analysis/results.json').read_text())
    c=json.loads((root/'coverage/results.json').read_text())
    f=json.loads((root/'fork/results.json').read_text())
    timing=json.loads((root/'analysis/timing_and_instrumentation.json').read_text())
    jit=json.loads((root/'analysis/actual_jit_consistency.json').read_text())
    summary=json.loads((root/'evaluation/summary.json').read_text())
    assert len(f['results'])==4 and (root/'fork/COMPLETE').exists()
    # Local controlled intervention figure, distinct from full77 trajectories.
    fig,axs=plt.subplots(3,1,figsize=(14,10),sharex=True)
    baseline=None
    for label,item in f['results'].items():
        frames=[json.loads(x) for x in Path(item['trace']).read_text().splitlines()][900:]
        yaw=np.array([x['rpy'][2] for x in frames]);ref=np.array([x['clean'][3] for x in frames])
        if label=='null':baseline=yaw
        t=18+np.arange(len(frames))*.02
        axs[0].plot(t,np.degrees(np.arctan2(np.sin(yaw-ref),np.cos(yaw-ref))),label=label)
        axs[1].plot(t,np.degrees(np.arctan2(np.sin(yaw-baseline),np.cos(yaw-baseline))),label=label)
        contacts=np.array([x['dynamics']['foot_contact'] for x in frames]);axs[2].plot(t,contacts.sum(-1),label=label,alpha=.65)
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    axs[0].set_ylabel('Wrapped yaw error (deg)');axs[1].set_ylabel('Gap vs null (deg)');axs[2].set_ylabel('Contact feet');axs[2].set_xlabel('Identical-prefix fork time (s)')
    fig.tight_layout();fig.savefig(root/'analysis/plots/controlled_fork.png',dpi=140);plt.close(fig)
    pg=c['historical_first_minibatch_gradient'][0]
    hist={r['group']:r for r in c['historical_coverage'] if r['mode']=='wm'}
    actor_changes=[x for x in v['first_update_changed_parameters'] if not x['parameter'].startswith('critic.')]
    plan=dict(status='proposed_only_not_started',source='heading300 training model_300.pt',
        unique_change='pointwise complete-teacher functional deviation bound after each actor minibatch step, on the existing fixed model25 anchor-bank inputs',
        reason='sensitive-state omission is not demonstrated; the current mean/tail soft penalty permits measured pointwise drift and is nearly inactive at the first minibatch',
        proposed_initial_action_max_abs_bound=1e-4,bound_status='engineering hypothesis; not validated to preserve Yaw',
        probe_source='model25 replay bank:19200 exact saved base+frozen latent+heading context inputs with complete teacher targets; fixed for both control and candidate',
        teacher='full frozen heading300 core Adapter plus learned heading correction, gain0.5',
        intervention='jointly backtrack core and heading parameter displacement; do not freeze any limb/head, change reward, or retune Adam/LR',
        updates_total=26,critic_warmup=25,actor_updates=1,num_envs=512,seed=42,rollout_steps=24,minibatches=16,epochs=3,
        matched_control='current Adam-restored condition; actorLR5e-7→2e-7 schedule, headingLR5e-6, criticLR5e-6, anchor1.6/0.8, scale0.087, tail0.01, critic1318, source Adam restored; all unchanged',
        evaluate_after=[0,25,26],evaluation='full77 raw/WM same seed, EMA0, projection on; protected Yaw/root velocity≤105% heading300, no new falls',
        promotion='no extension if either mode fails; passing protection is not a10% improvement or multi-motion acceptance',
        no_training_dataset_change=True,no_network_change=True,no_reward_change=True,
        limitation='functional action bound may still fail closed-loop protection or stall learning; no guarantee that1e-4 is sufficient; do not combine with sampler changes')
    (root/'next_minimal_config.json').write_text(json.dumps(plan,indent=2))
    lines=['# heading300第一次Actor更新精确诊断（2026-10-12）','',
        '本阶段仅诊断：未启动训练，未修改模型原权重、网络、奖励、critic、优化器或超参数，未做实机测试。当前最优参考仍为heading300。', '',
        '## 六个必须回答的问题','',
        '1. **model25保持heading300吗？保持。** 除critic八个张量因warmup更新外，所有Actor/冻结模块权重逐元素相同；完整77 raw/WM的关节、qpos/qvel、Yaw、策略动作、PD目标和7003维输入全程逐元素相同。不是仅凭随机输入或JIT成功推断。',
        '2. **model26修改了什么？** 第一次Actor更新包含48个minibatch step，改变核心Adapter五层的10个张量、heading头2个张量及critic8个张量；其余状态完全不变，std也未变。核心和heading都未被单独冻结。逐层变化量见`verification/verification.json`。',
        '3. **raw/WM为何不同？** 相同输入上，两种模式的直接动作变化RMS都约2.45e-4，并非WM直接输出漂移显著更大；两种模式都产生接触/状态/历史轨迹分离。raw行走/转向误差有所改善而下蹲更差，综合RMSE略降；WM多阶段误差恶化。raw是得失抵消，不能称没有退化。',
        '4. **存在机器人状态→WM参考的放大吗？没有这条路径的证据，而且当前实现没有该反馈输入。** 高层Motion-WM只吃clean参考的25帧历史；所有模型的处理后参考、原始历史及归一化输入全程相同。存在策略—机器人—状态/动作历史的闭环放大；应区别于Motion-WM参考反馈。AnyAdapter的冻结历史编码器参与actor，训练用world_model decoder不在部署actor forward中。',
        '5. **锚定遗漏了敏感状态吗？不能认定完全遗漏。** 历史WM rollout中高heading误差、转向及接触切换组合已有样本，已有anchor池中也存在明显pointwise动作漂移；证据更支持“平均/软尾部损失未提供逐状态功能保护”，而非简单缺少WM或大heading样本。具体未保存的接触标签、历史分组PPO梯度不能补造。',
        '6. **下一轮唯一修改？** 优先验证在现有固定完整教师anchor输入池上，对每个Actor minibatch候选步施加逐状态、逐关节最大动作偏差约束，并联合回退核心Adapter与heading头的参数位移。保留现有软锚定，其系数和所有其他条件不变。不先加采样、不先改critic、不冻结上下肢。该方案只是待验证假设，本阶段未实施训练。', '',
        '## 1. 一致性、条件和只读记录','',
        '- 模型：Adam恢复分支`model_25.pt`、`model_26.pt`；源`heading23500_gain05_pilot100_20261008/model_300.pt`。源hash及逐张量差异保存在verification。',
        f'- 随机128个7003维输入：model25/source误差0；三个模型JIT/eager误差均≤1e-6。六条实际轨迹的同batch JIT/eager最大误差为{max(x["jit_eager_same_batch_max_error"] for x in jit):.3g}；离线batch256与在线batch1最大差异≤{max(x["eager_vs_recorded_pre_ema_max"] for x in a["input_and_jit_receipts"]):.3g}，属于浮点batch差异，不能把它当更新漂移。',
        '- 全部使用CPU、seed42、完整77（64.14s/3207帧）、raw/WM、同步ready/frame协议、动作EMA0、Adapter增益0.5、PD步长1ms、20倍decimation、投影margin0.02/knee_min0。历史23500本阶段不加入，避免无投影条件干扰。',
        '- 新记录通过显式`--first-update-diagnostics`启用。只复制已消费的Actor输入到npy侧文件，以及读取WM历史；不额外更新任何历史、不抽随机数、不影响策略动作。启用记录后的heading300轨迹与上一轮未启用记录的基线逐元素相同。',
        '- 验证全部79帧AnyAdapter状态历史与过去state/action配对，以及10帧TWIST观测历史；错位误差均0。初始历史填充属于既有初始化语义，不被当作真实负时间帧。WM第24帧ready，第0–23帧回退原参考，25帧窗口每次重新运行GRU，无跨窗口携带hidden state。',
        f'- Motion-WM实际checkpoint：`{timing["wm_checkpoint"]}`，SHA256 `{timing["wm_checkpoint_sha256"]}`；高层路径及31维归一化输入保存在analysis。', '',
        '## 2. 完整77量化结果','',
        '|模型|模式|关节RMSE(rad)|根速度RMSE(m/s)|Yaw RMSE(°)|',
        '|---|---|---:|---:|---:|']
    for row in summary['rows']:
        m=row['metrics'];lines.append(f'|{row["label"]}|{row["reference_mode"]}|{m["joint_rmse"]:.9f}|{m["root_velocity_rmse"]:.9f}|{math.degrees(m["yaw_error"]):.6f}|')
    lines+=['','三模型均无跌倒代理触发。model26 raw关节/行走Yaw可改善，同时下蹲Yaw恶化；WM速度略改善但Yaw明显恶化，不能宣布整体更优。', '',
        '|阶段|raw model25→26 Yaw(°)|WM model25→26 Yaw(°)|',
        '|---|---:|---:|']
    for phase in ['descent','squat','rise','walking','turning']:
        raw=a['modes']['raw']['phase_yaw'][phase];wm=a['modes']['wm']['phase_yaw'][phase]
        lines.append(f'|{phase}|{raw["model25"]:.3f} → {raw["model26"]:.3f}|{wm["model25"]:.3f} → {wm["model26"]:.3f}|')
    lines+=['','阶段沿用已有phase_masks，可能重叠。后半程下蹲朝向差异可能继承此前转向产生的累计偏航，不能将下蹲阶段高误差直接认定为下蹲触发。', '',
        '## 3. 分离时间及同输入分解','',
        'Yaw分离阈值1/5/10°要求连续≥10帧（0.2s）；这只是透明声明的诊断事件阈值，不替换原Yaw RMSE定义。', '',
        '|事件|raw|WM|', '|---|---:|---:|']
    for event in ['policy_direct_gt1e_6','actual_joint_gt1e_6','contact_first_mismatch','yaw_gap_gt1deg','actual_joint_gt0_01rad','yaw_gap_gt5deg','yaw_gap_gt10deg']:
        rr=a['modes']['raw']['first_events'][event];ww=a['modes']['wm']['first_events'][event]
        lines.append(f'|{event}|{rr["time"]:.2f}s|{ww["time"]:.2f}s|')
    lines+=['','WM第一次接触状态不同发生在8.62s下蹲下降附近，持续1°Yaw分离在18.52s转向，5°在23.50s行走/转向。接触先分离是时序证据，不是单独接触事件的因果证明。raw的分离更早，因此不能说WM一定具有更大的放大增益。', '',
        '严格区分：直接效应=`π26(o25)−π25(o25)`；状态/历史部分=`π26(o26)−π26(o25)`。二者相加等于闭环动作差。后者只是固定策略的函数分解，不是控制单一关节/接触的因果效应。', '',
        '|模式|同输入动作RMS|同输入core RMS|同输入heading RMS|状态/历史部分RMS|', '|---|---:|---:|---:|---:|']
    for mode,m in a['modes'].items():lines.append(f'|{mode}|{m["direct_action_rms"]:.7f}|{m["direct_core_rms"]:.7f}|{m["direct_head_rms"]:.7f}|{m["feedback_component_rms"]:.7f}|')
    lines+=['','0秒输入、机器人状态、参考完全一致，最大动作差0.003724，最大项左肩Pitch；腿部还有右踝Pitch约−0.001216、右髋Yaw约−0.001086的动作差。它们不是足以单独归因的证据。23.50s同输入最大差仅0.000444，固定策略状态/历史部分最大差已约0.3211；不能把此时闭环大动作差归因于权重直接输出漂移。', '',
        '## 4. 受控短分叉：复制完整前缀，不作qpos-only恢复','',
        '复用现有RealTimePolicyController。固定记录下的同一WM参考，四组先完整重放0–18s的model25，再运行18–26s的8s分叉。所有前缀qpos/qvel、关节、Yaw、动作、PD逐元素相同；null分支整个26s也与原轨迹完全相同，因此继承了同一物理/控制历史而不是假设求解器warmstart可忽略。模块替换仅用于独立诊断策略，原文件未覆盖。', '',
        '|18s后策略|该8s Yaw RMSE(°)|相对null最大Yaw分离(°)|', '|---|---:|---:|']
    for name,item in f['results'].items():lines.append(f'|{name}|{item["post_yaw_rmse_degrees"]:.3f}|{item["post_yaw_gap_max_degrees"]:.3f}|')
    lines+=['','核心Adapter变化主导这一受控局部的Yaw轨迹分离；heading变化也有作用，两者组合非线性。但full26/core26在此局部RMSE反而较null低，说明从0秒开始的整段退化与进入转向前的状态/历史有关，不能把这一局部结论推广为“新core始终有害”或“只冻结heading即可解决”。', '',
        '## 5. 锚定覆盖及梯度证据','',
        f'- 历史真实第一次更新rollout含12288帧，其中WM {hist["all"]["frames"]}帧。WM heading≥30°占{100*sum(hist[k]["fraction_within_mode"] for k in ["heading_30_60","heading_60_120","heading_120_181"]):.2f}%；转向×单支撑×接触切换附近占{100*hist["wm_turn_single_switch"]["fraction_within_mode"]:.2f}%。不能认定这些类别完全没采到。',
        '- 历史model25 anchor bank有19200条输入，六个raw/WM×阶段池各3200，原sample对非空池均匀采样。它保存了真实base、冻结latent、heading context、完整教师目标和权重；没有保存真实接触标签和时间顺序，不能补出接触切换统计。各池model26的pointwise动作漂移最大约0.0041–0.0046，而RMS仅约0.00026–0.00028：已经进入anchor池的状态也允许产生尾部偏差。',
        f'- 历史首个minibatch的PPO/core梯度范数{pg["core_surrogate_gradient_norm"]:.6f}，完整锚定/core {pg["core_anchor_gradient_norm"]:.9f}，比值{100*pg["core_anchor_to_surrogate"]:.5f}%；heading比值{100*pg["heading_anchor_to_surrogate"]:.5f}%。锚定包括原on-policy、replay和tail，数值近零而非严格零。理想数学上，策略=教师时二次偏差损失的一阶梯度为零；浮点batch差异产生少量数值梯度。',
        '- 新独立固定策略审计：512环境、8×24步、98304帧、源配置/动作分布不变，optimizer_steps=0；原生GAE和timeout bootstrap。记录真实接触、clean阶段及动作概率，按raw/WM、heading、转向、支撑、切换、深蹲/行走分组。它不是历史rollout的重建。',
        '- 新审计WM全组model26条件梯度中on-policy+tail anchor/PPO比值core约10.80%、head约9.03%；转向×单支撑×切换组core约20.81%、head约18.10%，该组权重质量占比还高于样本占比。没有发现简单地把此组权重压低的证据。条件梯度最多2048个确定性均匀抽样状态，稀疏组不作因果结论。',
        '- 历史未保存完整Actor observation、采样action、优势对应的minibatch indices，无法准确重建历史每个分组的PPO梯度。已明确输出缺失项，只有保存的首minibatch全局梯度属于历史实测。独立审计的分组梯度仅含on-policy+tail；replay缺少支撑标签，不能声称重建了按支撑分组的完整replay梯度。', '',
        '## 6. 下一轮唯一最小方案（尚未训练）','',
        '保留现有完整教师、现有anchor输入池和所有训练设置，只新增Actor候选步的逐状态功能偏差保护。固定使用model25已有19200条anchor输入，确保原核心Adapter+heading教师目标正确；检查所有23个动作，不固定任何关节。候选Actor位移对core/head联合回退，避免单模块破坏已有补偿关系。',
        '- 首轮只到26次更新（25次critic warmup＋1次Actor更新），保存0/25/26，分别完整77 raw/WM评估。',
        '- 初始工程假设：池内最大完整教师动作偏差≤1e-4（动作单位，约为实测池内尾部的1/40）。超过阈值则对候选Actor参数位移回退；Adam公式/LR、critic更新、reward、sampler、WM、PD、projection、EMA均不改。checkpoint必须记录实际接受位移及优化器状态，不能把被拒绝步伪装成成功学习。',
        '- 1e-4尚未证明足以保护闭环，也可能导致学习停滞。只检验这一约束，不同时扩充数据池或改采样。保护门失败就停止；通过保护门仍不等于下蹲/抖动10%改善，更不等于多动作验收。',
        '- 选择约束而不是先改采样的理由：高heading/接触转向已有训练覆盖，同输入漂移并非只发生在WM；已有池上的尾部偏差与近零首次锚定梯度是更直接证据。若后续固定池保护通过但闭环仍退化，再单独检查池外动态覆盖。', '',
        '配置详见`next_minimal_config.json`，只是下一轮方案，没有启动训练。', '',
        '## 7. 产物与复现','',
        '- 统一评估：`evaluation/manifest.json`、`summary.json`、CSV、report.md、六条完整trace及`frames.jsonl.actor_inputs.npy`。',
        '- 精确权重和随机JIT检查：`verification/verification.json`；实际JIT：`analysis/actual_jit_consistency.json`。',
        '- 逐帧：`analysis/raw_frames.csv`、`wm_frames.csv`；关键状态：`analysis/key_states.csv`；同输入及闭环分组：`analysis/simulation_groups.csv`。',
        '- 组件：`analysis/*_components.npz`（base/core/adapter/head/full/latent）；反事实：`*_counterfactual.npz`；WM归一化输入：`*_wm_inputs.npz`。',
        '- 曲线：`analysis/plots/{raw,wm}_timeline.png`、`{raw,wm}_first_divergence_joints.png`、`controlled_fork.png`，以及原评估的全部23关节、PD、COM和频谱图。',
        '- 历史coverage/anchor-bank与独立分组梯度：`coverage/*.csv`、`results.json`；独立审计输入：`readonly_audit/audit_states.pt`及receipt。',
        '- 受控分叉：`fork/results.json`及四条逐帧trace；源checkpoint保留。', '',
        '实际评估命令：', '',
        '```bash',
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\",
        '/home/hank/anaconda3/envs/twist/bin/python tools/evaluate_sim.py \\',
        '  --checkpoint model25=legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_25.pt \\',
        '  --checkpoint model26=legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_26.pt \\',
        '  --mode standard --cases mocap1__77 --workers 1 \\',
        '  --first-update-diagnostics --no-historical-control \\',
        '  --output reports/heading300_first_actor_update_20261012/evaluation',
        '```', '',
        '输出已存在，复现必须使用新的独立目录，不能覆盖。', '',
        '本轮未检验其他动作、多个训练seed、实机，以及其他分叉时刻/更长干预的泛化。没有将数值一致、局部分叉改善或统计关联宣称为整体优化成功。']
    lines+=['', '## 补充：根因证据排序和接触/执行目标', '',
        '第一层（已测）：Actor首次更新的核心Adapter功能变化，通过机器人—状态/动作历史闭环形成路径依赖；不是外部Motion-WM参考被机器人反馈改写。冻结模块/JIT/同步异常已由完整轨迹对照排除。局部分叉支持core的影响大于heading，但并不证明每个关节或每个转向状态都变差。',
        '第二层（保护不足的直接证据）：完整教师均值/软尾部锚定在首minibatch几乎没有一阶抵消，且已有anchor池状态仍允许约0.004的逐点尾部漂移。该证据支持下一轮检验功能约束，尚不证明约束一定改善Yaw或下蹲。',
        '第三层（待验证）：具体足部碰撞时刻、摩擦/支撑切换、上肢反作用及历史反馈分别贡献多少。不能只按闭环动作差、参数L2、力矩饱和或contact标签关联直接归因。',
        '在0秒，最大动作差对应左肩Pitch，投影后的真实PD目标仍差0.001862rad；右髋Yaw和右踝Pitch的执行目标分别差约−0.000543/−0.000608rad。它们确实进入执行目标，并非全被投影抹掉，但没有单关节干预证明哪一个是唯一触发源。',
        'WM在2.00s已经出现>50N的足部力瞬时差异；8.62s第一次支撑标签不同：左足垂直力model25约1.09N、model26约191.73N，接触点4→7，足部位置最大差仅约1.56mm。因此这次差异不仅是跨越5N统计阈值，也包含真实接触载荷变化。尚不能据此把第一次50N差异定义为唯一原因。',
        'raw也在1.96s出现>50N足部力差异，进一步反驳“raw没有同等闭环敏感性”的简单解释。当前只能说净Yaw误差的方向不同，不能用一个模式的平均动作MAE预测另一个模式的结果。',
        '新增执行/力曲线：`analysis/plots/{raw,wm}_contact_and_targets.png`；实际Yaw及参考曲线：`{raw,wm}_yaw_and_reference.png`；关键力、投影和PD差异：`analysis/contact_key_states.json`。',
        '回归检查：原仿真自动评估系统15项单元测试通过；新增六条完整轨迹、四条同前缀分叉及只读审计均完成。未启动任何新训练。']
    text='\n'.join(lines)+'\n'
    (root/'diagnosis.md').write_text(text)
    (ROOT/'docs/heading300_first_actor_update_diagnosis_20261012.md').write_text(text)
    (root/'COMPLETE').write_text('read-only diagnosis; training not started; heading300 remains incumbent')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)
