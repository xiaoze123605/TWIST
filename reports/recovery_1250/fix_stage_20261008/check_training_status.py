import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
runs={'original':('recovery_1250/coordinated_1250_seed42_20260926',[(9600,10000),(19600,20000),(23100,23500),(29600,30000)]),'heading':('recovery_23500_heading/heading23500_gain05_pilot100_20261008',[(50,100),(150,200),(250,300)]),'plane':('recovery_23500_plane/plane23500_from300_pilot100_env2048_20261008',[(25,50),(75,100)])}
result={}
for label,(directory,windows) in runs.items():
 root=ROOT/'legged_gym/logs'/directory
 data={}
 for filename,stepkey in [('train_metrics.jsonl','completed_updates'),('anchor_metrics.jsonl','update')]:
  p=root/filename
  if not p.exists():continue
  records={}
  for line in p.read_text().splitlines():
   try:x=json.loads(line);records[x[stepkey]]=x
   except (ValueError,KeyError):continue
  keys=['Train/mean_reward','Train/mean_episode_length','Loss/value_func','Episode_rew_metrics/metric_support_slip','Episode_rew_metrics/metric_heading_tracking','Episode_rew_metrics/metric_hip_yaw_tracking','Episode_rew_metrics/metric_squat_knee_tracking','Episode_rew_metrics/metric_leg_plane_alignment','source_action_mae','max_rollout_kl','skipped_minibatches','actor_lr','anchor_coef_effective','posture_teacher_weight_mean']
  summary=[]
  for lo,hi in windows:
   subset=[x for step,x in records.items() if lo<step<=hi]
   if len(subset)<hi-lo:continue
   stats={k:{'mean':float(np.mean([x[k] for x in subset if k in x])),'std':float(np.std([x[k] for x in subset if k in x]))} for k in keys if any(k in x for x in subset)}
   summary.append({'updates':[lo+1,hi],'n':len(subset),'metrics':stats})
  data[filename]={'latest_update':max(records,default=0),'windows':summary}
 data['latest_checkpoint']=str(max(root.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]),default=root))
 result[label]=data
Path(__file__).with_name('training_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
