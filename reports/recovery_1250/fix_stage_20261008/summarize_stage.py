import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;rows=[];traces={}
for directory,label in [('control_unprojected','gain1'),('projection_ablation','gain1'),('projection_ablation','gain05'),('pilot_100_motion77','pilot100'),('heading_300_motion77','heading300'),('plane2048_50_motion77','plane50'),('plane2048_100_motion77','plane100')]:
 p=r/directory;assert (p/'COMPLETE').exists()
 for z in json.loads((p/'results.json').read_text()):
  if z['label']!=label:continue
  frames=[json.loads(l) for l in Path(z['trace']).read_text().splitlines()];q=np.asarray([f['joint'] for f in frames]);ref=np.asarray([f['clean'] for f in frames]);rp=np.asarray([f['rpy'] for f in frames]);e=q-ref[:,8:];ye=np.arctan2(np.sin(rp[:,2]-ref[:,3]),np.cos(rp[:,2]-ref[:,3]));mask=(ref[:,0]<.7)&(ref[:,[11,17]].mean(-1)>.8);target=np.asarray([f['pd_target'] for f in frames]);key=directory+'/'+label
  d=dict(configuration=key,mode=z['mode'],joint_rmse=z['joint_rmse'],leg_rmse=float(np.sqrt(np.mean(e[:,:12]**2))),velocity_rmse=z['root_velocity_rmse'],yaw_degrees=float(np.degrees(z['yaw_error'])),squat_right_hip_yaw_degrees=float(np.degrees(np.sqrt(np.mean(e[mask,8]**2)))),squat_left_knee_degrees=float(np.degrees(np.sqrt(np.mean(e[mask,3]**2)))),squat_right_knee_degrees=float(np.degrees(np.sqrt(np.mean(e[mask,9]**2)))),knee_target_negative_pct=float(np.mean(target[:,[3,9]]<0)*100),fall=z['fall_proxy']);rows.append(d);traces[key,z['mode']]=(ref,np.sqrt(np.mean(e[:,:12]**2,axis=1)),abs(np.degrees(ye)),np.degrees(e[:,8]))
for (k,m),(ref,*_) in traces.items():assert np.array_equal(ref,traces['control_unprojected/gain1',m][0])
with (r/'stage_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(r/'stage_results.json').write_text(json.dumps(rows,indent=2))
fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
for col,mode in enumerate(['raw','wm']):
 for key,name,color in [('control_unprojected/gain1','Original23500','#b22222'),('projection_ablation/gain05','Projection+gain0.5','#e69f00'),('heading_300_motion77/heading300','Heading300','#0072b2')]:
  _,leg,yaw,hip=traces[key,mode];t=np.arange(len(leg))*.02
  for j,x in enumerate([leg,yaw,abs(hip)]):axes[j,col].plot(t[49:],np.convolve(x,np.ones(50)/50,mode='valid'),label=name,color=color,lw=1)
 axes[0,col].set_title(mode.upper());axes[0,col].legend(fontsize=8);axes[-1,col].set_xlabel('Time (s)')
for j,title in enumerate(['Leg RMSE (rad)','Absolute heading error (deg)','Right hip yaw error (deg)']):
 axes[j,0].set_ylabel(title)
 for ax in axes[j]:ax.grid(alpha=.2)
fig.suptitle('mocap1/77: synchronized reference, 1-second moving averages');fig.tight_layout();fig.savefig(r/'before_after.png',dpi=180)
sel=dict(status='Experimental77 candidate; not a general deployment replacement',checkpoint=str((r.parents[2]/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt').resolve()),jit=str((r/'heading_300.pt').resolve()),jit_sha256=hashlib.sha256((r/'heading_300.pt').read_bytes()).hexdigest(),input_dim=7003,adapter_gain=.5,target_projection_required=True,projection_margin=.02,projection_knee_min=0.,heading_context='sin(error),1-cos(error)',evaluation='seed42 only; full77 raw/wm and8 development motions',hardware_trial=False)
(r/'candidate.json').write_text(json.dumps(sel,ensure_ascii=False,indent=2))
print('Saved comparison, chart and experimental candidate receipt')
