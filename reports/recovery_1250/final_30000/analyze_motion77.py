import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;winner=sys.argv[1];rows=[]
for name in ['motion77_initial','motion77_finalists']:
 d=root/name;assert (d/'COMPLETE').exists();rows+=json.loads((d/'results.json').read_text())
by={(x['label'],x['mode']):x for x in rows};table=[];traces={};windows=[]
for row in rows:
 x=[json.loads(l) for l in Path(row['trace']).read_text().splitlines()];clean=np.asarray([f['clean'] for f in x]);q=np.asarray([f['joint'] for f in x]);rpy=np.asarray([f['rpy'] for f in x]);vel=np.asarray([f['root_velocity'] for f in x]);err=q-clean[:,8:];yaw=np.arctan2(np.sin(rpy[:,2]-clean[:,3]),np.cos(rpy[:,2]-clean[:,3]));leg=np.sqrt(np.mean(err[:,:12]**2,axis=1));ve=np.sqrt(np.mean((vel-clean[:,4:7])**2,axis=1));traces[row['label'],row['mode']]=(clean,leg,yaw,ve,np.max(abs(rpy[:,:2]),axis=1));assert len(x)==3207
 d={k:row[k] for k in ['label','mode','joint_rmse','root_velocity_rmse','yaw_error','max_tilt','minimum_pelvis_height','fall_proxy']};d['yaw_rmse_deg']=float(np.degrees(row['yaw_error']));d['leg_rmse']=float(np.sqrt(np.mean(err[:,:12]**2)));d['arm_rmse']=float(np.sqrt(np.mean(err[:,15:]**2)));d['right_hip_yaw_rmse']=float(np.sqrt(np.mean(err[:,8]**2)))
 for base in ['baseline','ckpt4900','reference1250']:
  b=by[base,row['mode']];d['composite_gain_vs_'+base]=100*(1-sum(w*row[k]/b[k] for w,k in [(0.4,'joint_rmse'),(.3,'root_velocity_rmse'),(.3,'yaw_error')]))
 table.append(d)
 for start,stop in [(0,10),(10,20),(20,40),(40,64.14)]:
  sl=slice(round(start*50),round(stop*50));windows.append(dict(label=row['label'],mode=row['mode'],start=start,stop=stop,leg_rmse=float(np.sqrt(np.mean(err[sl,:12]**2))),velocity_rmse=float(np.sqrt(np.mean((vel[sl]-clean[sl,4:7])**2))),yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw[sl]**2))))))
for key,(ref,*_) in traces.items():assert np.array_equal(ref,traces['baseline',key[1]][0]),key
with (root/'motion77_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
(root/'motion77_analysis.json').write_text(json.dumps(dict(selected=winner,all_frames=3207,seconds=64.14,clean_references_identical=True,results=table,windows=windows),indent=2))
fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
for col,mode in enumerate(['raw','wm']):
 for label,color in [('baseline','black'),('ckpt4900','#dc7c23'),(winner,'#0072b2')]:
  clean,leg,yaw,vel,tilt=traces[label,mode];t=np.arange(len(leg))*.02
  for i,v in enumerate([leg,np.abs(np.degrees(yaw)),vel,np.degrees(tilt)]):
   smooth=np.convolve(v,np.ones(50)/50,mode='valid');axes[i,col].plot(t[49:],smooth,label=label,color=color,lw=1.1)
 axes[0,col].set_title(mode.upper());axes[0,col].legend(fontsize=8);axes[-1,col].set_xlabel('Time (s)')
for i,y in enumerate(['Leg RMSE (rad)','Absolute heading error (deg)','Velocity RMSE (m/s)','Max |roll,pitch| (deg)']):
 axes[i,0].set_ylabel(y)
 for ax in axes[i]:ax.grid(alpha=.2)
fig.suptitle('mocap1/77: identical clean reference, 1-second moving averages');fig.tight_layout();fig.savefig(root/'motion77_tracking_comparison.png',dpi=180)
print(json.dumps([x for x in table if x['label']==winner],indent=2))
