"""Static weighted reset-time coverage; does not claim realized on-policy exposure."""
import sys,pickle,json,csv
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d
ROOT=Path(__file__).resolve().parents[1]
sys.modules['numpy._core']=np.core;sys.modules['numpy._core.multiarray']=np.core.multiarray;sys.modules['numpy._core.numeric']=np.core.numeric
config=ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml'
cfg=yaml.safe_load(config.read_text());data_root=Path(cfg['root_path']);rows=[]
for i,entry in enumerate(cfg['motions']):
 with (data_root/entry['file']).open('rb') as f:d=pickle.load(f)
 fps=float(d['fps']);root=np.asarray(d['root_pos']);q=np.asarray(d['dof_pos']);h=root[:,2];knee=q[:,[3,9]].mean(-1)
 squat=(h<.7)&(knee>.8)
 hdot=np.gradient(uniform_filter1d(h,size=max(3,round(fps*.15)),mode='nearest'))*fps
 region=(h<.78)&(knee>.5);descent=region&(hdot<-.03);rise=region&(hdot>.03)
 links=list(d['link_body_list']);feet=np.asarray(d['local_body_pos'])[:,[links.index('left_ankle_roll_link'),links.index('right_ankle_roll_link')]]
 # MotionLib consumes root-local body points; match global conversion.
 rot=Rotation.from_quat(np.asarray(d['root_rot']));feet=np.einsum('nij,nkj->nki',rot.as_matrix(),feet)+root[:,None,:]
 vel=np.diff(feet,axis=0,append=feet[-1:])*fps
 confidence=np.clip((.20-np.linalg.norm(vel[:,:,:2],axis=-1))/.15,0,1)*np.clip((.12-abs(vel[:,:,2]))/.08,0,1)*np.clip((.15-feet[:,:,2])/.07,0,1)
 root_v=np.gradient(root,axis=0)*fps
 yaw=rot.as_euler('xyz')[:,2];yaw_v=np.gradient(np.unwrap(yaw))*fps
 inplace=(np.linalg.norm(root_v[:,:2],axis=-1)<.12)&(abs(yaw_v)<.12)&(confidence.min(-1)>.5)
 rows.append(dict(file=entry['file'],weight=float(entry['weight']),frames=len(h),seconds=(len(h)-1)/fps,squat=float(squat.mean()),descent=float(descent.mean()),rise=float(rise.mean()),squat_stance=float((squat&(confidence.min(-1)>.5)).mean()),squat_inplace=float((squat&inplace).mean()),stance_confidence=float(confidence.mean()),squat_confidence=float((squat*confidence.mean(-1)).mean())))
 if (i+1)%2000==0:print('Read',i+1,flush=True)
w=np.array([x['weight'] for x in rows]);w/=w.sum()
summary=dict(motions=len(rows),config=str(config),definition=dict(squat='reference root height<0.7m and mean knees>0.8rad',descent_rise='height<0.78m, mean knees>0.5rad, smoothed height velocity magnitude>0.03m/s',sampling='motion weight then uniform reference reset time; approximate native-frame statistics; survival and episode progression not measured'),weighted_reset_fraction={k:float(sum(p*x[k] for p,x in zip(w,rows))) for k in ['squat','descent','rise','squat_stance','squat_inplace','squat_confidence']},motion77=next((dict(x,reset_probability=float(w[i])) for i,x in enumerate(rows) if x['file']=='mocap1/77.pkl'),None),top_squat_contributors=sorted([dict(file=x['file'],reset_probability=float(p),contribution=float(p*x['squat'])) for p,x in zip(w,rows)],key=lambda x:x['contribution'],reverse=True)[:20])
out=ROOT/'reports/heading300_diagnosis_20261008';out.mkdir(exist_ok=True)
(out/'sampling_audit.json').write_text(json.dumps(summary,indent=2))
with (out/'sampling_by_motion.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2))
