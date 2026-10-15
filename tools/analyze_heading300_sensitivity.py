"""Offline full-teacher probes, normalized pool proximity and controlled-fork analysis."""
import json,sys,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
class NearestNeighbors:
    # Exact blocked nearest-neighbor calculation using installed Torch only.
    def __init__(self,n_neighbors=2,**kwargs):self.k=n_neighbors
    def fit(self,x):
        self.x=torch.as_tensor(x,dtype=torch.float64);self.sq=(self.x*self.x).sum(1);return self
    def kneighbors(self,x,n_neighbors=None,return_distance=True):
        k=n_neighbors or self.k;values=[];indices=[]
        with torch.no_grad():
            for a in torch.as_tensor(x,dtype=torch.float64).split(256):
                squared=((a*a).sum(1)[:,None]+self.sq[None,:]-2*a@self.x.T).clamp_min(0)
                v,i=squared.topk(k,dim=1,largest=False);values.append(v.sqrt().numpy());indices.append(i.numpy())
        return np.concatenate(values),np.concatenate(indices)

from analyze_heading300_first_update import outputs,load_row,csv_write
from export_twist_anyadapter_opentrack_jit import build_actor
from sim_eval_metrics import phase_masks,rms,segments
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/heading300_sensitivity_20261013'
SOURCE=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
NEW=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/hard1e4_adam_env512_seed42_20261012/model_26.pt'
POOL=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_25.pt'

def first(mask,persist=1):return next((int(a) for a,b in segments(mask) if b-a>=persist),None)
def delta_stats(x):
    a=np.abs(x);return dict(max_abs=float(a.max()),mean_abs=float(a.mean()),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),states_over_1e4=int((a.max(-1)>1e-4).sum()),states=len(x))
def arrays(frames):
    ref=np.array([x['clean'] for x in frames]);q=np.array([x['joint'] for x in frames]);yaw=np.array([x['rpy'][2] for x in frames]);force=np.array([x['dynamics']['foot_vertical_force'] for x in frames]);action=np.array([x['final_action'] for x in frames]);target=np.array([x['pd_target'] for x in frames]);contact=force>5
    return dict(ref=ref,q=q,yaw=yaw,force=force,action=action,target=target,contact=contact)
def actors():return build_actor(SOURCE),build_actor(NEW)

def full():
    torch.set_num_threads(1);old,new=actors();summary=json.loads((OUT/'evaluation_retry/summary.json').read_text());assert summary['status']=='complete'
    dest=OUT/'analysis';dest.mkdir(exist_ok=False);(dest/'plots').mkdir();groups=[];allrows=[];bank=torch.load(POOL,map_location='cpu')['recovery1250']['replay']['groups'];x=torch.cat([bank[g]['inputs'] for g in sorted(bank,key=int)]).numpy();features=np.c_[x[:,:105],x[:,1155:]]
    mean=features.mean(0);scale=features.std(0);active=scale>1e-6;normalized=(features[:,active]-mean[active])/scale[active];nn=NearestNeighbors(n_neighbors=2,algorithm='brute',n_jobs=1).fit(normalized)
    calibration=nn.kneighbors(normalized,return_distance=True)[0][:,1]/np.sqrt(active.sum());cut=float(np.quantile(calibration,.99));np.savez_compressed(dest/'pool_normalization.npz',mean=mean,scale=scale,active=active,calibration=calibration)
    poolreceipts=[]
    with torch.no_grad():
      for g in sorted(bank,key=int):
        p=bank[g]['inputs'];a=old.layerwise_actor(p[:,:1155],p[:,1155:1283])+old.heading_delta(p[:,1155:1283],p[:,-2:]);b=new.layerwise_actor(p[:,:1155],p[:,1155:1283])+new.heading_delta(p[:,1155:1283],p[:,-2:]);poolreceipts.append(dict(group=int(g),**delta_stats((b-a).numpy())))
    csv_write(dest/'pool_drift.csv',poolreceipts)
    full_results={}
    for row in summary['rows']:
      label,mode=row['label'],row['reference_mode'];frames,obs=load_row(row);actual=arrays(frames);teacher=outputs(old,obs);student=outputs(new,obs);chosen=teacher if label=='heading_300' else student;recorded=np.array([f['dynamics']['policy_action_pre_ema'] for f in frames]);assert abs(chosen['full']-recorded).max()<1e-5
      delta=student['full']-teacher['full'];context=obs[:,-2:];heading=np.degrees(abs(np.arctan2(context[:,0],1-context[:,1])));masks={'all':np.ones(len(obs),bool),**phase_masks(actual['ref'])};contact=actual['contact'];masks.update(double_support=contact.sum(1)==2,single_support=contact.sum(1)==1,no_support=contact.sum(1)==0)
      switch=np.r_[False,np.any(contact[1:]!=contact[:-1],axis=1)];masks['contact_switch_near']=binary_dilation(switch,iterations=2)
      for lo,hi in [(0,15),(15,30),(30,60),(60,120),(120,181)]:masks[f'heading_{lo}_{hi}']=(heading>=lo)&(heading<hi)
      feat=np.c_[obs[:,:105],teacher['latent'],context];norm=(feat[:,active]-mean[active])/scale[active];dist=nn.kneighbors(norm,n_neighbors=1)[0][:,0]/np.sqrt(active.sum());drift=abs(delta).max(1)
      for name,m in masks.items():
        if m.any():groups.append(dict(label=label,mode=mode,group=name,**delta_stats(delta[m]),near_pool_distance_p50=float(np.median(dist[m])),near_pool_distance_p99=float(np.quantile(dist[m],.99)),above_pool_loo_p99_fraction=float((dist[m]>cut).mean())))
      for i,f in enumerate(frames):allrows.append(dict(label=label,mode=mode,frame=i,time=i*.02,max_action_drift=float(drift[i]),pool_scaled_distance=float(dist[i]),above_pool_calibration=bool(dist[i]>cut),heading_error_degrees=float(heading[i]),double_support=bool(contact[i].sum()==2),**{k:bool(v[i]) for k,v in phase_masks(actual['ref']).items()}))
      np.savez_compressed(dest/f'{label}_{mode}_components.npz',actor_input=obs,teacher_core=teacher['core'],teacher_heading=teacher['head'],teacher_action=teacher['full'],student_core=student['core'],student_heading=student['head'],student_action=student['full'],same_input_delta=delta,pool_distance=dist)
      full_results[label+'_'+mode]=dict(**delta_stats(delta),first_exceed_time=None if first(drift>1e-4) is None else first(drift>1e-4)*.02,recorded_eager_error=float(abs(chosen['full']-recorded).max()),metrics=row['metrics'],trace=row['trace'])
      fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True);axes[0].plot(np.arange(len(obs))*.02,drift);axes[0].axhline(1e-4,color='r',ls='--');axes[0].set_ylabel('Same-input full action max deviation');axes[1].plot(np.arange(len(obs))*.02,dist);axes[1].axhline(cut,color='r',ls='--');axes[1].set_ylabel('Pool-scaled nearest distance (RMS)');axes[1].set_xlabel('Reference time (s)');fig.tight_layout();fig.savefig(dest/'plots'/f'{label}_{mode}_coverage.png',dpi=150);plt.close(fig)
    csv_write(dest/'full_state_groups.csv',groups);csv_write(dest/'full_frames.csv',allrows)
    (dest/'full_results.json').write_text(json.dumps(dict(results=full_results,pool_calibration_p99=cut,active_features=int(active.sum()),feature_definition='current base105 + frozen latent128 + heading2; each nonconstant feature standardized with pool mean/std; distance per-feature RMS; pool leave-one-out second neighbor P99 is descriptive sparse-neighborhood threshold, not proof of training OOD',pool_contact_labels_missing=True),indent=2))
    print('FULL ANALYSIS COMPLETE',flush=True)

def forks():
    torch.set_num_threads(1);old,new=actors();dest=OUT/'analysis';entries=json.loads((OUT/'fork_retry/results.json').read_text());rows=[]
    for mode in ['raw','wm']:
      for pivot in [0,8,12,18]:
        selection=[e for e in entries if e['mode']==mode and e['pivot_seconds']==pivot];base=next(e for e in selection if e['label']=='A');bf=[json.loads(x) for x in Path(base['trace']).read_text().splitlines()][int(pivot/.02):];ba=arrays(bf);fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
        for e in selection:
          frames=[json.loads(x) for x in Path(e['trace']).read_text().splitlines()][int(pivot/.02):];a=arrays(frames);obs=np.load(e['actor_inputs']);assert obs.shape==(250,7003);tc=outputs(old,obs);sc=outputs(new,obs);d=sc['full']-tc['full'];drift=abs(d).max(1);gap=np.degrees(np.arctan2(np.sin(a['yaw']-ba['yaw']),np.cos(a['yaw']-ba['yaw'])));jgap=abs(a['q']-ba['q']).max(1);fgap=abs(a['force']-ba['force']).max(1);target=abs(a['target']-ba['target']).max(1);switch=np.any(a['contact']!=ba['contact'],axis=1)
          r=dict(mode=mode,pivot=pivot,label=e['label'],window_seconds=5,initial_full_action_drift=float(drift[0]),**delta_stats(d),yaw_gap_max=float(abs(gap).max()),yaw_gap_rms=float(rms(gap)),yaw_gap_final=float(gap[-1]),yaw_gap_first_1s_max=float(abs(gap[:50]).max()),yaw_gap_last_1s_max=float(abs(gap[-50:]).max()),joint_gap_max=float(jgap.max()),foot_force_gap_max_N=float(fgap.max()),pd_target_gap_max=float(target.max()))
          for name,mask,persist in [('joint_1e3rad',jgap>1e-3,1),('force_50N',fgap>50,1),('contact_label',switch,1),('yaw_1deg',abs(gap)>1,10),('yaw_5deg',abs(gap)>5,10),('yaw_10deg',abs(gap)>10,10),('action_exceed',drift>1e-4,1)]:
            index=first(mask,persist);r['first_'+name+'_seconds']=None if index is None else index*.02
          if e['label'] in ['C','D']:
            end=e['duration_steps'];post=abs(gap[end:]);peak=float(post.max());last=float(abs(gap[-50:]).mean());r['post_switch_peak_degrees']=peak;r['post_switch_last1s_mean_degrees']=last;r['recovery']='negligible (<1deg)' if peak<1 else 'decayed >50% from peak' if last<peak*.5 else 'persisted/amplified'
          rows.append(r);time_axis=np.arange(250)*.02
          for ax,val in zip(axes,[gap,fgap,jgap,drift]):ax.plot(time_axis,val,label=e['label'])
          np.savez_compressed(Path(e['trace']).parent/'teacher_probe.npz',teacher_action=tc['full'],student_action=sc['full'],same_input_delta=d)
        for ax,name in zip(axes,['Yaw difference to A (deg)','Foot vertical force difference max (N)','Joint difference max (rad)','Same-input student-teacher max action drift']):ax.set_ylabel(name);ax.legend();ax.grid(alpha=.2)
        axes[-1].axhline(1e-4,color='r',ls='--');axes[-1].set_xlabel('Time after fork (s)');fig.suptitle(f'{mode} fork at {pivot}s; A baseline B continuous C 20ms D 200ms');fig.tight_layout();fig.savefig(dest/'plots'/f'{mode}_fork_{pivot}.png',dpi=150);plt.close(fig)
    csv_write(dest/'fork_comparison.csv',rows);(dest/'fork_results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':
    if '--forks' in sys.argv:forks()
    else:full()
