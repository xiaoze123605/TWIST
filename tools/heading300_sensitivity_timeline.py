"""Read-only temporal ordering and fixed-policy state/history decomposition."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_heading300_sensitivity import arrays,first,OUT
from sim_eval_metrics import phase_masks,rms
s=json.loads((OUT/'evaluation_retry/summary.json').read_text());results={}
for mode in ['raw','wm']:
 rows={r['label']:r for r in s['rows'] if r['reference_mode']==mode};frames={k:[json.loads(x) for x in Path(r['trace']).read_text().splitlines()] for k,r in rows.items()};a,b=arrays(frames['heading_300']),arrays(frames['hard26']);ref=a['ref'];assert np.array_equal(a['ref'],b['ref'])
 assert np.array_equal(np.array([f['processed'] for f in frames['heading_300']]),np.array([f['processed'] for f in frames['hard26']]))
 history_gap=0
 if mode=='wm':
  for old,new in zip(frames['heading_300'],frames['hard26']):
   assert old['reference_frame_index']==new['reference_frame_index']==old['frame_id'];assert np.array_equal(old['reference_wm_history'],new['reference_wm_history']);assert old['reference_wm_ready']==new['reference_wm_ready']
 source=np.load(OUT/'analysis'/f'heading_300_{mode}_components.npz');new=np.load(OUT/'analysis'/f'hard26_{mode}_components.npz');direct=new['student_action']-new['teacher_action'];feedback=new['teacher_action']-source['teacher_action'];total=new['student_action']-source['teacher_action'];assert abs(direct+feedback-total).max()<1e-6
 obs0=source['actor_input'];obs1=new['actor_input'];paired_errors={}
 for label,data,ff in [('heading_300',obs0,frames['heading_300']),('hard26',obs1,frames['hard26'])]:
  hist=data[:,1155:7001].reshape(-1,79,74);index=np.r_[31:36,36:59,59:82];state_error=float(abs(hist[1:,-1,:51]-data[:-1,index]).max());action_error=float(abs(hist[1:,-1,51:]-np.array([f['final_action'] for f in ff[:-1]])).max());assert state_error==0 and action_error<1e-6;paired_errors[label]=dict(state_history_pair_max_error=state_error,action_history_pair_max_error=action_error)
 gap=np.degrees(np.arctan2(np.sin(b['yaw']-a['yaw']),np.cos(b['yaw']-a['yaw'])));jg=abs(b['q']-a['q']).max(1);fg=abs(b['force']-a['force']).max(1);events={}
 for name,mask,p in [('joint_1e3rad',jg>1e-3,1),('force50N',fg>50,1),('contact_label',np.any(a['contact']!=b['contact'],axis=1),1),('yaw1deg',abs(gap)>1,10),('yaw5deg',abs(gap)>5,10),('yaw10deg',abs(gap)>10,10),('action_exceed',abs(direct).max(1)>1e-4,1)]:
  idx=first(mask,p);events[name]=dict(time=None if idx is None else idx*.02,phases=[] if idx is None else [k for k,m in phase_masks(ref).items() if m[idx]])
 results[mode]=dict(events=events,references_bitwise_identical=True,WM_history_bitwise_identical=True,history_alignment=paired_errors,direct_action_rms=float(rms(direct)),fixed_teacher_state_history_effect_rms=float(rms(feedback)),closed_loop_action_difference_rms=float(rms(total)),stage_yaw={k:{label:float(np.degrees(rms(np.arctan2(np.sin(z['yaw'][m]-ref[m,3]),np.cos(z['yaw'][m]-ref[m,3]))))) for label,z in [('heading300',a),('hard26',b)]} for k,m in phase_masks(ref).items()})
 t=np.arange(len(ref))*.02;fig,ax=plt.subplots(4,1,figsize=(13,11),sharex=True)
 for label,z in [('heading300',a),('hard26',b)]:ax[0].plot(t,np.degrees(np.arctan2(np.sin(z['yaw']-ref[:,3]),np.cos(z['yaw']-ref[:,3]))),label=label)
 ax[0].set_ylabel('Yaw tracking error (deg)')
 for label,z in [('same-input policy drift',direct),('fixed-teacher state/history effect',feedback),('closed-loop action gap',total)]:ax[1].plot(t,abs(z).max(1),label=label)
 ax[1].set_yscale('symlog',linthresh=1e-5);ax[1].set_ylabel('Action difference max')
 ax[2].plot(t,fg,label='Foot vertical force difference (N)');ax[3].plot(t,jg,label='Joint gap max (rad)');ax[3].plot(t,abs(b['target']-a['target']).max(1),label='PD target gap max (rad)')
 for z in ax:z.legend();z.grid(alpha=.2)
 ax[-1].set_xlabel('Reference time (s)');fig.tight_layout();fig.savefig(OUT/'analysis/plots'/f'{mode}_full_timeline.png',dpi=150);plt.close(fig)
(OUT/'analysis/timeline_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
