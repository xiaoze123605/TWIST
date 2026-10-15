"""Same-input policy probes and aligned-reference closed-loop diagnosis; no learning."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
from export_twist_anyadapter_opentrack_jit import build_actor
from heading300_first_update_verify import RUN,SOURCE,sha
from sim_eval_metrics import phase_masks,rms,segments


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def masks_from_data(d):
    context=np.asarray(d['context']);angle=np.degrees(np.abs(np.arctan2(context[:,0],1-context[:,1])))
    turn=np.asarray(d['turn']);contact=np.asarray(d['contact']);near=np.asarray(d['near_switch'])
    groups={'all':np.ones(len(context),dtype=bool),'double_support':contact.sum(-1)==2,
        'single_support':contact.sum(-1)==1,'no_support':contact.sum(-1)==0,
        'contact_switch_near':near,'contact_switch_far':~near,
        'squat':np.asarray(d['squat']),'walking':np.asarray(d['walking']),
        'turn_gt_0.5':turn>.5,'turn_le_0.5':turn<=.5}
    for lo,hi in [(0,15),(15,30),(30,60),(60,120),(120,181)]:groups[f'heading_{lo}_{hi}']=(angle>=lo)&(angle<hi)
    groups['wm_turn_single_switch']=(turn>.5)&(contact.sum(-1)==1)&near
    groups['turn_heading_gt30']=(turn>.5)&(angle>=30)
    return groups


@torch.no_grad()
def outputs(actor,observations):
    values={k:[] for k in ['base','core','adapter','head','full','latent']}
    for start in range(0,len(observations),256):
        x=torch.from_numpy(np.array(observations[start:start+256],copy=True)).float()
        b,h=actor.split_obs(x);z=actor.history_encoder(h)
        base=actor.layerwise_actor.base_forward(b);core=actor.layerwise_actor(b,z)
        head=actor.heading_delta(z,x[:,-2:])
        for key,v in dict(base=base,core=core,adapter=core-base,head=head,full=core+head,latent=z).items():values[key].append(v.numpy())
    return {k:np.concatenate(v) for k,v in values.items()}


def first_run(mask,length=10):
    # 0.2s persistence, declared diagnostic threshold; not a new evaluation metric.
    return next((int(a) for a,b in segments(mask) if b-a>=length),None)


def load_row(row):
    frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()]
    obs=np.load(frames[0]['actor_observation_file'],mmap_mode='r')
    assert obs.shape==(3207,7003)
    assert [f['frame_id'] for f in frames]==list(range(3207))
    assert all(f['reference_frame_index']==f['frame_id']==f['actor_observation_index'] for f in frames)
    return frames,obs


def analyze(root):
    out=root/'analysis';out.mkdir(exist_ok=False);(out/'plots').mkdir()
    summary=json.loads((root/'evaluation/summary.json').read_text());assert summary['status']=='complete'
    actors={k:build_actor(p) for k,p in [('heading_300',SOURCE),('model25',RUN/'model_25.pt'),('model26',RUN/'model_26.pt')]}
    rows={(r['label'],r['reference_mode']):r for r in summary['rows']}
    index=np.r_[np.arange(31,36),np.arange(36,59),np.arange(59,82)]
    results={};group_rows=[];key_rows=[];component_receipts=[]
    for mode in ['raw','wm']:
        data={}
        for label in ['heading_300','model25','model26']:
            row=rows[label,mode];frames,obs=load_row(row);comp=outputs(actors[label],obs)
            action=np.array([f['dynamics']['policy_action_pre_ema'] for f in frames])
            eager_error=float(np.max(abs(comp['full']-action)));assert eager_error<1e-5
            consumed=obs[:,1155:7001].reshape(-1,79,74)
            history_error=float(np.max(abs(consumed[1:,-1,:51]-obs[:-1,index])))
            action_history_error=float(np.max(abs(consumed[1:,-1,51:]-np.array([f['final_action'] for f in frames[:-1]]))))
            assert history_error==0 and action_history_error<1e-6
            component_receipts.append(dict(label=label,mode=mode,eager_vs_recorded_pre_ema_max=eager_error,
                history_state_t_action_t_pair_error=history_error,history_action_pair_error=action_history_error))
            np.savez_compressed(out/f'{label}_{mode}_components.npz',**comp)
            data[label]=dict(frames=frames,obs=obs,comp=comp)
        old,new,baseline=data['model25'],data['model26'],data['heading_300']
        for field in ['qpos','qvel','rpy','joint','processed','clean','final_action','pd_target']:
            assert np.array_equal(np.array([f[field] for f in old['frames']]),np.array([f[field] for f in baseline['frames']])),field
        assert np.array_equal(old['obs'],baseline['obs'])
        clean=np.array([f['clean'] for f in old['frames']]);proc=np.array([f['processed'] for f in old['frames']])
        new_clean=np.array([f['clean'] for f in new['frames']]);new_proc=np.array([f['processed'] for f in new['frames']])
        assert np.array_equal(clean,new_clean) and np.array_equal(proc,new_proc)
        wm_history_max=0.
        if mode=='wm':
            for a,b in zip(old['frames'],new['frames']):
                assert a['reference_wm_ready']==b['reference_wm_ready']
                x=np.asarray(a['reference_wm_history']);y=np.asarray(b['reference_wm_history'])
                wm_history_max=max(wm_history_max,float(np.max(abs(x-y))))
                assert len(x)==min(a['frame_id']+1,25)
                assert np.max(abs(x[-1,np.r_[0:3,4:31]]-clean[a['frame_id'],np.r_[0:3,4:31]]))<1e-6
            assert wm_history_max==0
        # Evaluate both policies on one and the same observation. Separate the
        # functional policy change from state/history feedback after trajectories diverge.
        counter=outputs(actors['model26'],old['obs'])
        direct=counter['full']-old['comp']['full']
        feedback=new['comp']['full']-counter['full']
        actual_difference=new['comp']['full']-old['comp']['full']
        assert np.max(abs(direct+feedback-actual_difference))<1e-6
        core_direct=counter['core']-old['comp']['core'];head_direct=counter['head']-old['comp']['head']
        yaw25=np.array([f['rpy'][2] for f in old['frames']]);yaw26=np.array([f['rpy'][2] for f in new['frames']])
        yaw_gap=np.arctan2(np.sin(yaw26-yaw25),np.cos(yaw26-yaw25))
        err25=np.arctan2(np.sin(yaw25-clean[:,3]),np.cos(yaw25-clean[:,3]))
        err26=np.arctan2(np.sin(yaw26-clean[:,3]),np.cos(yaw26-clean[:,3]))
        contact=np.array([f['dynamics']['foot_contact'] for f in old['frames']]);new_contact=np.array([f['dynamics']['foot_contact'] for f in new['frames']])
        switch=np.r_[False,np.any(contact[1:]!=contact[:-1],axis=-1)]
        near=binary_dilation(switch,iterations=2);phases=phase_masks(clean)
        context=np.asarray(old['obs'][:,-2:]);groups=masks_from_data(dict(context=context,turn=abs(proc[:,7]),contact=contact,near_switch=near,squat=phases['squat'],walking=phases['walking']))
        for name,mask in groups.items():
            if not mask.any():continue
            group_rows.append(dict(mode=mode,group=name,frames=int(mask.sum()),fraction=float(mask.mean()),
                direct_action_rms=rms(direct[mask]),direct_leg_rms=rms(direct[mask,:12]),
                direct_core_rms=rms(core_direct[mask]),direct_heading_rms=rms(head_direct[mask]),
                closed_loop_action_gap_rms=rms(actual_difference[mask]),feedback_component_rms=rms(feedback[mask]),
                yaw25_rmse_deg=math.degrees(rms(err25[mask])),yaw26_rmse_deg=math.degrees(rms(err26[mask])),
                caution='Groups aligned by baseline reference/support; post-divergence gap is association, not direct policy effect'))
        q25=np.array([f['joint'] for f in old['frames']]);q26=np.array([f['joint'] for f in new['frames']])
        dq25=np.array([f['joint_velocity'] for f in old['frames']]);dq26=np.array([f['joint_velocity'] for f in new['frames']])
        qgap=np.max(abs(q26-q25),axis=-1);vgap=np.max(abs(dq26-dq25),axis=-1)
        actiongap=np.max(abs(direct),axis=-1);fullgap=np.max(abs(actual_difference),axis=-1)
        observation_gap=np.max(abs(new['obs']-old['obs']),axis=-1)
        events=dict(policy_direct_gt1e_6=int(np.flatnonzero(actiongap>1e-6)[0]),
            actual_joint_gt1e_6=int(np.flatnonzero(qgap>1e-6)[0]),
            actual_joint_gt0_01rad=first_run(qgap>.01),
            contact_first_mismatch=int(np.flatnonzero(np.any(contact!=new_contact,axis=-1))[0]),
            yaw_gap_gt1deg=first_run(abs(yaw_gap)>np.deg2rad(1)),
            yaw_gap_gt5deg=first_run(abs(yaw_gap)>np.deg2rad(5)),
            yaw_gap_gt10deg=first_run(abs(yaw_gap)>np.deg2rad(10)))
        results[mode]=dict(model25_heading300_trajectory_exact=True,clean_difference_max=0.,processed_difference_max=0.,
            wm_history_difference_max=wm_history_max,first_events={k:dict(frame=v,time=v*.02 if v is not None else None,
                phases=[p for p,m in phases.items() if v is not None and m[v]]) for k,v in events.items()},
            direct_action_rms=rms(direct),direct_action_max=float(abs(direct).max()),
            feedback_component_rms=rms(feedback),closed_loop_action_gap_rms=rms(actual_difference),
            direct_core_rms=rms(core_direct),direct_head_rms=rms(head_direct),
            policy25_yaw_rmse_deg=math.degrees(rms(err25)),policy26_yaw_rmse_deg=math.degrees(rms(err26)),
            phase_yaw={p:dict(model25=math.degrees(rms(err25[m])),model26=math.degrees(rms(err26[m]))) for p,m in phases.items() if m.any()})
        frame_rows=[]
        for i in range(len(clean)):
            r=dict(frame=i,time=i*.02,yaw25_degrees=np.degrees(yaw25[i]),yaw26_degrees=np.degrees(yaw26[i]),
                reference_yaw_degrees=np.degrees(clean[i,3]),processed_yaw_degrees=np.degrees(proc[i,3]),
                yaw25_error_degrees=np.degrees(err25[i]),yaw26_error_degrees=np.degrees(err26[i]),yaw_gap_degrees=np.degrees(yaw_gap[i]),
                same_input_direct_max=actiongap[i],closed_loop_action_gap_max=fullgap[i],feedback_action_max=float(abs(feedback[i]).max()),
                actor_observation_max_difference=observation_gap[i],joint_max_difference=qgap[i],joint_velocity_max_difference=vgap[i],
                clean_reference_difference=0.,processed_reference_difference=0.,wm_history_difference=0.,
                support25=int(contact[i].sum()),support26=int(new_contact[i].sum()),near_contact_switch=bool(near[i]),
                phase=';'.join(p for p,m in phases.items() if m[i]))
            for j in range(23):r.update({f'joint{j}_direct':direct[i,j],f'joint{j}_core_direct':core_direct[i,j],f'joint{j}_head_direct':head_direct[i,j]})
            frame_rows.append(r)
        csv_write(out/f'{mode}_frames.csv',frame_rows)
        np.savez_compressed(out/f'{mode}_counterfactual.npz',direct=direct,core_direct=core_direct,head_direct=head_direct,
            feedback=feedback,actual_difference=actual_difference,yaw_gap=yaw_gap,q25=q25,q26=q26,
            context=context,latent=old['comp']['latent'],base=np.array(old['obs'][:,:1155]),
            contact=contact,near_switch=near,processed=proc,clean=clean)
        picks=sorted(set([0,1]+[int(v) for v in events.values() if v is not None]))
        for i in picks:
            r=frame_rows[i].copy();r['mode']=mode;r['top_direct_joints']=np.argsort(abs(direct[i]))[-5:][::-1].tolist();key_rows.append(r)
        t=np.arange(len(clean))*.02
        fig,ax=plt.subplots(4,1,figsize=(15,12),sharex=True)
        ax[0].plot(t,np.degrees(err25),label='model25=heading300');ax[0].plot(t,np.degrees(err26),label='model26');ax[0].set_ylabel('Wrapped yaw error (deg)');ax[0].legend()
        ax[1].plot(t,actiongap,label='same-input policy delta');ax[1].plot(t,fullgap,label='closed-loop action gap');ax[1].plot(t,np.max(abs(feedback),-1),label='fixed-policy state/history effect');ax[1].set_yscale('symlog',linthresh=1e-5);ax[1].legend();ax[1].set_ylabel('Action max abs')
        ax[2].plot(t,qgap,label='joint gap');ax[2].plot(t,observation_gap,label='observation gap (mixed units)');ax[2].plot(t,np.zeros_like(t),label='processed reference/WM history gap=0');ax[2].legend();ax[2].set_ylabel('Trajectory gap')
        ax[3].plot(t,contact.sum(-1),label='model25 support');ax[3].plot(t,new_contact.sum(-1),label='model26 support',alpha=.7);ax[3].set_ylabel('Contact feet');ax[3].set_xlabel('Reference-aligned time (s)');ax[3].legend()
        for a in ax:a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'plots'/f'{mode}_timeline.png',dpi=140);plt.close(fig)
        event=events['yaw_gap_gt5deg'];center=event*.02 if event is not None else 10.
        fig,ax=plt.subplots(6,1,figsize=(14,15),sharex=True)
        for a,j in zip(ax,[2,8,4,10,3,9]):
            a.plot(t,q25[:,j],label='actual25');a.plot(t,q26[:,j],label='actual26');a.plot(t,clean[:,8+j],label='clean ref',alpha=.6)
            a.set_ylabel(f'Joint {j} (rad)');a.legend(loc='upper right');a.grid(alpha=.2)
        ax[-1].set_xlim(max(0,center-2),min(64.14,center+2));ax[-1].set_xlabel('Reference-aligned time (s)')
        fig.tight_layout();fig.savefig(out/'plots'/f'{mode}_first_divergence_joints.png',dpi=140);plt.close(fig)
        print('Analyzed',mode,results[mode]['first_events'],flush=True)
    csv_write(out/'simulation_groups.csv',group_rows);csv_write(out/'key_states.csv',key_rows)
    (out/'results.json').write_text(json.dumps(dict(modes=results,input_and_jit_receipts=component_receipts,
        event_definitions='Yaw thresholds1/5/10deg and joint0.01rad persist>=10 frames=0.2s; numerical onset uses1e-6',
        caution='Direct effect uses identical observation. Feedback decomposition is an algebraic functional decomposition, not proof of any individual contact or joint causal mechanism.'),indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    analyze(p.parse_args().root)
