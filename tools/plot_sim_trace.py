"""Export diagnostic curves from synchronized 50-Hz MuJoCo trace files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

JOINTS = [s + '_' + j for s in ('left', 'right')
          for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
JOINTS += ['waist_yaw', 'waist_roll', 'waist_pitch']
JOINTS += [s + '_' + j for s in ('left', 'right')
           for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')]


def export(trace, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Trace contains no frames')
    ids = np.asarray([r['frame_id'] for r in rows])
    if len(ids) > 1 and not np.all(np.diff(ids) == 1):
        raise ValueError('Expected consecutive synchronized frame IDs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = ids * .02
    clean = np.asarray([r['clean'] for r in rows], dtype=float)
    ref = np.asarray([r['processed'] for r in rows], dtype=float)
    joint = np.asarray([r['joint'] for r in rows], dtype=float)
    rpy = np.asarray([r['rpy'] for r in rows])
    velocity = np.asarray([r['root_velocity'] for r in rows])
    action = np.asarray([r['final_action'] for r in rows])
    if clean.shape != (len(rows),31) or joint.shape != (len(rows),23):
        raise ValueError('Expected clean 31D and joint 23D arrays')
    if not all(np.isfinite(x).all() for x in (clean,ref,joint,rpy,velocity,action)):
        raise ValueError('Non-finite trace values')
    error = joint-clean[:,8:]
    angle_error = np.arctan2(np.sin(rpy-clean[:,1:4]), np.cos(rpy-clean[:,1:4]))
    height = np.asarray([r['height'] for r in rows])
    tilt = np.arccos(np.clip(np.cos(rpy[:,0])*np.cos(rpy[:,1]),-1,1))
    rate = np.r_[np.nan, np.linalg.norm(np.diff(action,axis=0),axis=1)/.02]
    rmse = lambda x: float(np.sqrt(np.mean(x*x)))
    summary = {'source': str(Path(trace).resolve()), 'dt_s': .02, 'frames': len(rows),
               'comparison': 'actual vs clean at identical frame_id; no phase shift',
               'windows': {}}
    for name, start in [('full_sequence',0),('after_warmup',24)]:
        if len(rows) <= start:
            continue
        sl = slice(start,None)
        summary['windows'][name] = dict(joint_rmse=rmse(error[sl]),
            leg_rmse=rmse(error[sl,:12]), arm_rmse=rmse(error[sl,15:]),
            root_velocity_rmse=rmse((velocity-clean[:,4:7])[sl]),
            height_rmse=rmse((height-clean[:,0])[sl]),
            roll_rmse=rmse(angle_error[sl,0]), pitch_rmse=rmse(angle_error[sl,1]),
            yaw_rmse=rmse(angle_error[sl,2]), max_tilt=float(tilt[sl].max()),
            per_joint_rmse=dict(zip(JOINTS,np.sqrt(np.mean(error[sl]**2,axis=0)).tolist())))
    (out/'curve_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    headers = ['time_s','frame_id','joint_rmse_rad','leg_rmse_rad','arm_rmse_rad',
               'roll_error_rad','pitch_error_rad','yaw_error_rad','tilt_rad',
               'action_rate_per_s','inference_ms']
    columns = [t, ids, np.sqrt(np.mean(error**2,axis=1)),
               np.sqrt(np.mean(error[:,:12]**2,axis=1)),np.sqrt(np.mean(error[:,15:]**2,axis=1)),
               *angle_error.T,tilt,rate,np.asarray([r['inference_ms'] for r in rows])]
    for j,name in enumerate(JOINTS):
        for label,values in [('clean',clean[:,8:]),('processed',ref[:,8:]),('actual',joint)]:
            headers.append(name+'_'+label+'_rad'); columns.append(values[:,j])
    with (out/'curves.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(headers); writer.writerows(zip(*columns))

    def finish(fig, axes, filename):
        for ax in np.asarray(axes).ravel():
            if not ax.get_visible(): continue
            ax.grid(alpha=.2); ax.set_xlabel('Simulation time (s)')
            ax.axvspan(0,.48,color='#aaaaaa',alpha=.15)
        fig.tight_layout(); fig.savefig(out/filename,dpi=140); plt.close(fig)

    for filename, indices in [('legs.png',list(range(12))),('upper_body.png',list(range(12,23)))]:
        fig, axes=plt.subplots(6,2,figsize=(14,15),sharex=True)
        for ax,j in zip(axes.ravel(),indices):
            ax.plot(t,clean[:,8+j],color='#444444',ls='--',lw=1,label='Clean target')
            ax.plot(t,ref[:,8+j],color='#dd9933',lw=.8,label='Policy reference')
            ax.plot(t,joint[:,j],color='#2878b5',lw=.8,label='Actual')
            ax.set_title(JOINTS[j]); ax.set_ylabel('Angle (rad)')
        for ax in axes.ravel()[len(indices):]: ax.set_visible(False)
        axes[0,0].legend(fontsize=8); finish(fig,axes,filename)
    fig, axes=plt.subplots(4,2,figsize=(14,12),sharex=True)
    actuals=[height,*rpy.T,*velocity.T]; targets=[clean[:,0],*clean[:,1:4].T,*clean[:,4:7].T]
    labels=['Height (m)','Roll (rad)','Pitch (rad)','Yaw (rad)','Root vx (m/s)','Root vy (m/s)','Root vz (m/s)']
    for ax,a,b,label in zip(axes.ravel(),actuals,targets,labels):
        ax.plot(t,b,'--',color='#444444',label='Clean'); ax.plot(t,a,color='#2878b5',label='Actual')
        ax.set_ylabel(label)
    axes[0,0].legend(); axes[-1,-1].plot(t,tilt,color='#2878b5'); axes[-1,-1].set_ylabel('Tilt (rad)')
    finish(fig,axes,'root.png')
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
    for label,e in [('All joints',error),('Legs',error[:,:12]),('Arms',error[:,15:])]:
        axes[0].plot(t,np.sqrt(np.mean(e**2,axis=1)),label=label,lw=.8)
    axes[0].legend(); axes[0].set_ylabel('Frame RMSE (rad)')
    axes[1].plot(t,rate,color='#2878b5'); axes[1].set_ylabel('Action change / s')
    axes[2].plot(t,columns[10],color='#2878b5'); axes[2].set_ylabel('Inference (ms)')
    finish(fig,axes,'tracking.png')
    if all('dtera' in r for r in rows):
        fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
        for key in ('gated_delta_dyn','gated_delta_err','applied_delta'):
            if all(key in r['dtera'] for r in rows):
                values=np.asarray([r['dtera'][key] for r in rows])
                axes[0].plot(t,np.linalg.norm(values,axis=1),label=key,lw=.8)
        for key in ('dynamics_gate','tracking_gate','gate'):
            if all(key in r['dtera'] for r in rows):
                axes[1].plot(t,[np.mean(r['dtera'][key]) for r in rows],label=key,lw=.8)
        axes[0].set_ylabel('Diagnostic residual L2'); axes[1].set_ylabel('Gate')
        for ax in axes: ax.legend(fontsize=8)
        finish(fig,axes,'dtera.png')
    print('Simulation curves saved to',out.resolve())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(); export(args.trace,args.out)
