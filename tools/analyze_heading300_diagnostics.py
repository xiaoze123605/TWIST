"""Matched reference phase, static support and continuous-segment spectral audit."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/heading300_diagnosis_20261008';DT=.02

from sim_eval_metrics import segments,spectrum,rms,analyze_trace

def main():
    allrows={};spectra={};cache={}
    for label in ['heading_wm','heading_raw','ema025','ema05','ema075','head_off','projection_off','original23500']:
     result=json.loads((OUT/label/'results.json').read_text())
     summary,arrays,case_spectra=analyze_trace(result,label)
     allrows[label]=summary;cache[label]=arrays;spectra.update(case_spectra)
     if label!='heading_raw':assert np.array_equal(arrays[0],cache['heading_wm'][0]),label
    (OUT/'diagnostic_summary.json').write_text(json.dumps(allrows,indent=2))
    fields=['label','phase','seconds','joint_rmse','pitch_rmse_degrees','pitch_error_p95_degrees','actual_com_forward_cm','reference_com_forward_cm','projection_frame_fraction','torque_saturation_fraction','policy_action_rate_rms','pd_target_rate_rms','actual_joint_acceleration_rms']
    with (OUT/'phase_comparison.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
     for label,row in allrows.items():
      for phase,stats in row['phases'].items():w.writerow(dict(label=label,phase=phase,**{k:stats[k] for k in fields[2:]}))
    fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for label,color in [('heading_wm','#0072b2'),('head_off','#777777'),('ema025','#009e73'),('ema05','#e69f00')]:
     ref,q,pitch,rpitch,ppitch,tpitch,com,rc,masks,proj,sat,act=cache[label];t=np.arange(len(q))*DT
     axs[0].plot(t,np.degrees(pitch),label=label,color=color,lw=.9)
     axs[1].plot(t,(com-rc)*100,label=label,color=color,lw=.9)
     axs[2].plot(t,np.linalg.norm(np.diff(act[:,:12],axis=0,prepend=act[:1,:12]),axis=-1)/DT,color=color,lw=.7)
     axs[3].plot(t,np.degrees(q[:,14]-ref[:,22]),color=color,lw=.9)
    axs[0].plot(t,np.degrees(cache['heading_wm'][3]),color='black',ls='--',label='clean reference torso')
    for ax,title in zip(axs,['Torso pitch (deg)','COM forward error vs reference (cm)','Leg action rate norm (action/s)','Waist pitch error (deg)']):
     ax.set_ylabel(title);ax.grid(alpha=.2)
     for a,b in segments(cache['heading_wm'][8]['squat']):ax.axvspan(a*DT,b*DT,color='purple',alpha=.08)
    axs[0].legend(ncol=3,fontsize=8);axs[-1].set_xlabel('Reference time (s)');fig.tight_layout();fig.savefig(OUT/'phase_diagnostics.png',dpi=160)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for label in ['heading_wm','ema025','ema05','ema075','head_off','original23500']:
     for ax,name in zip(axs,['actual','policy_post_ema']):
      if (label,name) in spectra:
       f,p=spectra[label,name];ax.semilogy(f,p.mean(-1),label=label)
    for ax,name in zip(axs,['Actual leg angle','Policy leg action']):ax.set_title(name);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'walking_spectra.png',dpi=160)
    print('Saved phase, COM/contact, torque, head and frequency analysis')

if __name__=="__main__":main()
