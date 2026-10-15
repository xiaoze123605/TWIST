"""Second pool proximity view: all compressed policy features, equally weighted blocks."""
import json,sys
import numpy as np
import torch
from analyze_heading300_sensitivity import OUT,POOL,NearestNeighbors,csv_write
from sim_eval_metrics import phase_masks
from pathlib import Path
torch.set_num_threads(1)
bank=torch.load(POOL,map_location='cpu')['recovery1250']['replay']['groups'];pool=torch.cat([bank[g]['inputs'] for g in sorted(bank,key=int)]).numpy();mean=pool.mean(0);std=pool.std(0);active=std>1e-6;weights=np.zeros(1285)
for a,b in [(0,105),(105,1155),(1155,1283),(1283,1285)]:weights[a:b]=1/np.sqrt(active[a:b].sum())/2
scale=std.copy();scale[~active]=1
normalized=(pool[:,active]-mean[active])/scale[active]*weights[active];nn=NearestNeighbors().fit(normalized);cal=nn.kneighbors(normalized)[0][:,1];cut=float(np.quantile(cal,.99));rows=[]
summary=json.loads((OUT/'evaluation_retry/summary.json').read_text())
for r in summary['rows']:
 label,mode=r['label'],r['reference_mode'];data=np.load(OUT/'analysis'/f'{label}_{mode}_components.npz');obs=data['actor_input'];teacher_latent=None
 # Reuse encoder inference helper; all encoder weights frozen and identical.
 from analyze_heading300_sensitivity import actors,outputs
 actor,_=actors();latent=outputs(actor,obs)['latent'];feat=np.c_[obs[:,:1155],latent,obs[:,-2:]];query=(feat[:,active]-mean[active])/scale[active]*weights[active];distance=nn.kneighbors(query,n_neighbors=1)[0][:,0]
 frames=[json.loads(x) for x in Path(r['trace']).read_text().splitlines()];masks={'all':np.ones(len(obs),bool),**phase_masks(np.array([f['clean'] for f in frames]))}
 for phase,mask in masks.items():rows.append(dict(label=label,mode=mode,phase=phase,states=int(mask.sum()),distance_p50=float(np.median(distance[mask])),distance_p99=float(np.quantile(distance[mask],.99)),distance_max=float(distance[mask].max()),sparse_fraction=float((distance[mask]>cut).mean()),calibration_p99=cut))
 np.savez_compressed(OUT/'analysis'/f'{label}_{mode}_coverage_full.npz',distance=distance)
csv_write(OUT/'analysis/coverage_full_compressed.csv',rows)
(OUT/'analysis/coverage_full_definition.json').write_text(json.dumps(dict(dim=1285,active=int(active.sum()),blocks=['current base105','TWIST base history1050','frozen latent128','heading context2'],definition='Pool mean/std per nonconstant feature, equal block squared-distance weights; all19200 nearest neighbors; full leave-one-out calibration; descriptive proximity only, not proof of training coverage',calibration_p99=cut),indent=2))
np.savez_compressed(OUT/'analysis/coverage_full_normalization.npz',mean=mean,std=std,weights=weights,active=active,calibration=cal)
print(rows)
