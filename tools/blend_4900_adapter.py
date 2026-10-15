"""Interpolate a pilot's adapter weights toward the untouched model_4900."""
import argparse,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'legged_gym/logs/g1_motion_wm_anyadapter/pilot_gpu_seed42_200/model_4900.pt'
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if not 0<a.alpha<1:raise ValueError('alpha must be in (0,1)')
source=torch.load(SOURCE,map_location='cpu',weights_only=False);candidate=torch.load(a.candidate,map_location='cpu',weights_only=False)
s=source['model_state_dict'];c=candidate['model_state_dict']
if set(s)!=set(c):raise ValueError('architecture mismatch')
for k in s:
 if k.startswith(('history_encoder.','world_model.','base_actor.','layerwise_actor.base_','layerwise_actor.motion_')) and not torch.equal(s[k],c[k]):
  raise ValueError('candidate changed protected tensor '+k)
 if k.startswith('layerwise_actor.adapter.'):
  c[k]=s[k]+a.alpha*(c[k]-s[k])
candidate['model_state_dict']=c
candidate['interpolation_from_4900']=dict(alpha=a.alpha,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest())
a.output.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists():raise FileExistsError(a.output)
torch.save(candidate,a.output)
print(a.output)
