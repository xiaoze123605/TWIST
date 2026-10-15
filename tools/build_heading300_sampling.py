"""Keep the full original motion distribution; mix a small support-squat component."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--focus-mass',type=float,default=.11);p.add_argument('--power',type=float,default=2.);a=p.parse_args()
if not 0<a.focus_mass<=.2:raise ValueError('preserve at least80% original sampling component')
config=ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml'
audit=ROOT/'reports/heading300_diagnosis_20261008/sampling_by_motion.csv'
cfg=yaml.safe_load(config.read_text());by_file={r['file']:r for r in csv.DictReader(audit.open())};rows=[by_file[x['file']] for x in cfg['motions']]
w=np.array([x['weight'] for x in cfg['motions']],dtype=float)
assert np.array_equal(w,np.array([float(r['weight']) for r in rows])), 'audit weight mismatch'
w/=w.sum();s=np.array([float(r['squat_stance']) for r in rows]);focus=w*s**a.power;focus/=focus.sum();mixed=(1-a.focus_mass)*w+a.focus_mass*focus
assert all(x['file']!='mocap1/77.pkl' for x in cfg['motions'])
new=dict(cfg,motions=[dict(x,weight=float(v)) for x,v in zip(cfg['motions'],mixed)])
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:yaml.safe_dump(new,stream,allow_unicode=False,sort_keys=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
summary=dict(source_config=str(config),source_sha256=sha(config),audit_sha256=sha(audit),sampling_sha256=sha(a.output),focus_mass=a.focus_mass,power=a.power,motions=len(rows),original_component=1-a.focus_mass,all_original_motions_retained=True,time_sampling_unchanged=True,motion77_held_out=True,predicted_reset_coverage={k:float(mixed@np.array([float(r[k]) for r in rows])) for k in ['squat','squat_stance','descent','rise','squat_inplace']},top_focus=[dict(file=cfg['motions'][i]['file'],probability=float(mixed[i])) for i in np.argsort(mixed)[-15:][::-1]])
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
