"""Read-only tensor and JIT verification of the saved first actor update."""
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
from export_twist_anyadapter_opentrack_jit import build_actor,DeployActor

SOURCE=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
RUN=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(out):
    out.mkdir(parents=True,exist_ok=False)
    paths=dict(heading300=SOURCE,model25=RUN/'model_25.pt',model26=RUN/'model_26.pt')
    snapshots={k:torch.load(p,map_location='cpu') for k,p in paths.items()}
    state={k:v['model_state_dict'] for k,v in snapshots.items()}
    assert set(state['heading300'])==set(state['model25'])==set(state['model26'])
    warmup_changed=[k for k,v in state['heading300'].items() if not torch.equal(v,state['model25'][k])]
    assert all(k.startswith('critic.') for k in warmup_changed),warmup_changed
    changes=[]
    for key,value in state['model25'].items():
        other=state['model26'][key]
        if torch.equal(value,other):continue
        diff=(other-value).double()
        changes.append(dict(parameter=key,shape=list(value.shape),changed_elements=int((diff!=0).sum()),
                            max_abs=float(diff.abs().max()),l2=float(diff.norm())))
    assert all(x['parameter'].startswith(('critic.','layerwise_actor.adapter.')) for x in changes)
    actors={k:build_actor(p) for k,p in paths.items()}
    generator=torch.Generator().manual_seed(9025)
    x=torch.randn(128,7003,generator=generator)
    jit_checks={}
    with torch.no_grad():
        for name,actor in actors.items():
            deploy=DeployActor(actor).eval();jit=torch.jit.trace(deploy,x[:2])
            jit.save(str(out/(name+'.pt')))
            error=float((jit(x)-actor.actor_mean(x)).abs().max())
            assert error<=1e-6,error
            jit_checks[name]=dict(eager_jit_max_abs=error,input_dim=7003,output_dim=23,
                gain=actor.layerwise_actor.adapter_gain,jit=str(out/(name+'.pt')))
        source_vs25=float((actors['heading300'].actor_mean(x)-actors['model25'].actor_mean(x)).abs().max())
        assert source_vs25==0
        existing=torch.jit.load(str(ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt')).eval()
        source_vs_existing=float((existing(x)-actors['heading300'].actor_mean(x)).abs().max());assert source_vs_existing==0
    result=dict(actor25_source_weights_exact=True,actor25_source_random_input_error=source_vs25,
        source_existing_jit_error=source_vs_existing,warmup_changed_keys=warmup_changed,
        first_update_changed_parameters=changes,all_other_parameters_exact=True,jit_checks=jit_checks,
        iterations={k:v.get('iter') for k,v in snapshots.items()},
        source_hashes={k:sha(p) for k,p in paths.items()},
        parameter_grad_flags_not_saved='requires_grad is not in checkpoint state_dict; eligibility checked against training code and optimizer groups')
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().output),indent=2))
