"""Offline core/heading interchange with strict shared-module and output checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'legged_gym/scripts')]
import torch
from export_twist_anyadapter_opentrack_jit import build_actor,DeployActor

SOURCE=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
NEW=ROOT/'legged_gym/logs/recovery_heading300_sampling/sample_only_alpha014_p1_env512_seed42_20261009/model_50.pt'
SOURCE_JIT=ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt'
NEW_JIT=NEW.parent/'evaluations/update_00000050_1791515363900723420_fc6937d3/exports/checkpoint/policy.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct(output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    before={str(p):sha(p) for p in [SOURCE,NEW,SOURCE_JIT,NEW_JIT]}
    old,new=build_actor(SOURCE),build_actor(NEW)
    states=[old.state_dict(),new.state_dict()]
    if states[0].keys()!=states[1].keys():raise ValueError('incompatible state keys')
    for actor in [old,new]:
        if len(actor.layerwise_actor.base_layers)!=5 or len(actor.adapter)!=6:
            raise ValueError('requires five core layers and one separate heading layer')
        if actor.adapter[5].weight.shape!=(23,130) or actor.layerwise_actor.adapter_gain!=.5:
            raise ValueError('heading shape or gain differs')
    core={k for k in states[0] if k.startswith('layerwise_actor.adapter.') and k.split('.')[2] in ('0','1','2','3','4')}
    head={k for k in states[0] if k.startswith('layerwise_actor.adapter.5.')}
    if len(core)!=10 or len(head)!=2:raise ValueError('unexpected module state layout')
    shared=set(states[0])-core-head-{k for k in states[0] if k.startswith('critic.')}
    if any(not torch.equal(states[0][k],states[1][k]) for k in shared):
        raise ValueError('TWIST/history/WM/normalizer/std differ; direct swap not safe')
    policies={};receipt={}
    for label,ci,hi in [('A',0,0),('B',1,0),('C',0,1),('D',1,1)]:
        state=dict(states[0])
        state.update({k:states[ci][k] for k in core})
        state.update({k:states[hi][k] for k in head})
        actor=copy.deepcopy(old);actor.load_state_dict(state,strict=True);actor.eval()
        eager=DeployActor(actor).eval();error=0.
        with torch.no_grad():
            for batch in [1,2,17,64]:
                obs=torch.randn(batch,7003)
                obs[:,-2]=torch.sin(obs[:,-2]);obs[:,-1]=1-torch.cos(obs[:,-1])
                base,hist=old.split_obs(obs);z=old.history_encoder(hist)
                expected=[old,new][ci].layerwise_actor(base,z)+[old,new][hi].heading_delta(z,obs[:,-2:])
                error=max(error,float((eager(obs)-expected).abs().max()))
                if label in ('A','D'):
                    original=torch.jit.load(str(SOURCE_JIT if label=='A' else NEW_JIT))
                    error=max(error,float((eager(obs)-original(obs)).abs().max()))
        if error!=0:raise ValueError('combination output mismatch '+label+': '+str(error))
        sample=torch.randn(2,7003);script=torch.jit.trace(eager,sample)
        with torch.no_grad():jit_error=float((script(sample)-eager(sample)).abs().max())
        if jit_error!=0:raise ValueError('JIT mismatch '+label)
        path=output/(label+'.pt')
        if label in ('A','D'):
            # Identical artifact copies let the evaluator deduplicate exactly identical controls.
            shutil.copyfile(SOURCE_JIT if label=='A' else NEW_JIT,path)
        else:script.save(str(path))
        torch.save(dict(model_state_dict=state,recovery1250=dict(adapter_gain=.5),diagnostic_only=True,
                        combination=label,source_hashes=before),output/(label+'_diagnostic_state.pth'))
        receipt[label]=dict(core='old' if ci==0 else 'new',heading='old' if hi==0 else 'new',
                            jit=str(path.resolve()),sha256=sha(path),input_dim=7003,output_dim=23,
                            gain=.5,formula_max_abs_error=error,jit_max_abs_error=jit_error)
        policies[label]=actor
    if before!={str(p):sha(p) for p in [SOURCE,NEW,SOURCE_JIT,NEW_JIT]}:
        raise ValueError('source artifact changed')
    result=dict(shared_state_exact=True,core_keys=sorted(core),heading_keys=sorted(head),
                sources=before,combinations=receipt,no_optimizer_or_training=True,
                independence='heading consumes frozen history latent plus context; core changes do not feed heading directly')
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    construct(parser.parse_args().output)
