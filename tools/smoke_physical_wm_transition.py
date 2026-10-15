"""Verify opt-in physical transition hooks in real Isaac Gym; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose'),str(ROOT/'legged_gym/scripts')]
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args
from legged_gym.envs.g1.physical_wm_transition_mixin import PhysicalWMTransitionMixin
from heading23500_env import Heading23500Env
from export_twist_anyadapter_opentrack_jit import build_actor
from rsl_rl.modules.physical_world_model import validate_transition


class RecordedHeadingEnv(PhysicalWMTransitionMixin,Heading23500Env):pass


def restore(obj,values):
    for key,value in values.items():
        if isinstance(value,dict):
            child=getattr(obj,key,None)
            if isinstance(child,dict):setattr(obj,key,value);continue
            if not hasattr(child,'__dict__'):child=SimpleNamespace();setattr(obj,key,child)
            restore(child,value)
        else:setattr(obj,key,value)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-steps',type=int,default=128)
    p.add_argument('--force-timeout-step',type=int,default=-1);custom,remaining=p.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining;args=get_args()
    out=custom.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008'
    manifest=json.loads((source/'manifest.json').read_text());cfg,train=task_registry.get_cfgs('g1_motion_wm_anyadapter')
    restore(cfg,manifest['environment']);cfg.motion.motion_file=str(ROOT/'track_dataset/twist_motion_dataset/accad/B3___walk1.pkl')
    cfg.env.num_envs=16;cfg.seed=42;args.num_envs=16;args.seed=42;args.headless=True;args.task='physical_wm_contract_smoke'
    task_registry.register(args.task,RecordedHeadingEnv,cfg,train)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    actor=build_actor(source/'model_300.pt').to(env.device).eval().requires_grad_(False)
    actor_hash_before=sha(source/'model_300.pt');before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    obs=env.get_observations();packets=[];mismatch=projection=terminals=0;raw_max=0.;masked_max=0.
    with torch.no_grad():
        for step in range(custom.smoke_steps):
            if step==custom.force_timeout_step:
                env.episode_length_buf[0]=int(env.max_episode_length)+1
            action=actor.actor_mean(obs);original_action=action.clone()
            result=env.step(action);obs=result[0];packet=result[4]['physical_wm_transition'];validate_transition(packet)
            assert torch.equal(action,original_action)
            mismatch+=int((packet['commanded_action']-packet['executed_action']).abs().amax(-1).gt(1e-7).sum())
            raw=env.default_dof_pos_all+packet['executed_action']*env.cfg.control.action_scale
            lo,hi=env._target_bounds();expected=torch.maximum(torch.minimum(raw,hi),lo)
            torch.testing.assert_close(packet['executed_pd_target'],expected,rtol=0,atol=0)
            projection+=int((expected-raw).abs().amax(-1).gt(1e-7).sum());terminals+=int(packet['done'].sum())
            if step==custom.force_timeout_step:
                assert bool(packet['done'][0]) and not bool(packet['valid'][0])
                # Returned simulator state is after RSI reset; packet is pre-reset.
                assert not torch.equal(packet['next_state'][0],env._physical_wm_state()[0])
            raw_max=max(raw_max,float(packet['state'][:,28+torch.tensor([4,5,10,11],device=env.device)].abs().max()))
            masked_max=max(masked_max,float(obs[:,59+torch.tensor([4,5,10,11],device=env.device)].abs().max()))
            packets.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in packet.items()})
    assert mismatch>0 and raw_max>0 and masked_max==0
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    assert sha(source/'model_300.pt')==actor_hash_before
    torch.save(packets,out/'physical_packets.pt')
    receipt=dict(status='complete',envs=16,steps=custom.smoke_steps,transitions=16*custom.smoke_steps,
        device=env.device,command_execution_mismatch_transitions=mismatch,projected_transitions=projection,
        terminal_transitions=terminals,forced_timeout_step=custom.force_timeout_step,
        raw_ankle_velocity_max=raw_max,legacy_actor_ankle_velocity_max=masked_max,
        actor_weight_changes=0,optimizer_steps=0,legacy_observation_dim=int(obs.shape[-1]),
        checkpoint_sha256=actor_hash_before,scope='Actual PhysX data-flow smoke; no new policy or control improvement claim.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'COMPLETE').write_text('Real physical transition hooks verified, frozen actor unchanged.\n')
    print('SMOKE COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
