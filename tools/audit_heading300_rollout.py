"""Short fixed-policy training-environment audit. No optimizer or learning."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose')]
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args
from types import SimpleNamespace
def restore_configuration(obj, values):
 for key,value in values.items():
  if isinstance(value,dict):
   child=getattr(obj,key,None)
   if isinstance(child,dict):setattr(obj,key,value);continue
   if not hasattr(child,"__dict__"):child=SimpleNamespace();setattr(obj,key,child)
   restore_configuration(child,value)
  else:setattr(obj,key,value)
from heading23500_env import Heading23500Env
sys.path.insert(0,str(ROOT/'legged_gym/scripts'))
from export_twist_anyadapter_opentrack_jit import build_actor
out=ROOT/'reports/heading300_diagnosis_20261008'
source=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008'
manifest=json.loads((source/'manifest.json').read_text());cfg,train=task_registry.get_cfgs('g1_motion_wm_anyadapter');restore_configuration(cfg,manifest['environment'])
args=get_args();args.task='heading300_diagnostic_audit';args.num_envs=128;args.headless=True;args.seed=42
cfg.env.num_envs=128;task_registry.register(args.task,Heading23500Env,cfg,train);env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
actor=build_actor(source/'model_300.pt').to(env.device).eval();obs=env.get_observations();data=[];resets=0
with torch.no_grad():
 for step in range(512):
  actions=actor.actor_mean(obs);result=env.step(actions);obs=result[0];resets+=int(result[3].sum())
  knee=env._ref_dof_pos[:,[3,9]].mean(-1);h=env._ref_root_pos[:,2]-env.episode_init_origin[:,2]
  squat=(knee>.8)&(h<.7);region=(knee>.5)&(h<.78)
  stance=getattr(env,'ref_stance_confidence',torch.zeros((128,2),device=env.device))
  masks={'squat':squat,'descent':region&(env._ref_root_vel[:,2]<-.03),'rise':region&(env._ref_root_vel[:,2]>.03),'squat_stance':squat&(stance.min(-1).values>.5),'squat_inplace':squat&env._in_place_reference_mask()}
  r=dict(step=step,**{k:float(v.float().mean()) for k,v in masks.items()},squat_confidence=float((squat*stance.mean(-1)).mean()))
  for name in ['squat_knee_tracking','hip_yaw_tracking','reference_stance_slip','heading_tracking']:
   value=getattr(env,'_reward_'+name)();r[name+'_raw']=float(value.mean());r[name+'_scaled']=float(value.mean()*env.reward_scales[name])
  r['knee_error_active_mean']=float((env.dof_pos[:,[3,9]]-env._ref_dof_pos[:,[3,9]])[squat].abs().mean()) if squat.any() else None
  data.append(r)
  if step%128==0:print('Audit step',step,flush=True)
summary=dict(policy=str(source/'model_300.pt'),base_commit='5e45fe8',optimizer_steps=0,environment_count=128,control_steps=512,total_environment_frames=128*512,sim_seconds_per_env=512*env.dt,resets=resets,seed=42,actor_obs_dim=obs.shape[-1],critic_obs_dim=env.get_privileged_observations().shape[-1],weighted_exposure={k:sum(x[k] for x in data)/len(data) for k in data[0] if k!='step' and all(x[k] is not None for x in data)},limits='fresh resets, 10.24s/environment, deterministic actor with existing training domain randomization/noise; not historical PPO coverage')
(out/'rollout_sampling_audit.json').write_text(json.dumps(summary,indent=2));(out/'rollout_sampling_steps.json').write_text(json.dumps(data));print(json.dumps(summary,indent=2))
