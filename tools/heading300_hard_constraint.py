"""Full-pool functional constraint with transactional, group-consistent Adam retries."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from heading300_teacher import complete_teacher_action

RATIOS=(1.,.5,.25,.125,.0625,.03125)
BOUND=1e-4


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone_state(state):
    return {key:value.detach().clone() if torch.is_tensor(value) else deepcopy(value) for key,value in state.items()}


class FixedCompleteTeacherPool:
    """Frozen19200 compressed inputs; frozen encoder latents are already saved."""
    def __init__(self,actor,teacher,checkpoint,chunk=1024):
        self.actor=actor;self.teacher=teacher;self.path=Path(checkpoint);self.digest=sha(checkpoint)
        self.chunk=chunk;self.device=next(actor.parameters()).device
        state=torch.load(checkpoint,map_location='cpu');recovery=state['recovery1250']
        if state['iter']!=25 or not recovery.get('complete_heading_teacher') or recovery.get('adapter_gain')!=.5:
            raise ValueError('fixed pool must be complete-teacher gain0.5 model25')
        if actor.layerwise_actor.adapter_gain!=.5 or teacher.layerwise_actor.adapter_gain!=.5:
            raise ValueError('gain must remain0.5')
        source=teacher.state_dict()
        if any(not torch.equal(source[k].cpu(),v) for k,v in state['model_state_dict'].items() if not k.startswith('critic.')):
            raise ValueError('fixed-pool actor differs from heading300 teacher')
        groups=recovery['replay']['groups']
        if set(map(int,groups))!=set(range(6)):raise ValueError('expected six original replay strata')
        ordered=[groups[g] for g in sorted(groups,key=int)]
        self.cpu_inputs=torch.cat([g['inputs'] for g in ordered]).float()
        saved=torch.cat([g['targets'] for g in ordered]).float()
        if self.cpu_inputs.shape!=(19200,1285) or saved.shape!=(19200,23):
            raise ValueError('all19200 states and23 targets required')
        if self.cpu_inputs.requires_grad or saved.requires_grad:raise ValueError('teacher pool must stop gradient')
        if not torch.isfinite(self.cpu_inputs).all() or not torch.isfinite(saved).all():raise ValueError('nonfinite pool')
        # Forward paths contain no stochastic or running-statistics modules.
        for model in [actor.layerwise_actor,teacher.layerwise_actor]:
            if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for m in model.modules()):
                raise ValueError('constraint path would require special mutable/stochastic state handling')
        actor_before={k:v.detach().clone() for k,v in actor.state_dict().items()}
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None
        self.cpu_actor=deepcopy(actor).cpu().eval().requires_grad_(False)
        self.cpu_teacher=deepcopy(teacher).cpu().eval().requires_grad_(False)
        self.inputs=self.cpu_inputs.to(self.device)
        with torch.no_grad():
            self.targets,self.core_targets,self.head_targets=self._outputs(teacher,self.inputs)
            self.cpu_targets,_,_=self._outputs(self.cpu_teacher,self.cpu_inputs)
        saved_error=float((self.cpu_targets-saved).abs().max())
        if saved_error>1e-5:raise ValueError('saved targets omit/change complete heading teacher')
        if any(not torch.equal(v,actor.state_dict()[k]) for k,v in actor_before.items()):raise RuntimeError('shadow creation changed actor')
        if not torch.equal(rng,torch.get_rng_state()):raise RuntimeError('pool initialization drew CPU RNG')
        if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.device)):raise RuntimeError('pool initialization drew GPU RNG')
        self.receipt=dict(checkpoint=str(self.path.resolve()),sha256=self.digest,states=19200,action_dim=23,
            input_dim=1285,layout='1155 base +128 frozen history latent +2 heading context',
            saved_complete_teacher_max_error=saved_error,gain=.5,teacher_frozen=True,
            check_devices=[str(self.device),'cpu'],chunk=chunk,soft_replay_unchanged=True)

    def _sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _outputs(self,actor,inputs):
        core=[];head=[]
        for x in inputs.split(self.chunk):
            b,z,c=x[:,:1155],x[:,1155:1283],x[:,-2:]
            core.append(actor.layerwise_actor(b,z));head.append(actor.heading_delta(z,c))
        core=torch.cat(core);head=torch.cat(head)
        return core+head,core,head

    @torch.no_grad()
    def check(self):
        self._sync();start=time.perf_counter()
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None
        full,core,head=self._outputs(self.actor,self.inputs)
        self.cpu_actor.adapter.load_state_dict(self.actor.adapter.state_dict(),strict=True)
        cpu,_,_=self._outputs(self.cpu_actor,self.cpu_inputs)
        errors=(full-self.targets).abs();cpu_errors=(cpu-self.cpu_targets).abs()
        gpu_max=float(errors.max());cpu_max=float(cpu_errors.max())
        max_abs=max(gpu_max,cpu_max)
        if not math.isfinite(max_abs):raise RuntimeError('nonfinite functional deviation')
        state_max=errors.amax(-1);quant=torch.quantile(errors.flatten(),torch.tensor([.95,.99],device=self.device))
        sq=torch.quantile(state_max,torch.tensor([.95,.99],device=self.device))
        index=int(errors.flatten().argmax())
        result=dict(max_abs=max_abs,training_device_max=gpu_max,cpu_max=cpu_max,
            mean_abs=float(errors.mean()),p95_abs=float(quant[0]),p99_abs=float(quant[1]),
            state_max_mean=float(state_max.mean()),state_max_p95=float(sq[0]),state_max_p99=float(sq[1]),
            full_rms=float((full-self.targets).square().mean().sqrt()),
            core_output_drift_rms=float((core-self.core_targets).square().mean().sqrt()),
            heading_output_drift_rms=float((head-self.head_targets).square().mean().sqrt()),
            worst_training_state=index//23,worst_training_action=index%23,states_checked=19200,actions_checked=23)
        self._sync();result['check_seconds']=time.perf_counter()-start
        if not torch.equal(rng,torch.get_rng_state()):raise RuntimeError('constraint check drew CPU RNG')
        if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.device)):raise RuntimeError('constraint check drew GPU RNG')
        return result


class BoundedAdamStep:
    """Retry exactly the same gradients from one pre-state; critic advances once."""
    def __init__(self,optimizer,checker,log_path=None,actor_groups=(0,2),bound=BOUND,ratios=RATIOS):
        if type(optimizer) is not torch.optim.Adam:raise TypeError('only the reviewed shared Adam is supported')
        if any(g.get('differentiable',False) for g in optimizer.param_groups):raise ValueError('differentiable Adam unsupported')
        self.optimizer=optimizer;self.original_step=optimizer.step;self.checker=checker
        self.actor_groups=tuple(actor_groups);self.bound=bound;self.ratios=tuple(ratios)
        self.actor=[p for i,g in enumerate(optimizer.param_groups) if i in self.actor_groups for p in g['params']]
        self.critic=[p for i,g in enumerate(optimizer.param_groups) if i not in self.actor_groups for p in g['params']]
        if not self.actor or not self.critic or len(set(map(id,self.actor+self.critic)))!=len(self.actor+self.critic):raise ValueError('invalid/disjoint optimizer groups')
        self.log_path=Path(log_path) if log_path else None;self.context={};self.records=[]

    def snapshot(self,parameters):
        return {p:(p.detach().clone(),clone_state(self.optimizer.state[p]) if p in self.optimizer.state else None) for p in parameters}

    @torch.no_grad()
    def restore(self,snapshot):
        for p,(value,state) in snapshot.items():
            p.copy_(value)
            if state is None:self.optimizer.state.pop(p,None)
            else:self.optimizer.state[p]=clone_state(state)

    def verify_step_counts(self,before,parameters,delta):
        for p in parameters:
            previous=before[p][1];old=float(previous['step']) if previous and 'step' in previous else 0.
            current=self.optimizer.state.get(p,{});new=float(current['step']) if 'step' in current else 0.
            expected=delta if p.grad is not None else 0
            if new!=old+expected:raise RuntimeError('Adam step advanced inconsistently')

    def equal_snapshot(self,snapshot):
        for p,(value,state) in snapshot.items():
            if not torch.equal(p,value):return False
            now=self.optimizer.state.get(p)
            if state is None:
                if now is not None:return False
            elif now is None or set(now)!=set(state):return False
            else:
                for k,v in state.items():
                    if torch.is_tensor(v):
                        if not torch.equal(v,now[k]):return False
                    elif v!=now[k]:return False
        return True

    def log(self,record):
        self.records.append(record)
        if self.log_path:
            with self.log_path.open('a') as f:f.write(json.dumps(record)+'\n')

    def step(self,closure=None):
        if closure is not None:raise ValueError('closure would recompute gradients/RNG and is unsupported')
        start=time.perf_counter();record=dict(self.context)
        if not any(p.grad is not None for p in self.actor):
            self.original_step()
            record.update(actor_attempted=False,status='critic_only_warmup',actor_accepted=False,
                accepted_ratio=0.,actor_parameter_l2=0.,total_step_seconds=time.perf_counter()-start)
            self.log(record);return
        if any(not torch.isfinite(p.grad).all() for p in self.actor+self.critic if p.grad is not None):raise RuntimeError('nonfinite gradients')
        before=self.snapshot(self.actor+self.critic);base_rates=[g['lr'] for g in self.optimizer.param_groups]
        record['base_group_learning_rates']=list(base_rates)
        gradients={p:p.grad.detach().clone() if p.grad is not None else None for p in self.actor+self.critic}
        trials=[];critic_after=None;accepted=None;adam_seconds=0.
        try:
            for attempt,ratio in enumerate(self.ratios):
                if attempt:self.restore(before)
                for i,g in enumerate(self.optimizer.param_groups):g['lr']=base_rates[i]*(ratio if i in self.actor_groups else 1.)
                tick=time.perf_counter();self.original_step()
                if self.actor[0].device.type=='cuda':torch.cuda.synchronize(self.actor[0].device)
                adam_seconds+=time.perf_counter()-tick
                if critic_after is None:critic_after=self.snapshot(self.critic)
                elif not self.equal_snapshot(critic_after):raise RuntimeError('critic retry differs from exactly one original Adam update')
                for p,v in gradients.items():
                    if v is not None and not torch.equal(p.grad,v):raise RuntimeError('retry changed the clipped gradient')
                self.verify_step_counts(before,self.actor+self.critic,1)
                stats=self.checker();trials.append(dict(ratio=ratio,**stats))
                if stats['max_abs']<=self.bound:
                    accepted=ratio;break
            if accepted is None:
                self.restore({p:before[p] for p in self.actor})
                if not self.equal_snapshot(critic_after):raise RuntimeError('rejection corrupted critic update')
                if not self.equal_snapshot({p:before[p] for p in self.actor}):raise RuntimeError('rejection failed to restore Actor and Adam')
                final=self.checker()
                if final['max_abs']>self.bound:raise RuntimeError('pre-update Actor was already outside hard bound')
                self.verify_step_counts(before,self.actor,0);self.verify_step_counts(before,self.critic,1)
            else:final=trials[-1]
        except Exception:
            self.restore(before);raise
        finally:
            for g,rate in zip(self.optimizer.param_groups,base_rates):g['lr']=rate
        changes=[(p-before[p][0]).detach().square().sum() for p in self.actor]
        actor_l2=float(torch.stack(changes).sum().sqrt())
        changes_by_group={str(i):float(torch.stack([(p-before[p][0]).detach().square().sum() for p in self.optimizer.param_groups[i]['params']]).sum().sqrt()) for i in self.actor_groups}
        record.update(actor_attempted=True,actor_accepted=accepted is not None,accepted_ratio=accepted or 0.,
            status='accepted' if accepted is not None else 'rejected_all_six_ratios',
            reason=None if accepted is not None else 'all allowed Actor learning-step ratios exceed1e-4',
            trials=trials,backtracks=len(trials)-1,actor_parameter_l2=actor_l2,
            final=final,adam_state_consistent=True,actor_adam_step_delta=1 if accepted is not None else 0,
            critic_adam_step_delta=1,adam_seconds=adam_seconds,total_step_seconds=time.perf_counter()-start)
        record['actor_parameter_l2_by_group']=changes_by_group
        record['actual_accepted_group_learning_rates']=[rate*((accepted or 0.) if i in self.actor_groups else 1.) for i,rate in enumerate(base_rates)]
        self.log(record)
