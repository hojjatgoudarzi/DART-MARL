"""On-policy, event-boundary recurrent CTDE trajectories.

Every physical tick advances the simulated world. In asynchronous mode only
rovers due by their own clock or a locally observed event select a task.
All other intentions and hidden states persist. Team reward is accumulated
between the UNION of rover decision boundaries with gamma**duration.
This makes team-value SMDP GAE legitimate without global allocations.
"""
import numpy as np
import torch
from hr_mappo.envs.rover import RoverEnv
from .observations import observe_agents, critic_state
from .control import control
from .gae import macro_gae
from .physics import marginal_physical_credit, shapley_service_credit, physical_overstaff, service_potential
from .communication import CommConfig, LocalChannel
from .scheduler import DecisionScheduler


def tensorize(d,device):
    return {k:torch.as_tensor(v,device=device) for k,v in d.items()}


def trajectory(actor, critic, env, seed, cfg, deterministic=False, selector=None, audit_hook=None):
    env.reset(seed=seed)
    device=next(actor.parameters()).device
    # Baseline-only deterministic trajectories are NEVER PPO data. Avoid
    # running unused neural actor/critic at every task-selection boundary.
    classical_eval=(selector is not None and deterministic)
    N=env.config.max_agents
    ah=torch.zeros(N,cfg.hidden,device=device)
    ch=torch.zeros(1,cfg.hidden,device=device)
    previous=np.full(N,-1,dtype=np.int32)
    channel=LocalChannel(N,env.config.max_pois,CommConfig(
        cfg.communication,cfg.message_delay,cfg.message_dropout,
        cfg.message_ttl,cfg.message_range),seed+400301)
    channel.advance(env,previous)
    scheduler=DecisionScheduler(N,cfg.decision_interval,cfg.decision_mode,cfg.async_min_interval)
    records=[]; physical=[]; raw_return=0.; current=None; reset_in_segment=np.ones(N,dtype=bool)
    last_choices=np.full(N,-1,dtype=np.int32)
    last_decision=np.full(N,-1,dtype=np.int32)
    previous_potential=service_potential(env) if cfg.potential_shaping_weight else 0.
    options=[]
    with torch.no_grad():
        while not env.is_done:
            t=int(env.step_index)
            obs=observe_agents(env,previous,channel)
            active=env.active.copy()
            decide,signatures=scheduler.ready(obs,t,previous)
            # Between global union boundaries no central critic or actor call.
            if decide.any():
                if current is not None:
                    current['memory_keep']=reset_in_segment.copy()
                    records.append(current)
                reset_in_segment=np.ones(N,dtype=bool)
                cs=critic_state(env)
                if classical_eval:
                    dist=None
                    new_ah=ah
                    val=torch.zeros(1,device=device)
                    new_ch=ch
                else:
                    ot=tensorize(obs,device)
                    st=tensorize(cs,device)
                    dist,new_ah=actor.distribution(ot,ah)
                    val,new_ch=critic.forward_step(st,ch)
                if selector is None:
                    idx=dist.logits.argmax(-1) if deterministic else dist.sample()
                    proposed=idx.cpu().numpy().astype(np.int32)
                    proposed[proposed==env.config.max_pois]=-1
                else:
                    proposed=np.asarray(selector(obs,env,previous),dtype=np.int32)
                    if proposed.shape != (N,):
                        raise ValueError('task selector returned wrong shape')
                    for i in np.flatnonzero(decide):
                        if not obs['legal'][i,proposed[i] if proposed[i]>=0 else -1]:
                            raise ValueError('task selector proposed illegal action')
                actions=previous.copy()
                actions[decide]=proposed[decide]
                actions[~active]=-1
                # For nondue agents, use the always-legal IDLE as a log-prob
                # placeholder, and a zero actor loss mask in PPO.
                stored=np.full(N,env.config.max_pois,dtype=np.int64)
                stored[decide]=np.where(actions[decide]<0,env.config.max_pois,actions[decide])
                oldlog=(np.zeros(N,dtype=np.float32) if classical_eval else
                        dist.log_prob(torch.as_tensor(stored,device=device)).cpu().numpy())
                ah=torch.where(torch.as_tensor(decide,device=device).unsqueeze(-1),new_ah,ah)
                ch=new_ch
                # Only agent decisions can incur a switching cost.
                old_legal=np.array([bool(obs['legal'][i,old]) if old>=0 else False
                                    for i,old in enumerate(previous)],dtype=bool)
                switched=decide & active & old_legal & (actions>=0) & (previous!=actions)
                switch_cost=float(cfg.switch_penalty)*int(switched.sum())
                # Optional *explicit* IDLE decision cost, no hidden oracle.
                # Never apply it if the agent has no locally legal task.
                idle_with_options=decide & active & (actions<0) & obs['legal'][:,:-1].any(axis=1)
                idle_cost=float(cfg.idle_decision_penalty)*int(idle_with_options.sum())
                for i in np.flatnonzero(decide):
                    if last_decision[i]>=0:
                        options.append(dict(rover=int(i),start=int(last_decision[i]),end=t,
                                            duration=int(t-last_decision[i]),reason='redecision'))
                    last_decision[i]=t
                current=dict(obs=obs,state=cs,actions=stored,old_logp=oldlog,
                             active=active.copy(),decide=decide.copy(),old_value=float(val.item())*cfg.value_output_scale,
                             reward=-switch_cost-idle_cost,shaped_reward=-switch_cost-idle_cost,
                             switch_cost=switch_cost,idle_cost=idle_cost,overstaff_cost=0.,
                             overstaff_rover_steps=0.,
                             marginal_reward=np.zeros(N,dtype=np.float32),
                             choices=actions.copy(),duration=0,terminal=False,truncated=False,
                             raw_reward=0.,boundary_step=t,
                             decision_reasons={str(i):scheduler.pending_reasons[i]
                                               for i in np.flatnonzero(decide)})
                previous=actions
                scheduler.commit(t,decide,signatures,active)
            else:
                # Event-driven mode: activity tracker only; do not reset the
                # decision signature until a real decision is made.
                scheduler.step_activity(active)
            # Offline diagnostics receive only a copy of the legitimate actor's view;
            # hooks are never used for selecting actions or updating the policy.
            if audit_hook is not None:
                audit_hook(env=env, t=t, obs=obs, decide=decide.copy(),
                           previous=previous.copy(), signatures=signatures)
            # No actor/critic call at intermediate physical ticks.
            before_active=env.active.copy()
            previous[~before_active]=-1
            env.set_roles(previous)
            _,r,term,trunc,info=env.step(control(env,previous))
            raw_return+=float(r)
            new_potential=(0. if (term or trunc) else service_potential(env)) if cfg.potential_shaping_weight else 0.
            shaping=float(cfg.potential_shaping_weight)*(cfg.gamma*new_potential-previous_potential)
            surplus=physical_overstaff(info,env.config)
            overstaff_cost=float(cfg.overstaff_penalty_weight)*surplus
            physical_credit=(shapley_service_credit(info,env.config)
                if cfg.credit_mode=='shapley_service' else marginal_physical_credit(info,env.config))
            previous_potential=new_potential
            current['reward']+=(cfg.gamma**current['duration'])*(float(r)+shaping-overstaff_cost)
            current['overstaff_cost']+=overstaff_cost
            current['overstaff_rover_steps']+=int(surplus)
            current['raw_reward']+=float(r)
            current['shaped_reward']+=float(r)+shaping-overstaff_cost
            current['marginal_reward']+=(cfg.gamma**current['duration'])*physical_credit
            current['duration']+=1
            current['terminal']=bool(term)
            current['truncated']=bool(trunc)
            physical.append(dict(step=int(info['transition_step']),reward=float(r),
                                 served=info['poi_served'].tolist(),
                                 serving_agents=info['serving_agents'].tolist(),
                                 poi_coverage=info['poi_coverage'].tolist(),
                                 poi_rewards=info['poi_rewards'].tolist(),
                                 overstaff_rover_steps=int(surplus),
                                 active=info['post_active_mask'].tolist(),
                                 positions=info['post_positions'].tolist(),
                                 choices=previous.tolist(),
                                 events=info['events_next_boundary'],
                                 decision_rovers=np.flatnonzero(decide).tolist(),
                                 decision_reasons=current['decision_reasons'] if decide.any() else {}))
            # Always reset a newly deactivated agent's GRU at the *actual*
            # boundary. Don't feed its lifecycle to any other rover's actor.
            after_active=env.active.copy()
            dropped=before_active & ~after_active
            newly_active=~before_active & after_active
            reset=dropped | newly_active
            for i in np.flatnonzero(dropped):
                if last_decision[i] >= 0:
                    options.append(dict(rover=int(i),start=int(last_decision[i]),
                                        end=int(env.step_index),
                                        duration=int(env.step_index-last_decision[i]),
                                        reason='physical_failure_interrupt'))
                    last_decision[i]=-1
            # A fresh replacement slot is always a new identity. An optional
            # temporary reentry memory ablation only preserves disabled slots.
            if cfg.reset_on_reentry:
                reset_in_segment &= ~reset
                ah=ah*torch.as_tensor(~reset,device=device).unsqueeze(-1)
            if not (term or trunc):
                channel.advance(env,previous)
            if term or trunc:
                break
        current['memory_keep']=reset_in_segment.copy()
        records.append(current)
        # Terminal vs time-limit truncation: terminal never bootstraps.
        bootstrap=(0. if classical_eval or records[-1]['terminal'] else float(
            critic.forward_step(tensorize(critic_state(env),device),ch)[0].item())*cfg.value_output_scale)
    for i in np.flatnonzero(last_decision>=0):
        options.append(dict(rover=int(i),start=int(last_decision[i]),
                            end=int(env.step_index),duration=int(env.step_index-last_decision[i]),
                            reason='terminal' if records[-1]['terminal'] else 'horizon_censor'))
    vals=np.asarray([r['old_value'] for r in records],np.float32)
    advantages,returns=macro_gae([r['reward'] for r in records],vals,
                                 np.r_[vals[1:],bootstrap],
                                 [r['duration'] for r in records],
                                 [r['terminal'] for r in records],cfg.gamma,cfg.lam,
                                 lambda_per_step=(cfg.decision_mode=='asynchronous'))
    for t,r in enumerate(records):
        r['advantage']=float(advantages[t]);r['return']=float(returns[t])
    future=np.zeros(N,dtype=np.float64)
    for r in reversed(records):
        future=np.where(r['memory_keep'],future,0.)
        future=np.asarray(r['marginal_reward'],dtype=np.float64)+(cfg.gamma**r['duration'])*future
        r['marginal_return']=future.astype(np.float32).copy()
    return dict(records=records,team_return=raw_return,physical=physical,
                on_policy_compatible=(not classical_eval),
                overstaff_cost=sum(r['overstaff_cost'] for r in records),
                overstaff_rover_steps=sum(r['overstaff_rover_steps'] for r in records),
                decisions=len(records),agent_decisions=sum(int(r['decide'].sum()) for r in records),
                options=options,seed=seed,
                intervention_status=getattr(env,'intervention_status',None))
