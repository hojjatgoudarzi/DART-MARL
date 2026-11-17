"""Physical coalition efficiency and right-censored fault recovery statistics.

Service at transition t-1 is the *pre-event* reference for an event applied
at decision boundary t. Recovery requires subsequent, sustained *physical*
service. Multiple affected rovers at one boundary are one task interruption,
not multiple fabricated independent samples. None of the metrics is fed into
execution-time task selection.
"""
from collections import Counter, defaultdict
import numpy as np

DISRUPTIONS={'temporary_disable','permanent_failure','replacement_agent',
             'capability_degradation'}
ACTIVE_REPAIR={'active_coalition_reallocation','replacement_contribution'}


def _staff(x,p):
    if 'poi_coverage' in x:
        return np.asarray(x['poi_coverage'],dtype=int)[:p]
    serving=np.asarray(x.get('serving_agents',[]),dtype=int)
    return np.bincount(serving[serving>=0],minlength=p)[:p]


def physical_metrics(trace, config, sustain_steps=2):
    if not trace: raise ValueError('empty physical trace')
    if sustain_steps<1: raise ValueError('sustain_steps must be >=1')
    tasks=config.pois;p=len(tasks)
    by_step={int(x['step']):x for x in trace}
    if len(by_step)!=len(trace): raise ValueError('duplicate physical step in trace')
    service=np.asarray([[bool(x['served'][j]) for j in range(p)] for x in trace],dtype=bool)
    staff=np.asarray([_staff(x,p) for x in trace],dtype=np.int64)
    if (staff<0).any(): raise ValueError('negative physical task count')
    coupling=np.asarray([task.coupling for task in tasks],dtype=np.int64)
    available=np.asarray([[int(task.window[0])<=int(x['step'])<=int(task.window[1])
                         for task in tasks] for x in trace],dtype=bool)
    repeat=np.asarray([bool(task.repeat) for task in tasks],dtype=bool)
    coupled=np.asarray([task.coupling>1 for task in tasks],dtype=bool)
    # Actual saturation: redundant rover mass counts only if task is served.
    surplus=np.where(service,np.maximum(staff-coupling,0),0)
    missing=np.where(available&~service,np.maximum(coupling-staff,0),0)
    per_task_service=service.sum(axis=0).astype(int)
    valid_coupled=available[:,coupled]
    # Never interpret absent/unavailable one-shot work as missed guaranteed service.
    per_tick_served_repeat=(service&repeat).sum(axis=1)
    team_value=np.asarray([float(task.value) for task in tasks])
    # This is only an unconstrained per-task upper bound, not the value that
    # a feasible controller could necessarily recover with shared resources.
    not_served_value_bound=((available&~service)*team_value).sum(axis=1)

    selection=Counter();last=None;switches=0;total_active=0;idle=0
    for x in trace:
        choices=x['choices']
        if last is not None:
            switches+=sum(bool(active and old_active and a!=b and a>=0 and b>=0)
                          for active,old_active,a,b in zip(x['active'],last['active'],choices,last['choices']))
        last=x
        for active,choice in zip(x['active'],choices):
            if active:
                total_active+=1;selection[int(choice)]+=1;idle+=int(choice<0)

    event_list=[ev for x in trace for ev in x['events'] if ev['kind'] in DISRUPTIONS]
    event_by_step=defaultdict(list)
    for ev in event_list:event_by_step[int(ev['step'])].append(ev)
    recoveries=[];non_disruptive=0;exposed_faults=set();processed=set()
    for at,events in sorted(event_by_step.items()):
        before=by_step.get(at-1);after=by_step.get(at)
        if before is None or after is None:continue
        for j,task in enumerate(tasks):
            if not task.repeat or not before['served'][j] or (at,j) in processed:continue
            pre=np.asarray(before.get('serving_agents',[]))
            members=set(np.flatnonzero(pre==j).tolist())
            # Older synthetic tests omit contributor IDs entirely. Such
            # incidents can be right-censored, but never labeled as active
            # repair without a real post-fault membership observation.
            affected=([ev for ev in events if int(ev.get('rover',-1)) in members]
                      if pre.size else events)
            if not affected:continue
            processed.add((at,j));exposed_faults.add((at,j))
            if after['served'][j]:
                non_disruptive+=1
                continue
            # Right-censor when another disruption occurs or task expires.
            next_fault=min((int(x) for x in event_by_step if int(x)>at),default=config.horizon)
            limit=min(next_fault,config.horizon,int(task.window[1])+1)
            candidate=[by_step[t] for t in range(at,limit) if t in by_step]
            hit=None
            for k in range(len(candidate)-sustain_steps+1):
                if all(bool(x['served'][j]) for x in candidate[k:k+sustain_steps]):
                    hit=int(candidate[k]['step']);break
            kind=affected[0]['kind'];recovered=hit is not None
            new_members=[];retasked=[];new_from_idle=[];label='censored'
            victims=sorted({int(ev.get('rover',-1)) for ev in affected})
            if recovered:
                post=by_step[hit]
                final_members=set(np.flatnonzero(np.asarray(post.get('serving_agents',[]))==j).tolist())
                new_members=sorted(final_members-members)
                natural=any(e['kind']=='automatic_reentry' and int(e['rover']) in victims
                            and at<int(e['step'])<=hit for x in trace for e in x['events'])
                if natural and any(v in final_members for v in victims):label='natural_reentry'
                elif any(i>=len(config.rovers) for i in new_members):label='replacement_contribution'
                elif new_members:label='active_coalition_reallocation'
                else:label='service_resumption_unattributed'
                for i in new_members:
                    prev=int(before['choices'][i])
                    if prev>=0 and prev!=j:retasked.append(i)
                    if prev<0:new_from_idle.append(i)
            recoveries.append(dict(step=at,kind=kind,trigger_rovers=victims,
                 rover=victims[0],poi=j,recovered=recovered,
                 recovery_lag=None if not recovered else hit-at,censor_at=limit,
                 recovery_type=label,new_members=new_members,
                 retasked_from_other_poi=retasked,new_members_from_idle=new_from_idle,
                 pre_failure_members=sorted(members),
                 policy_experienced_service_drop=True))
    n=service.shape[0]
    coupled_services=int(service[:,coupled].sum())
    coupled_opps=int(available[:,coupled].sum())
    recovery_active=sum(x['recovery_type'] in ACTIVE_REPAIR for x in recoveries)
    return dict(
        team_return=float(sum(x['reward'] for x in trace)),
        physical_service_steps=per_task_service.tolist(),
        coupled_service_steps=coupled_services,
        selected_task_counts={str(k):int(v) for k,v in selection.items()},
        active_rover_steps=total_active,idle_fraction=float(idle/max(total_active,1)),
        task_switch_count=switches,
        coupled_service_rate=float(coupled_services/max(n*int(coupled.sum()),1)),
        coupled_eligible_service_rate=float(coupled_services/max(coupled_opps,1)),
        eligible_service_opportunities=available.sum(axis=0).astype(int).tolist(),
        one_shot_completed=sum(bool(per_task_service[j]) for j,t in enumerate(tasks) if not t.repeat),
        # v2.5 coalition efficiency -- report both coverage and surplus.
        physical_contributor_steps_per_poi=staff.sum(axis=0).astype(int).tolist(),
        surplus_contributor_steps_per_poi=surplus.sum(axis=0).astype(int).tolist(),
        total_surplus_contributor_steps=int(surplus.sum()),
        surplus_rover_steps_per_physical_step=float(surplus.sum()/n),
        overstaffed_served_poi_steps=int((surplus>0).sum()),
        physical_service_rover_steps=int((staff*service).sum()),
        mean_serving_rovers_per_served_poi_step=float((staff*service).sum()/max(int(service.sum()),1)),
        mean_concurrent_repeatable_pois=float(per_tick_served_repeat.mean()),
        steps_with_2plus_repeatable_pois=int((per_tick_served_repeat>=2).sum()),
        steps_with_3plus_repeatable_pois=int((per_tick_served_repeat>=3).sum()),
        concurrent_2plus_fraction=float((per_tick_served_repeat>=2).mean()),
        task_shortfall_rover_steps=int(missing.sum()),
        # Upper bound is NOT a feasible oracle benefit; expose name honestly.
        nominal_unserved_task_value_upper_bound=float(not_served_value_bound.sum()),
        recovery_lags_observed=[x['recovery_lag'] for x in recoveries if x['recovered']],
        recovery_events=recoveries,recovery_censored=sum(not x['recovered'] for x in recoveries),
        recovery_natural_reentry=sum(x['recovery_type']=='natural_reentry' for x in recoveries),
        recovery_active_reallocation=recovery_active,
        recovery_after_retask=sum(bool(x['retasked_from_other_poi']) for x in recoveries),
        recovery_after_idle_recruitment=sum(bool(x['new_members_from_idle']) for x in recoveries),
        disrupted_service_events=len(recoveries),
        faults_without_immediate_service_drop=non_disruptive,
        recovery_sustain_steps=sustain_steps,
        active_recovery_rate=(recovery_active/len(recoveries) if recoveries else None),
        restored_service_rate=(sum(bool(x['recovered']) for x in recoveries)/len(recoveries) if recoveries else None),
        fault_count=len(event_list),
        faults_exposed_to_served_coalition=len(exposed_faults),
        recovery_opportunity_absent=(len(recoveries)==0),
        recovery_opportunities=len(recoveries),
        service_continuity_under_fault=non_disruptive)
