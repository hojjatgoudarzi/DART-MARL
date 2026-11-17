"""Explicit overlap-attribution ablation, without modifying legacy RoverEnv.

'intent' preserves original physical rules and self-selected task tie breaking.
'declared' requires BOTH rover-selected task ID and physical eligible presence.
'neutral' attributes each rover to the geometrically nearest eligible POI,
regardless of selected task; this tests role-bookkeeping sensitivity.
"""
import numpy as np
from hr_mappo.envs.rover import RoverEnv

class NeutralAttributionEnv(RoverEnv):
    def _physical_coverage(self, t):
        assigned=np.full(self.config.max_agents,-1,dtype=np.int32)
        counts=np.zeros(self.config.max_pois,dtype=np.int32)
        eligible=self._eligible_pois(t)
        for i in np.flatnonzero(self.active):
            candidates=[]
            for p in np.flatnonzero(eligible):
                task=self.config.pois[p]
                if task.required_skill is not None and self.skills[i]!=task.required_skill:
                    continue
                dist=float(np.linalg.norm(self.positions[i]-self.poi_positions[p]))
                if dist < task.radius:
                    candidates.append((dist/task.radius,int(p)))
            if candidates:
                _,p=min(candidates)
                assigned[i]=p;counts[p]+=1
        served=np.zeros(self.config.max_pois,dtype=bool)
        for p,task in enumerate(self.config.pois):
            served[p]=eligible[p] and counts[p]>=task.coupling
        return counts,served,assigned


class DeclaredIntentEnv(RoverEnv):
    """v2.3 task contract: IDLE NEVER services tasks; no allocator or teleporting.

    A rover only contributes to its declared task, at its actual post-move
    position, while active, compatible, and inside an available POI's radius.
    The *same* subclass must be used for all same-information baselines.
    """
    def _physical_coverage(self, t):
        assigned=np.full(self.config.max_agents,-1,dtype=np.int32)
        counts=np.zeros(self.config.max_pois,dtype=np.int32)
        eligible=self._eligible_pois(t)
        for i in np.flatnonzero(self.active):
            p=int(self.roles[i])
            if p<0 or p>=len(self.config.pois) or not eligible[p]:
                continue
            task=self.config.pois[p]
            if task.required_skill is not None and self.skills[i]!=task.required_skill:
                continue
            if np.linalg.norm(self.positions[i]-self.poi_positions[p]) < task.radius:
                assigned[i]=p
                counts[p]+=1
        served=np.zeros(self.config.max_pois,dtype=bool)
        for p,task in enumerate(self.config.pois):
            served[p]=eligible[p] and counts[p]>=task.coupling
        return counts,served,assigned


def marginal_physical_credit(info, scenario):
    """TRAINING-ONLY instantaneous leave-one-rover-out physical service credit.

    This is NOT D++ and is NOT a full trajectory counterfactual. Removing
    the rover from the post-move service set leaves the POI unserved only if
    it was pivotal (coverage == coupling). Credits use actual POI rewards
    and are deliberately zero for non-pivotal agents. Never feed into actor
    observations or evaluation return.
    """
    n=scenario.max_agents
    credit=np.zeros(n,dtype=np.float32)
    counts=np.asarray(info['poi_coverage'])
    for p,poi in enumerate(scenario.pois):
        if info['poi_served'][p] and counts[p]==poi.coupling:
            credit[np.asarray(info['serving_agents'])==p]+=float(info['poi_rewards'][p])
    return credit


def shapley_service_credit(info, scenario):
    """Symmetric Shapley credit for ACTUAL, per-tick threshold service.

    For a task requiring c participants and served by n>=c eligible rovers,
    the characteristic game pays R iff a subset has at least c participants.
    Under a uniformly random order each agent is pivotal with probability
    1/n, so its exact Shapley credit is R/n (not a learned D++ estimate).
    Unserved POIs produce no physical credit. Only training uses this vector.
    One-shot rewards are split only at the real completion tick.
    """
    participants=np.asarray(info['serving_agents'])
    served=np.asarray(info['poi_served'],dtype=bool)
    rewards=np.asarray(info['poi_rewards'],dtype=np.float64)
    credit=np.zeros(scenario.max_agents,dtype=np.float32)
    for p,task in enumerate(scenario.pois):
        members=np.flatnonzero(participants==p)
        if served[p] and len(members)>=task.coupling and rewards[p]>0:
            credit[members]=float(rewards[p])/len(members)
    return credit


def physical_overstaff(info, scenario):
    """Count surplus *physical contributors* to actually serviced tasks.

    No role-only or future-schedule penalty, no penalty for incomplete tasks.
    The team receives at most one task reward regardless of extra members.
    """
    participants=np.asarray(info['serving_agents'])
    return sum(max(0,int(np.count_nonzero(participants==p))-task.coupling)
               for p,task in enumerate(scenario.pois) if info['poi_served'][p])


def service_potential(env):
    """Current-state potential (not an allocation oracle).

    Uses only actually present + compatible + self-declared participants.
    A partial group receives a *potential*, not physical service reward.
    Difference of potentials is a discount-consistent training-only signal.
    No remote state is passed to the decentralized actor.
    """
    counts,_,_=env._physical_coverage(env.step_index)
    value=0.
    for j,task in enumerate(env.config.pois):
        if task.window[0] <= env.step_index <= task.window[1] and not env.poi_completed[j]:
            value+=float(task.value)*min(int(counts[j]),task.coupling)/task.coupling
    return float(value)


def make_env(scenario, cfg):
    if cfg.service_attribution=='intent':
        return RoverEnv(scenario)
    if cfg.service_attribution=='declared':
        return DeclaredIntentEnv(scenario)
    if cfg.service_attribution=='neutral':
        return NeutralAttributionEnv(scenario)
    raise ValueError(cfg.service_attribution)
