"""Decentralized SAME-INFORMATION baselines (no env state allowed).

Every chooser receives only its own view from observe_agents. `FixedInitial`
locks a public-map decision for the whole rover life, including after
unavailability. It is the important v2.4 static-allocation comparator.
"""
import numpy as np


def choose(obs, previous, method, rng=None):
    n=obs['ego'].shape[0]
    p=obs['tasks'].shape[1]
    result=np.full(n,-1,np.int32)
    if method=='always_idle':
        return result
    for i in range(n):
        if not obs['active'][i]: continue
        legal=np.flatnonzero(obs['legal'][i,:p])
        if not len(legal): continue
        if method=='random':
            rng=np.random.default_rng(0) if rng is None else rng
            result[i]=int(rng.choice(np.r_[legal,-1])); continue
        if method=='static_highest_value_repeatable':
            repeated=legal[obs['tasks'][i,legal,8]>.5]
            if len(repeated):legal=repeated
            score=obs['tasks'][i,legal,5]
            result[i]=int(legal[np.argmax(score)])
            continue
        if method not in ('nearest','value_travel','persistent','coalition_local','window_aware'):
            raise ValueError(method)
        dist=obs['tasks'][i,legal,2]
        if method=='nearest': score=-dist
        else:
            values=obs['tasks'][i,legal,5]
            coupling=obs['tasks'][i,legal,4]
            speed=max(float(obs['ego'][i,2]),.05)
            score=values/(.1+dist/speed)*(1/(.4+coupling))
            if method in ('persistent','coalition_local'):
                score+=.1*(legal==previous[i])
            if method=='coalition_local':
                reports=obs['status_mask'][i]
                intentions=np.rint(obs['status'][i,reports,1]*(p+1)-1).astype(int)
                counts=np.array([np.count_nonzero(intentions==task) for task in legal])
                needs=np.maximum(1,np.rint(obs['tasks'][i,legal,4]*n).astype(int))
                score*=1+np.minimum(counts,needs-1)/needs
                score*=np.where(counts>=needs,0.75,1.0)
            if method=='window_aware':
                # Public availability windows; no dynamic fault state or
                # future fault tape. Relative distances from own position.
                now=float(obs['ego'][i,5])
                arrival=now+(dist/max(speed,1e-4)) * (1./6.)  # distance/(speed*horizon), 40x40/120
                start=obs['tasks'][i,legal,6]
                end=obs['tasks'][i,legal,7]
                remaining=np.maximum(0,end-np.maximum(arrival,start))
                future_wait=np.maximum(0,start-arrival)
                # Do not move early to a distant unopened task merely
                # because it has a longer remaining window.
                window_factor=np.minimum(remaining,.24)*np.exp(-4.*future_wait)
                score=values*window_factor/(.035+dist/max(speed,1e-4))
                score*=np.where(legal==previous[i],1.12,1.)
        result[i]=int(legal[np.argmax(score)])
    return result


class FixedInitial:
    """Frozen public-map task choice; returns IDLE once the task is illegal.

    The initial decision is made using only the agent's permitted view.
    A replacement slot gets its own first decision. No global allocation.
    """
    def __init__(self, method='static_highest_value_repeatable'):
        self.method=method
        self.assignment=None
        self.initialized=None

    def __call__(self, obs, env, previous):
        n=obs['ego'].shape[0];p=obs['tasks'].shape[1]
        if self.assignment is None:
            self.assignment=np.full(n,-1,np.int32)
            self.initialized=np.zeros(n,bool)
        new=np.flatnonzero(obs['active'] & ~self.initialized)
        if len(new):
            chosen=choose(obs,previous,self.method)
            self.assignment[new]=chosen[new]
            self.initialized[new]=True
        result=self.assignment.copy()
        for i in range(n):
            if not obs['active'][i] or result[i]<0 or not obs['legal'][i,result[i]]:
                result[i]=-1
        return result


def make_selector(method, rng=None):
    if method=='fixed_initial_value':
        return FixedInitial()
    if method=='fixed_initial_nearest':
        return FixedInitial('nearest')
    return lambda obs,env,prev:choose(obs,prev,method,rng)
