"""Versioned seed-defined *new* task distribution; legacy benchmarks unchanged."""
from dataclasses import replace
import numpy as np
from hr_mappo.envs.config import load_config, POISpec, RoverSpec
from hr_mappo.disruptions.events import DisruptionEvent


def create_scenario(base, seed: int, variant='random_faults'):
    """Public map is different each episode; fault events never passed to actor.

    All evaluations should freeze seed lists independently of development seeds.
    """
    if variant == 'legacy':
        return base
    if variant in ('coordination_v24','coordination_v24_nominal','coordination_v24_heldout','coordination_v24_stress'):
        from .scenarios_adaptive import scenario_v24
        return scenario_v24(base,seed,heldout=variant=='coordination_v24_heldout',
                            faults=variant!='coordination_v24_nominal',
                            stress=variant=='coordination_v24_stress')
    if variant in ('coordination_v23','coordination_v23_nominal','coordination_v23_heldout','coordination_v23_stress'):
        from .scenarios_heterogeneous import scenario_v23
        return scenario_v23(base,seed,heldout=variant=='coordination_v23_heldout',
                            faults=variant!='coordination_v23_nominal',
                            severe=variant=='coordination_v23_stress')
    if variant in ('recovery_challenge','recovery_heldout','recovery_nominal'):
        return recovery_challenge(base,seed,heldout=(variant=='recovery_heldout'),
                                  faults=(variant!='recovery_nominal'))
    if variant not in ('random_faults', 'random_nominal', 'heterogeneous', 'stress'):
        raise ValueError(variant)
    g = np.random.default_rng(seed)
    rovers = []
    for i, r in enumerate(base.rovers):
        rovers.append(replace(r, start=tuple(g.uniform(8, 32, 2)),
                              speed=float(g.uniform(.55 if variant == 'stress' else .75, 1.5 if variant == 'stress' else 1.3)) if variant in ('heterogeneous','stress') else r.speed,
                              sensor_radius=float(g.uniform(9, 17)) if variant in ('heterogeneous','stress') else r.sensor_radius,
                              skill=i % 2 if variant in ('heterogeneous','stress') else r.skill))
    pois = []
    for j, p in enumerate(base.pois):
        skill = j % 2 if variant in ('heterogeneous','stress') and j < 4 else None
        pois.append(replace(p, position=tuple(g.uniform(2 if variant == 'stress' else 5, 38 if variant == 'stress' else 35, 2)),
                            radius=p.radius*(.8 if variant == 'stress' else 1.),
                            coupling=(3 if j == 0 else 2 if j < 4 else 1),
                            value=float(p.value * g.uniform(.75, 1.25)),
                            required_skill=skill))
    events = ()
    if variant != 'random_nominal':
        slots = g.choice(len(rovers), 4, replace=False)
        ticks = sorted(g.choice(np.arange(15, base.horizon - 12), 4, replace=False).tolist())
        events = (
            DisruptionEvent(kind='temporary_disable', step=ticks[0], rover=int(slots[0]), duration=8, detection_delay=3),
            DisruptionEvent(kind='capability_degradation', step=ticks[1], rover=int(slots[1]), speed_factor=.55, sensor_factor=.7, detection_delay=3),
            DisruptionEvent(kind='replacement_agent', step=ticks[2], rover=int(slots[2]), new_slot=12, detection_delay=4, spawn=tuple(g.uniform(8,32,2))),
            DisruptionEvent(kind='permanent_failure', step=ticks[3], rover=int(slots[3]), detection_delay=4),
        )
    return replace(base, name=f'hr_{variant}_{seed}', rovers=tuple(rovers), pois=tuple(pois),
                   events=events, spawn_jitter=0.0)


def recovery_challenge(base, seed, heldout=False, faults=True):
    """Difficult but feasible 12+2 coalition restoration distribution.

    Four repeated objectives: (4,4,3,3) simultaneous physically compatible
    agents; 6 agents of each of two skills; two reserve-skilled active agents.
    Spatial initial conditions create *possible*, not prescribed, service.
    Faults target pre-registered spatial cohorts, NEVER the learned actor's
    realized assignments, which avoids a policy-dependent fault oracle.
    Each trial randomizes task geometry, rover labels, speed, skill labeling,
    task values, window timing and the (unknown-to-actor) fault schedule.
    Held-out variant uses different geometric templates and wider jitter.
    """
    from hr_mappo.disruptions.events import DisruptionEvent
    g=np.random.default_rng(seed)
    assert len(base.rovers)==12 and base.max_agents==14 and len(base.pois)==6
    anchors=np.asarray([[9.,9.],[31.,9.],[9.,31.],[31.,31.],
                        [20.,17.],[20.,25.]],dtype=float)
    if heldout:
        anchors=np.asarray([[10.,15.],[30.,25.],[29.,9.],[11.,31.],
                            [18.,8.],[23.,32.]],dtype=float)
    # Random symmetries plus continuous independent per-task variation.
    transform=int(g.integers(4))
    for _ in range(transform):
        anchors=np.stack([40.-anchors[:,1],anchors[:,0]],axis=1)
    if bool(g.integers(2)):
        anchors[:,0]=40.-anchors[:,0]
    anchors=np.clip(anchors+g.uniform(-2.5 if heldout else -2.,
                                       2.5 if heldout else 2.,anchors.shape),4,36)
    skill_offset=int(g.integers(2))
    s0=skill_offset; s1=1-skill_offset
    pois=[]
    for j,p in enumerate(base.pois):
        skill=(s0 if j in (0,2) else s1) if j<4 else None
        # Each skill has only six active rovers: serving both of its
        # recurring POIs at once would require seven -> genuine allocation.
        coupled=(4,4,3,3,1,1)[j]
        high=base.horizon-1
        window=((0,68),(0,80),(38,high),(57,high),(17,63),(43,98))[j]
        pois.append(replace(p,position=tuple(anchors[j]),radius=(3.0 if j<4 else 2.7),
                            coupling=coupled,required_skill=skill,
                            value=float((2.6,2.8,4.1,4.4,7.,8.)[j]*g.uniform(.85,1.15)),
                            window=window,repeat=j<4))
    cohorts=[]
    # A meaningful, feasible initial arrangement: coalition agents plus one
    # reallocation-capable reserve per skill. Labels shuffled across episodes.
    for poi_idx, count, skill in ((0,4,s0),(1,4,s1),(2,1,s0),(3,1,s1)):
        for _ in range(count):
            start=np.clip(anchors[poi_idx]+g.normal(0,.55,2),2,38)
            cohorts.append((tuple(start),skill,poi_idx))
    for skill in (s0,s1):
        start=np.clip(np.array([20.,20.])+g.uniform(-4,4,2),2,38)
        cohorts.append((tuple(start),skill,-1))
    order=g.permutation(12).tolist()
    shuffled=[cohorts[int(j)] for j in order]
    rovers=[]
    for i,(start,skill,_) in enumerate(shuffled):
        rovers.append(replace(base.rovers[i],start=start,skill=skill,
                              speed=float(g.uniform(.75,1.25)),
                              sensor_radius=float(g.uniform(9.,14.))))
    def cohort_slot(p):
        return int(g.choice([i for i,(_,_,j) in enumerate(shuffled) if j==p]))
    if faults:
        # Always well-separated so recovery episodes can be right-censored
        # at the next event. Changes in schedule never reach Actor observations.
        times=[25+int(g.integers(-3,4)),53+int(g.integers(-3,4)),
               78+int(g.integers(-3,4)),103+int(g.integers(-3,4))]
        slots=[cohort_slot(k) for k in (0,1,2,3)]
        events=(DisruptionEvent(kind='temporary_disable',step=times[0],rover=slots[0],
                                duration=20,detection_delay=5),
                DisruptionEvent(kind='capability_degradation',step=times[1],rover=slots[1],
                                speed_factor=.38,sensor_factor=.6,detection_delay=6),
                DisruptionEvent(kind='replacement_agent',step=times[2],rover=slots[2],
                                new_slot=12,spawn=tuple(np.clip(g.uniform(14,26,2),0,40)),
                                detection_delay=6),
                DisruptionEvent(kind='permanent_failure',step=times[3],rover=slots[3],
                                detection_delay=7))
    else:
        events=()
    return replace(base,name=f'hr_recovery_{"heldout" if heldout else "development"}_{seed}',
                   rovers=tuple(rovers),pois=tuple(pois),events=events,spawn_jitter=0.)
