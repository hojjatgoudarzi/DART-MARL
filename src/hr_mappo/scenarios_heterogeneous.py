"""Versioned v2.3 12+2 task distribution, independent of legacy/v2.2.

Seeded maps, feasible task coalitions and nontrivial starts; no active
coupling is satisfied at reset. This is a fixed benchmark *distribution*,
not an RL-friendly per-policy oracle. Failure targets are chosen before
any policy rollout and are never conditioned on learned assignments.
"""
from dataclasses import replace
import numpy as np
from hr_mappo.disruptions.events import DisruptionEvent

COUPLING = (4, 4, 3, 3, 1, 1)
STAGE_COUPLING = {
    'single_service': (1,1,1,1,1,1),
    'coupled_nominal': (2,2,2,2,1,1),
    'heterogeneous_nominal': (3,3,2,2,1,1),
    'heterogeneous_faults': COUPLING,
    'robust_faults': COUPLING,
}


def scenario_v23(base, seed, heldout=False, faults=True, severe=False):
    """Construct 12 active (6/skill) and 2 unused slots on a 40x40 world.

    Distinct heldout geometry is disjoint from training seeds by caller.
    Two same-skill POIs require 4+3 > six available, so choices compete.
    After every failure at least four skill-compatible agents remain for
    either of that skill's repeatable tasks (not simultaneous service).
    """
    if len(base.rovers)!=12 or base.max_agents!=14 or len(base.pois)!=6:
        raise ValueError('v2.3 distribution requires 12 active + 2 slots and six POIs')
    g=np.random.default_rng(seed)
    anchors=np.array([[9.,9.],[31.,9.],[10.,31.],[30.,31.],[20.,14.],[20.,26.]],float)
    if heldout:
        anchors=np.array([[11.,14.],[29.,26.],[29.,11.],[11.,29.],[21.,7.],[19.,33.]],float)
    for _ in range(int(g.integers(4))):
        anchors=np.column_stack([40.-anchors[:,1], anchors[:,0]])
    if int(g.integers(2)):
        anchors[:,0]=40.-anchors[:,0]
    anchors=np.clip(anchors+g.uniform(-2.25 if not heldout else -3.0,
                                       2.25 if not heldout else 3.0, anchors.shape),3.0,37.0)
    flip=int(g.integers(2))
    skills=[flip if k in (0,2) else 1-flip if k in (1,3) else None for k in range(6)]
    windows=((0,110),(0,110),(25,119),(30,119),(16,69),(45,102))
    values=(2.7,2.7,4.2,4.2,7.5,8.)
    pois=tuple(replace(p,position=tuple(anchors[k]),coupling=COUPLING[k],
                       radius=3.1 if k<4 else 2.7,
                       required_skill=skills[k],window=windows[k],
                       repeat=k<4,value=float(values[k]*g.uniform(.85,1.15)))
               for k,p in enumerate(base.pois))
    rover_skills=np.array([flip]*6+[1-flip]*6)
    g.shuffle(rover_skills)
    starts=[]
    for i,s in enumerate(rover_skills):
        # The bias is spatially symmetric between same-skill tasks; it does
        # NOT assign a task or create a completed coalition at reset.
        compatible=[k for k in range(4) if skills[k]==s]
        for _ in range(20000):
            center=anchors[int(g.choice(compatible))]
            angle=float(g.uniform(0,2*np.pi));dist=float(g.uniform(8,16))
            pos=np.clip(center+dist*np.array([np.cos(angle),np.sin(angle)]),2.,38.)
            if any(np.linalg.norm(pos-anchors[k]) < pois[k].radius+2.5 for k in range(6)):
                continue
            if starts and min(np.linalg.norm(pos-np.asarray(q)) for q in starts)<.9:
                continue
            if min(np.linalg.norm(pos-anchors[k]) for k in compatible) > 20:
                continue
            starts.append(tuple(pos));break
        else:
            raise RuntimeError('v2.3 start generator could not place feasible rovers')
    rovers=tuple(replace(r,start=starts[i],skill=int(rover_skills[i]),
                         speed=float(g.uniform(.85,1.25)),
                         sensor_radius=float(g.uniform(10,16)))
                 for i,r in enumerate(base.rovers))
    events=()
    if faults:
        # Targets are sampled from fixed starting geometry, independently of
        # any actor decisions. Both skills retain feasible surviving groups.
        skill0=[i for i,r in enumerate(rovers) if r.skill==flip]
        skill1=[i for i,r in enumerate(rovers) if r.skill!=flip]
        def near(pool,task,exclude=()):
            remaining=[i for i in pool if i not in exclude]
            return min(remaining,key=lambda i:np.linalg.norm(np.asarray(rovers[i].start)-anchors[task]))
        p=near(skill0,0)
        q=near(skill1,1)
        j=near(skill0,2,(p,))
        k=near(skill1,3,(q,))
        jitter=lambda: int(g.integers(-3,4))
        events=(
            DisruptionEvent(step=37+jitter(),kind='permanent_failure',rover=p,detection_delay=5),
            DisruptionEvent(step=61+jitter(),kind='temporary_disable',rover=q,
                            duration=46 if not severe else 55,detection_delay=6),
            DisruptionEvent(step=82+jitter(),kind='replacement_agent',rover=j,new_slot=12,
                            spawn=tuple(np.clip(anchors[2]+g.uniform(-7,7,2),2,38)),detection_delay=5),
            DisruptionEvent(step=100+jitter(),kind='capability_degradation',rover=k,
                            speed_factor=.35,sensor_factor=.55,detection_delay=7))
    variant='heldout' if heldout else 'dev'
    if severe: variant += '_severe'
    scene=replace(base,name=f'hr_v23_{variant}_{seed}',rovers=rovers,pois=pois,
                  events=events,spawn_jitter=0.)
    # Check objective feasibility after worst-case lost members: the repair
    # is *physically possible*, not guaranteed by any algorithm.
    for task in scene.pois[:4]:
        assert sum(r.skill==task.required_skill for r in scene.rovers)-2 >= task.coupling
    return scene
