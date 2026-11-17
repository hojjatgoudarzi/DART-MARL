"""Pre-registered, stage-gated 12-rover learning distribution.

Training uses the stage's generated tasks. Advancement depends on fixed,
SEPARATE development seeds, not on training return or final held-out seeds.
Final-test seeds must never enter this module. No allocator demonstrations.
"""
from dataclasses import dataclass
from dataclasses import replace
from .scenarios import create_scenario

STAGES=('single_service','coupled_nominal','heterogeneous_nominal',
        'heterogeneous_faults','robust_faults')


def stage_scenario(base, seed, stage, distribution="v22"):
    if stage not in STAGES:
        raise ValueError(stage)
    if distribution=='v24':
        from .scenarios_adaptive import scenario_v24
        scene=scenario_v24(base,seed,faults=stage in STAGES[3:],
                           stress=stage=='robust_faults')
        couplings={'single_service':(1,1,1,1,1,1),
                   'coupled_nominal':(2,2,2,2,1,1),
                   'heterogeneous_nominal':(3,3,3,3,1,1),
                   'heterogeneous_faults':(3,3,3,3,1,1),
                   'robust_faults':(3,3,3,3,1,1)}
        pois=tuple(replace(p,coupling=couplings[stage][j],
                           required_skill=None if stage in STAGES[:2] else p.required_skill)
                   for j,p in enumerate(scene.pois))
        return replace(scene,name=f'hr_v24_{stage}_{seed}',pois=pois)
    if distribution=='v23':
        from .scenarios_heterogeneous import scenario_v23,STAGE_COUPLING
        fault=stage in STAGES[3:]
        scene=scenario_v23(base,seed,faults=fault,severe=stage=='robust_faults')
        pois=tuple(replace(p,coupling=STAGE_COUPLING[stage][j],
                           required_skill=None if stage in STAGES[:2] else p.required_skill)
                   for j,p in enumerate(scene.pois))
        return replace(scene,name=f'hr_v23_{stage}_{seed}',pois=pois)
    if distribution!='v22':
        raise ValueError(distribution)
    hetero=stage in STAGES[2:]
    fault=stage in STAGES[3:]
    scenario=create_scenario(base,seed,
        ('recovery_challenge' if fault else 'recovery_nominal') if hetero else 'random_nominal')
    if not fault:
        scenario=replace(scenario,events=())
    if stage == 'single_service':
        # Still 12 active rovers; only physical coupling is initially relaxed.
        pois=tuple(replace(p,coupling=1,required_skill=None) for p in scenario.pois)
        return replace(scenario,name=f'hr_{stage}_{seed}',pois=pois)
    if stage == 'coupled_nominal':
        pois=tuple(replace(p,required_skill=None) for p in scenario.pois)
        return replace(scenario,name=f'hr_{stage}_{seed}',pois=pois)
    if stage == 'heterogeneous_nominal':
        return replace(scenario,name=f'hr_{stage}_{seed}',events=())
    if stage == 'robust_faults':
        # Preserves recovery challenge physics but extends communication
        # uncertainty via curriculum/config; legacy detection_delay is NOT
        # a privileged actor message. Evaluate delayed local heartbeats.
        events=tuple(replace(ev,detection_delay=ev.detection_delay+2) for ev in scenario.events)
        return replace(scenario,name=f'hr_{stage}_{seed}',events=events)
    return replace(scenario,name=f'hr_{stage}_{seed}')


@dataclass
class GateState:
    index: int = 0
    updates_in_stage: int = 0
    attempts: int = 0

    @property
    def name(self):
        return STAGES[self.index]

    def record_update(self):
        self.updates_in_stage+=1

    def assess(self, actor_return, random_return, min_updates, margin):
        self.attempts+=1
        passed_gate=(self.updates_in_stage>=min_updates and
                     actor_return >= random_return+margin)
        if passed_gate and self.index<len(STAGES)-1:
            self.index+=1
            self.updates_in_stage=0
            return True
        return False
