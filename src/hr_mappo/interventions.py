"""v2.4 policy-conditioned service-interruption stress test.

At a PRE-REGISTERED physical decision boundary (t=50), if a policy has
actually formed a repeatable-task coalition, the experimenter disables one
actual contributing rover. This deliberately conditions the target on
REALIZED POLICY BEHAVIOR. It is a diagnostic causal intervention only;
not the main fixed-fault matched scenario or a fair return ranking between
policies. Nothing about the intervention target is an Actor input.

This separate stress explicitly tests whether deployed agents can recover
when a serviced task actually loses a member. If no coalition is present,
the opportunity is absent and recovery rate is UNDEFINED, not zero.
"""
from dataclasses import replace
import numpy as np
from hr_mappo.disruptions.events import DisruptionEvent
from .physics import DeclaredIntentEnv


class ConditionalCoalitionFaultEnv(DeclaredIntentEnv):
    def __init__(self, scenario, intervention_step=50, mode="single"):
        if not 1 <= intervention_step < scenario.horizon:
            raise ValueError('intervention must occur inside episode')
        if scenario.events:
            raise ValueError('conditional-fault diagnostic requires nominal scenario without other events')
        if mode not in ('single','minimal_feasible_cut'):
            raise ValueError('unknown intervention mode')
        self.mode=mode
        self._base_scenario=scenario
        self.intervention_step=int(intervention_step)
        self.intervention_status=None
        self._injected=False
        super().__init__(scenario)

    def reset(self, *, seed=None):
        # The injected event is EPISODE-LOCAL. A second reset must restore
        # the original nominal tape; never retain a past policy's target.
        if hasattr(self, '_event_applied'):
            self.config=self._base_scenario
            self._event_applied=np.zeros(len(self.config.events),dtype=bool)
        self._injected=False
        self.intervention_status=None
        return super().reset(seed=seed)

    def _apply_boundary_events(self):
        if self.step_index==self.intervention_step and not self._injected:
            # Here, movement/service at t-1 has already been simulated.
            # The intervention is applied only at the NEXT decision boundary.
            _,served,assigned=self._physical_coverage(self.step_index-1)
            eligible=[j for j,p in enumerate(self.config.pois)
                      if p.repeat and bool(served[j])]
            self._injected=True
            # The single-victim legacy intervention is preserved for parity.
            # Minimal feasible cut removes n-c+1 contributors, but ONLY if at
            # least c compatible ACTIVE rovers remain in the whole team.
            # That is a structural feasibility condition, NOT a guarantee of
            # travel-time feasibility or learned recovery.
            candidates=[];had_service=bool(eligible)
            for target in eligible:
                task=self.config.pois[target]
                contributing=np.flatnonzero((assigned==target)&self.active)
                required=1 if self.mode=='single' else len(contributing)-task.coupling+1
                if required<=0:continue
                if self.mode=='minimal_feasible_cut':
                    compatible=sum(bool(self.active[i] and
                        (task.required_skill is None or self.skills[i]==task.required_skill))
                        for i in range(self.config.max_agents))
                    if compatible-required<task.coupling:continue
                candidates.append((float(task.value),-target,target,contributing,required))
            if candidates:
                _,_,target,contributing,required=max(candidates,key=lambda x:(x[0],x[1]))
                victims=sorted(map(int,contributing))[:required]
                injected=tuple(DisruptionEvent(step=self.step_index,
                     kind='permanent_failure',rover=v,detection_delay=5) for v in victims)
                self.config=replace(self.config,events=self.config.events+injected)
                self._event_applied=np.append(self._event_applied,np.zeros(len(injected),dtype=bool))
                self.intervention_status=dict(opportunity=True,step=int(self.step_index),
                         task=int(target),rover=victims[0],victims=victims,
                         mode=self.mode,required_coupling=int(self.config.pois[target].coupling),
                         target_cut_size=required,contributors_before=contributing.tolist(),
                         structural_feasibility_after_cut=True if self.mode=='minimal_feasible_cut' else None,
                         target_rule='highest-value eligible currently served repeatable task')
            else:
                self.intervention_status=dict(opportunity=False,step=int(self.step_index),
                     mode=self.mode,had_serviced_task=had_service,
                     reason=('no feasible, minimally-disruptive-to-feasibility cut'
                             if had_service else 'no repeatable POI was being served'))
        return super()._apply_boundary_events()
