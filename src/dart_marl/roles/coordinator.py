"""Event-triggered role coordinator with delayed-roster reports.

The only trigger based on disruption is a change in reported_active().
It never tests `env.active` to decide whether/when to replan. Actual command
application is masked by the environment, analogously to an undeliverable
message to a failed rover. The reported roster is retained separately from
physical actuator state. No privileged fault schedule is consulted.
"""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from dart_marl.algorithms.mappo.roles import assign_initial_roles
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.recovery import score_recovery_assignment


class RoleCoordinator:
    def __init__(self, config: AllocationConfig, *, profile_latency: bool = False,
                 controller_sha: str | None = None):
        self.config = config
        self.profile_latency = bool(profile_latency)
        self.controller_sha = controller_sha
        self.cft_estimator = None
        self.previous_commanded: np.ndarray | None = None
        self.last_roster: np.ndarray | None = None
        self.last_task_signature: tuple[tuple[bool, bool], ...] | None = None
        self.last_replan_step = -1
        self.decisions: list[dict[str, Any]] = []
        self.last_role_switch_step = 0

    @staticmethod
    def _task_signature(env: RoverEnv) -> tuple[tuple[bool, bool], ...]:
        return tuple((bool(env.poi_completed[p]),
                      poi.window[0] <= env.step_index <= poi.window[1])
                     for p, poi in enumerate(env.config.pois))

    def reset(self, env: RoverEnv) -> dict[str, Any]:
        """Same initial roles for every method, enabling matched post-fault comparisons."""
        if self.config.method in ("cft", "cft_rank"):
            from dart_marl.cft.training import CFTEstimator
            self.cft_estimator = CFTEstimator(
                self.config.cft_model_path, expected_config=env.config,
                expected_controller_sha=self.controller_sha, strict_controller=True,
                horizon=self.config.forecast_horizon,
                discount=self.config.forecast_discount)
            if self.config.method == "cft_rank" and self.cft_estimator.format != 'dart-cft-rank-v2':
                raise ValueError('cft_rank requires legacy_study ranking model')
        assign_initial_roles(env, "initial_coupling")
        self.previous_commanded = env.roles.copy()
        self.last_roster = env.reported_active().copy()
        self.last_task_signature = self._task_signature(env)
        self.last_replan_step = int(env.step_index)
        self.decisions = []
        self.last_role_switch_step = int(env.step_index)
        return self._record(env, "shared_initial", 0.0, 0, ())

    def _record(self, env: RoverEnv, trigger: str, objective: float,
                feasible_coalitions: int, projected_tasks: tuple[int, ...]
                ) -> dict[str, Any]:
        assert self.previous_commanded is not None
        roles = self.previous_commanded.copy()
        delivered = roles.copy()
        # Delivery failure filtering occurs only in the simulator interface;
        # it does not inform planning or trigger timing.
        delivered[~env.active] = -1
        if not np.array_equal(delivered, env.roles):
            env.set_roles(delivered)
        log = {
            "step": int(env.step_index), "trigger": trigger,
            "method": self.config.method,
            "reported_roster": env.reported_active().astype(int).tolist(),
            "commanded_roles": roles.tolist(), "delivered_roles": delivered.tolist(),
            "objective": float(objective),
            "feasible_coalitions": feasible_coalitions,
            "projected_tasks": list(projected_tasks),
        }
        self.decisions.append(log)
        return log

    def maybe_reassign(self, env: RoverEnv) -> dict[str, Any] | None:
        """Invoke after simulator step (its next-decision boundary is already applied)."""
        if self.previous_commanded is None or self.last_roster is None:
            raise RuntimeError("reset coordinator at start of each episode")
        if env.is_done:
            return None
        if self.config.method == "fixed":
            return None
        roster = env.reported_active().copy()
        task_signature = self._task_signature(env)
        roster_changed = not np.array_equal(roster, self.last_roster)
        task_changed = task_signature != self.last_task_signature
        periodic = (self.config.periodic_steps > 0 and
                    env.step_index - self.last_replan_step >= self.config.periodic_steps)
        if not (roster_changed or task_changed or periodic):
            return None
        trigger = "+".join([x for x, yes in (("roster_report", roster_changed),
                                                ("task_change", task_changed),
                                                ("periodic", periodic)) if yes])
        before_delivered = env.roles.copy()
        start_planning = perf_counter() if self.profile_latency else None
        if self.config.method in ("cft", "cft_rank"):
            from dart_marl.cft.selector import select_learned_assignment
            allocation = select_learned_assignment(env, roster, self.previous_commanded,
                                                   self.config, self.cft_estimator)
        else:
            allocation = allocate_roles(env, roster, self.previous_commanded, self.config)
        latency_ms = ((perf_counter() - start_planning) * 1000.0
                      if start_planning is not None else None)
        old_assignment = self.previous_commanded.copy()
        held = False
        selection_reason = "optimizer"
        predicted_advantage: float | None = None
        current_score = None
        if (self.config.method.startswith("r_cart") or self.config.method in ("cft", "cft_rank")) and (
            not np.array_equal(allocation.roles, old_assignment)
        ):
            c = self.config
            current_score = score_recovery_assignment(
                env, roster, old_assignment, old_assignment,
                horizon_steps=c.forecast_horizon, discount=c.forecast_discount,
                travel_weight=c.travel_weight,
                switch_weight=0.0 if c.method == "r_cart_no_switch" else c.switch_weight,
                sync_weight=0.0 if c.method == "r_cart_no_sync" else c.sync_weight,
                sync_model=("service" if c.method == "r_cart_service"
                            else "normalized" if c.method == "r_cart_bounded_sync" else "raw"))
            incumbent_objective = current_score.objective
            if current_score.feasible:
                if c.method in ("cft", "cft_rank"):
                    from dart_marl.cft.selector import learned_score
                    learned = learned_score(env, roster, old_assignment,
                                            old_assignment, c, self.cft_estimator)
                    assert learned is not None
                    incumbent_objective = learned[3]
                predicted_advantage = allocation.objective - incumbent_objective
                # Never hold a provably infeasible coalition; only valid incumbents
                # can be protected from thrashing by dwell and hysteresis.
                if c.method != "r_cart_no_hysteresis":
                    within_dwell = env.step_index - self.last_role_switch_step < c.min_dwell_steps
                    insufficient_gain = predicted_advantage <= c.hysteresis_margin
                    held = within_dwell or insufficient_gain
                    selection_reason = ("min_dwell" if within_dwell else "hysteresis_margin"
                                        if insufficient_gain else "optimizer")
            else:
                selection_reason = "infeasible_incumbent_override"
        if held:
            from dart_marl.roles.allocator import Allocation
            assert current_score is not None
            components = current_score.components()
            if self.config.method in ("cft", "cft_rank"):
                assert learned is not None
                components.update({'learned_return_residual':learned[2],
                                   'predicted_realized_horizon_return':learned[1],
                                   'learned_objective':learned[3]})
            allocation = Allocation(old_assignment.copy(), self.config.method,
                                    incumbent_objective, current_score.selected_tasks,
                                    current_score.travel, current_score.switches,
                                    allocation.feasible_coalitions, components)
        self.previous_commanded = allocation.roles.copy()
        self.last_roster = roster.copy()
        self.last_task_signature = task_signature
        self.last_replan_step = int(env.step_index)
        record = self._record(env, trigger, allocation.objective,
                              allocation.feasible_coalitions, allocation.projected_tasks)
        record["estimated_travel"] = allocation.estimated_travel
        if latency_ms is not None:
            record["planner_runtime_ms"] = latency_ms
        if self.config.method.startswith("r_cart") or self.config.method in ("cft", "cft_rank"):
            record["recovery_prediction"] = allocation.components
            record["hysteresis_held"] = held
            record["selection_reason"] = selection_reason
            record["predicted_improvement_over_incumbent"] = predicted_advantage
        record["role_switches"] = int(np.sum(env.active & (before_delivered != env.roles)))
        record["reported_role_switches"] = allocation.switches
        if record["role_switches"] > 0:
            self.last_role_switch_step = int(env.step_index)
        return record

    def summary(self) -> dict[str, Any]:
        changes = self.decisions[1:]
        return {"method": self.config.method, "config": asdict(self.config),
                "decision_count": len(changes),
                "role_switches": sum(int(e.get("role_switches", 0)) for e in changes),
                "decision_steps": [d["step"] for d in changes],
                "decisions": self.decisions}
