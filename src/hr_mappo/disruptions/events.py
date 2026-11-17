"""Explicit pre-decision event contracts. No implicit real-world fault detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal[
    "permanent_failure",
    "temporary_disable",
    "replacement_agent",
    "capability_degradation",
]


@dataclass(frozen=True, slots=True)
class DisruptionEvent:
    """Event at decision boundary ``step`` before action for that step.

    ``replacement_agent`` replaces ``rover`` with a previously unused ``new_slot``.
    ``temporary_disable`` automatically restores the same slot after ``duration``
    decision intervals; no restoration occurs past the episode horizon.
    Detection is a separate concern: ``detection_delay`` delays status *reports*
    to the coordinator but never delays physical failure.
    """

    step: int
    kind: EventKind
    rover: int
    duration: int | None = None
    new_slot: int | None = None
    spawn: tuple[float, float] | None = None
    speed_factor: float = 1.0
    sensor_factor: float = 1.0
    new_skill: int | None = None
    detection_delay: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"permanent_failure", "temporary_disable",
                             "replacement_agent", "capability_degradation"}:
            raise ValueError(f"unknown event kind: {self.kind}")
        if self.step < 0 or self.rover < 0 or self.detection_delay < 0:
            raise ValueError("event step/rover/detection_delay must be nonnegative")
        if self.kind == "temporary_disable" and (self.duration is None or self.duration < 1):
            raise ValueError("temporary_disable requires positive duration")
        if self.kind == "replacement_agent":
            if self.new_slot is None or self.new_slot < 0 or self.new_slot == self.rover:
                raise ValueError("replacement_agent requires distinct nonnegative new_slot")
        if self.kind == "capability_degradation":
            if not (0 <= self.speed_factor <= 1 and 0 <= self.sensor_factor <= 1):
                raise ValueError("degradation factors must be within [0,1]")
        if self.spawn is not None and len(self.spawn) != 2:
            raise ValueError("spawn must contain two coordinates")

    @classmethod
    def from_mapping(cls, data: dict) -> DisruptionEvent:
        data = data.copy()
        if "spawn" in data and data["spawn"] is not None:
            data["spawn"] = tuple(float(x) for x in data["spawn"])
        return cls(**data)
