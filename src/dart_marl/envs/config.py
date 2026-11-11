"""Validated YAML-to-dataclass configuration with explicit schema version."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dart_marl.disruptions.events import DisruptionEvent

SCHEMA_VERSION = 1


def _xy(value: list[float] | tuple[float, ...]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("all positions and dimensions must be two-dimensional")
    return float(value[0]), float(value[1])


@dataclass(frozen=True, slots=True)
class RoverSpec:
    start: tuple[float, float]
    speed: float = 1.0
    sensor_radius: float = 12.0
    sensors: int = 4
    skill: int = 0

    def __post_init__(self) -> None:
        if self.speed < 0 or self.sensor_radius < 0 or self.sensors < 1:
            raise ValueError("invalid rover speed/radius/sensors")
        if self.skill < 0:
            raise ValueError("skill must be nonnegative")


@dataclass(frozen=True, slots=True)
class POISpec:
    position: tuple[float, float]
    radius: float
    coupling: int
    value: float
    window: tuple[int, int]
    repeat: bool = True
    objective: int = 0
    required_skill: int | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.coupling < 1 or self.value < 0:
            raise ValueError("invalid POI radius/coupling/value")
        if self.window[0] < 0 or self.window[1] < self.window[0]:
            raise ValueError("POI window requires 0 <= start <= end")
        if self.objective < 0 or (self.required_skill is not None and self.required_skill < 0):
            raise ValueError("objective/required_skill must be nonnegative")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    name: str
    dimensions: tuple[float, float]
    horizon: int
    rovers: tuple[RoverSpec, ...]
    pois: tuple[POISpec, ...]
    max_agents: int = 12
    max_pois: int = 6
    observation_mode: str = "research"  # or professor_density/professor_count
    poi_sensor_temperature: float = 2.0
    rover_sensor_temperature: float = 2.0
    spawn_jitter: float = 0.0
    events: tuple[DisruptionEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if any(x <= 0 for x in self.dimensions) or self.horizon < 1:
            raise ValueError("dimensions and horizon must be positive")
        if not self.rovers or not self.pois:
            raise ValueError("at least one rover and one POI required")
        if len(self.rovers) > self.max_agents or len(self.pois) > self.max_pois:
            raise ValueError("max_agents/max_pois too small for configured entities")
        if len({r.sensors for r in self.rovers}) != 1:
            raise ValueError("version 1 requires uniform number of directional sensors")
        if self.observation_mode not in {"research", "professor_density", "professor_count"}:
            raise ValueError("unrecognized observation_mode")
        if self.poi_sensor_temperature <= 0 or self.rover_sensor_temperature <= 0:
            raise ValueError("sensor temperatures must be positive")
        if self.spawn_jitter < 0:
            raise ValueError("spawn_jitter must be nonnegative")
        for rover in self.rovers:
            if not all(0 <= p <= d for p, d in zip(rover.start, self.dimensions)):
                raise ValueError("rover start outside map")
        for poi in self.pois:
            if not all(0 <= p <= d for p, d in zip(poi.position, self.dimensions)):
                raise ValueError("POI outside map")
        for event in self.events:
            if event.step >= self.horizon:
                raise ValueError("event step must be before horizon")
            if event.rover >= self.max_agents or (
                event.new_slot is not None and event.new_slot >= self.max_agents
            ):
                raise ValueError("event references an out-of-range rover slot")
            if event.spawn is not None and not all(
                0 <= p <= d for p, d in zip(event.spawn, self.dimensions)
            ):
                raise ValueError("event spawn outside map")
        if any(event.rover >= len(self.rovers) and event.kind != "replacement_agent"
               for event in self.events):
            raise ValueError("event references absent initial rover")

    @property
    def num_objectives(self) -> int:
        return max(p.objective for p in self.pois) + 1

    @property
    def num_sensors(self) -> int:
        return self.rovers[0].sensors


def config_from_mapping(raw: dict) -> EnvironmentConfig:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    rovers = tuple(
        RoverSpec(
            start=_xy(r["start"]),
            speed=float(r.get("speed", 1)),
            sensor_radius=float(r.get("sensor_radius", 12)),
            sensors=int(r.get("sensors", 4)),
            skill=int(r.get("skill", 0)),
        )
        for r in raw["rovers"]
    )
    pois = tuple(
        POISpec(
            position=_xy(p["position"]),
            radius=float(p["radius"]),
            coupling=int(p["coupling"]),
            value=float(p["value"]),
            window=tuple(int(t) for t in p["window"]),
            repeat=bool(p.get("repeat", True)),
            objective=int(p.get("objective", 0)),
            required_skill=(int(p["required_skill"]) if p.get("required_skill") is not None else None),
        )
        for p in raw["pois"]
    )
    events = tuple(DisruptionEvent.from_mapping(e) for e in raw.get("events", []))
    return EnvironmentConfig(
        name=str(raw["name"]),
        dimensions=_xy(raw["dimensions"]),
        horizon=int(raw["horizon"]),
        max_agents=int(raw.get("max_agents", 12)),
        max_pois=int(raw.get("max_pois", 6)),
        rovers=rovers,
        pois=pois,
        observation_mode=str(raw.get("observation_mode", "research")),
        poi_sensor_temperature=float(raw.get("poi_sensor_temperature", 2)),
        rover_sensor_temperature=float(raw.get("rover_sensor_temperature", 2)),
        spawn_jitter=float(raw.get("spawn_jitter", 0)),
        events=events,
    )


def load_config(path: str | Path) -> EnvironmentConfig:
    with Path(path).open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config_from_mapping(raw)
