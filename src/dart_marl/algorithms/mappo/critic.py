"""Masked agent/POI entity critic; one shared-team value per participating rover."""

from __future__ import annotations

import math

import torch
from torch import nn


class CentralizedCritic(nn.Module):
    """Per-agent V(s, i), conditioned on globally available CTDE state.

    Masked pooling prevents padded/failed rovers from contaminating estimates;
    a task-attention block gives each rover context about the POIs.
    """

    def __init__(self, agent_dim: int = 9, poi_dim: int = 10, hidden: int = 128):
        super().__init__()
        self.agent_net = nn.Sequential(nn.Linear(agent_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh())
        self.poi_net = nn.Sequential(nn.Linear(poi_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.hidden = hidden

    def forward(self, agents: torch.Tensor, pois: torch.Tensor,
                active: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        """Shapes: agents [B,N,A], pois [B,P,F], masks [B,N]/[B,P]."""
        if agents.ndim != 3 or pois.ndim != 3:
            raise ValueError("critic inputs must have shapes [batch, agents, features]")
        alive = active.bool()
        tasks = task_mask.bool()
        agent_vec = self.agent_net(agents)
        poi_vec = self.poi_net(pois)
        af = alive.unsqueeze(-1).to(agent_vec.dtype)
        pf = tasks.unsqueeze(-1).to(poi_vec.dtype)
        team = (agent_vec * af).sum(dim=1, keepdim=True) / af.sum(dim=1, keepdim=True).clamp(min=1)
        task_global = (poi_vec * pf).sum(dim=1, keepdim=True) / pf.sum(dim=1, keepdim=True).clamp(min=1)
        attn = torch.matmul(self.query(agent_vec), self.key(poi_vec).transpose(1, 2))
        attn = attn / math.sqrt(self.hidden)
        attn = attn.masked_fill(~tasks.unsqueeze(1), -1e9)
        probs = torch.softmax(attn, dim=-1) * tasks.unsqueeze(1).to(attn.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        task_local = torch.matmul(probs, self.value(poi_vec))
        pooled = torch.cat([
            agent_vec, team.expand_as(agent_vec),
            task_global.expand_as(agent_vec), task_local,
        ], dim=-1)
        return self.head(pooled).squeeze(-1) * alive.to(agent_vec.dtype)


class TemporalTeamCritic(nn.Module):
    """V2: shared TEAM value, explicit finite-horizon time and task proximity.

    The old critic returns one value per rover even though the reward is shared.
    Here one exchangeable team estimate is broadcast to participating rovers,
    preserving the existing PPO buffer contract. No padded/failed slot is used
    in the team encoding. `return_scale` multiplies the normalized network
    output into raw environment reward units; the PPO value loss divides by
    the same scale. This is intentionally a different checkpoint architecture.
    """

    def __init__(self, agent_dim: int = 9, poi_dim: int = 10,
                 hidden: int = 128, return_scale: float = 64.0):
        super().__init__()
        if not (math.isfinite(return_scale) and return_scale > 0):
            raise ValueError("return_scale must be finite and positive")
        self.return_scale = float(return_scale)
        self.agent_encoder = nn.Sequential(nn.Linear(agent_dim, hidden), nn.SiLU(),
                                           nn.Linear(hidden, hidden), nn.SiLU())
        self.poi_encoder = nn.Sequential(nn.Linear(poi_dim, hidden), nn.SiLU(),
                                         nn.Linear(hidden, hidden), nn.SiLU())
        # 2 exchangeable masked aggregates and 8 physical, finite-horizon stats.
        self.head = nn.Sequential(nn.Linear(2 * hidden + 8, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))
        self.time_skip = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.time_skip.weight)

    def forward(self, agents: torch.Tensor, pois: torch.Tensor,
                active: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        if agents.ndim != 3 or pois.ndim != 3 or active.shape != agents.shape[:2] or task_mask.shape != pois.shape[:2]:
            raise ValueError("invalid [B,N,A], [B,P,F] critic input shapes")
        if agents.shape[-1] != 9 or pois.shape[-1] != 10:
            raise ValueError("v2 expects canonical 9-agent and 10-task features")
        alive, tasks = active.bool(), task_mask.bool()
        a = alive.unsqueeze(-1).to(agents.dtype)
        p = tasks.unsqueeze(-1).to(pois.dtype)
        n_active = a.sum(dim=1).clamp(min=1)
        n_tasks = p.sum(dim=1).clamp(min=1)
        encoded_agents = (self.agent_encoder(agents) * a).sum(dim=1) / n_active
        encoded_tasks = (self.poi_encoder(pois) * p).sum(dim=1) / n_tasks
        step = (agents[..., 8:9] * a).sum(dim=1) / n_active
        # Distances normalized to map size, with padded/disabled rovers excluded.
        distances = torch.cdist(agents[..., :2], pois[..., :2], p=2)
        distances = distances.masked_fill(~alive[:, :, None], 1e4)
        nearest = distances.min(dim=1).values
        task_float = tasks.to(agents.dtype)
        distance_mean = (nearest.clamp(max=2.0) * task_float).sum(dim=1, keepdim=True) / n_tasks
        task_radii = pois[..., 2]
        within_radius = (distances < task_radii[:, None, :]) & alive[:, :, None]
        counts = within_radius.sum(dim=1).to(agents.dtype)
        coupling = (pois[..., 3] * agents.shape[1]).clamp(min=1.0)
        coverage = torch.minimum(counts / coupling, torch.ones_like(counts))
        covered = (coverage * task_float).sum(dim=1, keepdim=True) / n_tasks
        available = ((pois[..., 8] < 0.5) & tasks).to(agents.dtype)
        stats = torch.cat([
            step, 1 - step,
            a.sum(dim=1) / agents.shape[1],
            p.sum(dim=1) / pois.shape[1],
            (pois[..., 4] * p.squeeze(-1)).sum(dim=1, keepdim=True) / n_tasks,
            (available * task_float).sum(dim=1, keepdim=True) / n_tasks,
            distance_mean, covered,
        ], dim=-1)
        val = self.head(torch.cat([encoded_agents, encoded_tasks, stats], dim=-1))
        val = (val + self.time_skip(1.0 - step)).squeeze(-1) * self.return_scale
        return val[:, None].expand_as(active).clone() * alive.to(val.dtype)


def build_critic(*, architecture: str = "agent_v1", hidden: int = 128,
                 return_scale: float = 1.0) -> nn.Module:
    """Versioned critic factory; legacy checkpoints always select agent_v1."""
    if architecture == "agent_v1":
        return CentralizedCritic(hidden=hidden)
    if architecture == "temporal_team_v2":
        return TemporalTeamCritic(hidden=hidden, return_scale=return_scale)
    raise ValueError(f"unsupported critic architecture: {architecture}")
