"""Bounded stochastic independent execution with *correct* tanh-Gaussian log-probs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from dart_marl.algorithms.mappo.role_guidance import role_navigation_prior


class SharedActor(nn.Module):
    """One policy for every active rover. Inputs: local sensor + assigned target only."""

    def __init__(self, obs_dim: int, hidden: int = 128, action_dim: int = 2,
                 role_guidance_gain: float = 0.0):
        super().__init__()
        if role_guidance_gain < 0:
            raise ValueError("role_guidance_gain must be nonnegative")
        self.role_guidance_gain = float(role_guidance_gain)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.mean(self.net(observation))
        if self.role_guidance_gain:
            mean = mean + role_navigation_prior(observation, gain=self.role_guidance_gain)
        log_std = self.log_std.clamp(-5.0, 1.0)
        return Normal(mean, log_std.exp().expand_as(mean))

    @staticmethod
    def _log_prob(dist: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        # Stable log(1 - tanh(x)^2); ensures gradients remain finite for large |x|.
        log_det = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return (dist.log_prob(pre_tanh) - log_det).sum(dim=-1)

    def act(self, obs: torch.Tensor, *, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.rsample()
        return torch.tanh(latent), latent, self._log_prob(dist, latent)

    def evaluate(self, obs: torch.Tensor, latent: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        logp = self._log_prob(dist, latent)
        # Analytic entropy of the *base* Gaussian: a documented exploration proxy,
        # NOT the exact entropy of the squashed distribution.
        entropy_proxy = dist.entropy().sum(dim=-1)
        return logp, entropy_proxy
