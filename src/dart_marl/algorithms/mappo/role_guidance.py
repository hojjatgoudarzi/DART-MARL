"""Optional role-conditioned navigation inductive bias.

This is an explicitly labeled MAPPO+role-prior variant, never called vanilla MAPPO.
It modifies the actor's *distribution mean*, so actions and PPO log probabilities
remain on-policy. It does not secretly override actions after sampling.
"""

from __future__ import annotations

import torch


def role_navigation_prior(observations: torch.Tensor, *, gain: float) -> torch.Tensor:
    """Mean offset [...,2] from the 15 research-only target features.

    Requires role-conditioned research observations with target direction in
    indices 5:7 of the trailing 15 features, validity at indices 11 and 14.
    No environment state or future information is supplied to the actor.
    The offset decays to zero near the target to permit holding position.
    """
    if gain < 0:
        raise ValueError("role guidance gain must be nonnegative")
    if observations.shape[-1] < 15:
        raise ValueError("role-navigation prior requires research-mode observations")
    if gain == 0:
        return observations[..., -15:-13] * 0.0
    features = observations[..., -15:]
    delta = features[..., 5:7]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    # Approximately full guidance beyond 10% of map width; near-center fade.
    strength = (distance / 0.10).clamp(max=1.0)
    direction = delta / distance.clamp_min(1e-8)
    valid = features[..., 11:12].clamp(0.0, 1.0) * features[..., 14:15].clamp(0.0, 1.0)
    return gain * direction * strength * valid
