"""Small residual outcome model: expected H-step raw return minus analytic forecast."""
from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
