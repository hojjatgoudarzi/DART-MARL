import torch
import torch.nn as nn

from models.utils import flatten


"""
ValueNetwork
    Centralized value function, used only in training: a
    transformer encoder over every living rover and every POI
    (absolute positions, no communication limit), with one value
    per rover slot read from that rover's output token (each
    rover's expected return of its own reward stream: the team
    reward, or its difference reward).
"""
class ValueNetwork(nn.Module):
    """
    ValueNetwork.__init__
        Args:
            agent_dim: int rover feature size (its absolute position is appended)
            poi_dim: int POI feature size
            hidden: int token width
            heads: int attention heads
            layers: int transformer layers
    """
    def __init__(self, agent_dim, poi_dim, hidden=64, heads=4, layers=2):
        super().__init__()
        self.rover_encoder = nn.Sequential(nn.Linear(agent_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.poi_encoder = nn.Sequential(nn.Linear(poi_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    """
    ValueNetwork.forward
        Args:
            obs: dict of observation tensors (see RoverEnv.observe)
                with leading dims (...)

        Returns:
            torch.Tensor of size (..., A) values of every rover slot
    """
    def forward(self, obs):
        obs, lead = flatten(obs)
        A = obs['agent'].shape[1]
        rovers = self.rover_encoder(torch.cat([obs['agent'], obs['pos']], dim=-1))
        pois = self.poi_encoder(torch.cat([obs['poi'], obs['poi_pos']], dim=-1))
        tokens = torch.cat([rovers, pois], dim=1)

        # failed and empty rover slots are not attended to (their own outputs still exist); POIs always are
        pad = torch.cat([~obs['alive'], torch.zeros_like(obs['poi'][..., 0], dtype=torch.bool)], dim=1)
        h = self.transformer(tokens, src_key_padding_mask=pad)
        return self.head(h[:, :A]).reshape(*lead, A)
