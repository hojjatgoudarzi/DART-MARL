import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.utils import flatten, CrossAttentionBlock


"""
GraphTransformer
    Decentralized POI-selection policy with limited communication.
    Each rover encodes its own state into a personal embedding and
    broadcasts one message per step. It then attends, over several
    layers, to its own message, the messages of the rovers it can
    hear, and all POIs, each tagged with a learned encoding of its
    position relative to the rover (Point-Transformer style: an
    MLP of the offset and its length).
    The keys/values are fixed for the step (one exchange), and only
    the rover's own token is refined. A pointer head scores every
    POI against the final token, plus one idle logit, under a
    masked categorical. Shared by every rover.
"""
class GraphTransformer(nn.Module):
    """
    GraphTransformer.__init__
        Args:
            agent_dim: int rover feature size
            poi_dim: int POI feature size
            hidden: int token width
            msg_dim: int broadcast message size (bandwidth)
            heads: int attention heads
            layers: int cross-attention layers
    """
    def __init__(self, agent_dim, poi_dim, hidden=64, msg_dim=64, heads=4, layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(agent_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.message = nn.Linear(hidden, msg_dim)
        self.receive = nn.Linear(msg_dim, hidden)
        self.poi_encoder = nn.Sequential(nn.Linear(poi_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pos_encoder = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm_kv = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden, heads) for _ in range(layers)])
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.idle = nn.Linear(hidden, 1)

    """
    GraphTransformer.forward
        Args:
            obs: dict of observation tensors (see RoverEnv.observe)
                with leading dims (...)

        Returns:
            torch.distributions.Categorical over (..., A) actions in [0, P],
                P = idle, unselectable POIs masked out
    """
    def forward(self, obs):
        obs, lead = flatten(obs)
        N, A, _ = obs['agent'].shape
        P = obs['poi'].shape[1]

        # personal embeddings, the messages they broadcast, and POI tokens
        e = self.encoder(obs['agent'])
        messages = self.receive(self.message(e))
        tokens = torch.cat([messages, self.poi_encoder(obs['poi'])], dim=1)

        # each rover's keys/values: tokens plus their position relative to it (offset and distance)
        others = torch.cat([obs['pos'], obs['poi_pos']], dim=1)
        rel = others[:, None] - obs['pos'][:, :, None]
        rel = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)
        kv = tokens[:, None] + self.pos_encoder(rel)
        kv_norm = self.norm_kv(kv)

        # a rover hears itself and the rovers in comm range, and sees every POI
        hears = obs['comm'] | torch.eye(A, dtype=torch.bool, device=e.device)
        mask = torch.cat([hears, torch.ones(N, A, P, dtype=torch.bool, device=e.device)], dim=-1)

        x = e
        for block in self.blocks:
            x = block(x, kv_norm, mask)

        # pointer head over POIs, plus idle
        poi_logits = torch.einsum('nah,naph->nap', self.query(x), self.key(kv[:, :, A:])) / math.sqrt(x.shape[-1])
        logits = torch.cat([poi_logits, self.idle(x)], dim=-1)
        logits = logits.masked_fill(~obs['action_mask'][:, None], torch.finfo(logits.dtype).min)
        return Categorical(logits=logits.reshape(*lead, A, P + 1))
