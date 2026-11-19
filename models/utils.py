import torch.nn as nn
import torch.nn.functional as F


"""
flatten
    Merges the leading dimensions (e.g. time and worlds) of every
    observation tensor into one.

    Args:
        obs: dict of tensors sharing leading dimensions

    Returns:
        dict of tensors with one leading dimension N
        tuple of the original leading dimensions
"""
def flatten(obs):
    lead = obs['action_mask'].shape[:-1]
    return {k: v.reshape(-1, *v.shape[len(lead):]) for k, v in obs.items()}, lead


"""
CrossAttentionBlock
    Pre-LN transformer decoder block in which each rover's own
    token (the query) attends to its own set of keys/values (the
    messages it hears and the POIs). The keys/values are not
    updated by the block (and are normalized once by the caller),
    so stacking blocks refines a rover's own token without any
    further communication.
"""
class CrossAttentionBlock(nn.Module):
    """
    CrossAttentionBlock.__init__
        Args:
            dim: int token width
            heads: int attention heads
    """
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    """
    CrossAttentionBlock.forward
        Args:
            x: torch.Tensor of size (N, A, D) one query token per rover
            kv: torch.Tensor of size (N, A, M, D) each rover's normalized keys/values
            mask: torch.Tensor of size (N, A, M) bool, True = attend

        Returns:
            torch.Tensor of size (N, A, D)
    """
    def forward(self, x, kv, mask):
        N, A, M, D = kv.shape
        h, dh = self.heads, D // self.heads

        # every rover has its own key set, so fold rovers into the batch: one query of length 1 each
        q = self.q(self.norm_q(x)).view(N * A, 1, h, dh).transpose(1, 2)
        k, v = self.kv(kv).view(N * A, M, 2, h, dh).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.view(N * A, 1, 1, M))

        x = x + self.out(out.reshape(N, A, D))
        return x + self.ff(self.norm_ff(x))
