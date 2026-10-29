"""Tầng kết hợp (mục 4.6): Cross-Attention (bước 1) và Gated fusion / Concatenation (bước 2)."""
import torch
import torch.nn as nn


class AspectCrossAttention(nn.Module):
    """Khía cạnh a_j là Query, tập vùng ảnh V là Key/Value: c_j = Σ_k α_jk v_k."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, aspects, regions, region_mask):
        ctx, weights = self.attn(aspects, regions, regions, key_padding_mask=~region_mask,
                                 need_weights=True, average_attn_weights=True)
        return self.norm(ctx), weights            # B×A×d, B×A×R


class GatedFusion(nn.Module):
    """g = σ(W_g[a_j; c_j] + b_g);  m_j = g ⊙ a_j + (1 − g) ⊙ c_j."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, a, c):
        g = torch.sigmoid(self.gate(torch.cat([a, c], dim=-1)))
        return g * a + (1 - g) * c


class ConcatFusion(nn.Module):
    """Phương án nối vector: m_j = W[a_j; c_j]."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, a, c):
        return self.proj(torch.cat([a, c], dim=-1))


def masked_mean(x, mask, dim: int = 1):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1.0)
