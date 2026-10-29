"""Mô-đun Contrastive Learning cấp độ khía cạnh (mục 4.5, Thuật toán 5.1).

Với mỗi khía cạnh j của mẫu i có vùng ảnh được gán nhãn:
    z = g_t(a_j^i)/‖·‖ ,  u = g_v(v)/‖·‖  với mọi vùng ảnh v trong batch
    p = Σ_{u∈P(i,j)} exp(z·u/τ) / Σ_{u∈U} exp(z·u/τ)
    ℓ = −log p
Mẫu âm gồm các vùng khác trong cùng ảnh (hard negative) và vùng ảnh của mẫu khác trong batch (random negative).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def projection_head(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


class AspectContrastiveLoss(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 256, tau: float = 0.07):
        super().__init__()
        self.g_t = projection_head(dim, proj_dim)
        self.g_v = projection_head(dim, proj_dim)
        self.tau = tau

    def forward(self, aspects, aspect_mask, regions, region_mask, region_pos, has_region):
        """
        aspects:     B×A×d   biểu diễn khía cạnh
        aspect_mask: B×A     khía cạnh hợp lệ
        regions:     B×R×d   biểu diễn vùng ảnh (R = số ảnh tối đa × số patch)
        region_mask: B×R     vùng ảnh hợp lệ
        region_pos:  B×A×R   vùng ảnh dương P(i, j)
        has_region:  B×A     khía cạnh có ít nhất một vùng ảnh dương
        """
        B, A, _ = aspects.shape
        R = regions.shape[1]
        valid = aspect_mask & has_region
        if not valid.any():
            return aspects.sum() * 0.0
        idx_b, idx_a = valid.nonzero(as_tuple=True)                 # Q khía cạnh
        z = F.normalize(self.g_t(aspects[idx_b, idx_a]), dim=-1)    # Q×p        (dòng 1)
        u_all = F.normalize(self.g_v(regions), dim=-1)              # B×R×p
        flat_mask = region_mask.reshape(-1)
        U = u_all.reshape(B * R, -1)[flat_mask]                     # M×p        (dòng 2)
        sim = z @ U.t() / self.tau                                  # Q×M        (dòng 6)
        pos = torch.zeros(len(idx_b), B, R, dtype=torch.bool, device=aspects.device)
        pos[torch.arange(len(idx_b), device=aspects.device), idx_b] = region_pos[idx_b, idx_a] & region_mask[idx_b]
        pos = pos.reshape(len(idx_b), B * R)[:, flat_mask]          # Q×M
        keep = pos.any(dim=-1)                                      # vùng dương có thể bị che (ảnh lỗi)
        if not keep.any():
            return aspects.sum() * 0.0
        sim, pos = sim[keep], pos[keep]
        log_prob = sim.log_softmax(dim=-1)
        log_p = torch.logsumexp(log_prob.masked_fill(~pos, float("-inf")), dim=-1)   # (dòng 7)
        return -log_p.mean()                                                          # (dòng 8, 10)
