"""Mô hình MABSA đề xuất (Sơ đồ 4.1) và các baseline (mục 5.2) trong cùng một lớp.

model.type:
  mabsa         mô hình đề xuất: PhoBERT + ViT/CLIP + Contrastive + Cross-Attention + Gated fusion
                (use_contrastive=false  → "Cross-attention fusion (không contrastive)" / ablation)
                (use_cross_attention=false → ablation "Loại bỏ cross-attention")
  text_only     PhoBERT (chỉ văn bản)
  image_only    ViT/ResNet (chỉ ảnh)
  early_fusion  nối đặc trưng văn bản và ảnh ở đầu vào tầng phân loại
  late_fusion   hai nhánh độc lập, kết hợp ở tầng quyết định
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mabsa.aspects import ASPECT_CODES, SENTIMENTS
from mabsa.models.contrastive import AspectContrastiveLoss
from mabsa.models.encoders import ImageEncoder, TextEncoder
from mabsa.models.fusion import AspectCrossAttention, ConcatFusion, GatedFusion, masked_mean

MODEL_TYPES = ["mabsa", "text_only", "image_only", "early_fusion", "late_fusion"]


class MABSAModel(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int = None):
        super().__init__()
        m = {"max_seq_len": cfg["data"]["max_seq_len"], "image_size": cfg["data"]["image_size"], **cfg["model"]}
        self.cfg = cfg
        self.type = m.get("type", "mabsa")
        assert self.type in MODEL_TYPES, self.type
        self.use_contrastive = self.type == "mabsa" and m.get("use_contrastive", True)
        self.use_cross_attention = self.type == "mabsa" and m.get("use_cross_attention", True)
        self.alpha, self.beta = m.get("alpha", 0.7), m.get("beta", 0.3)
        d = m.get("hidden_dim", 768)
        n_cls = len(SENTIMENTS)
        drop = m.get("dropout", 0.1)

        self.use_text = self.type != "image_only"
        self.use_image = self.type != "text_only"
        if self.use_text:
            self.text_encoder = TextEncoder(m, vocab_size)
            self.text_proj = nn.Sequential(nn.Linear(self.text_encoder.dim, d), nn.LayerNorm(d))
        if self.use_image:
            self.image_encoder = ImageEncoder(m)
            self.grid = self.image_encoder.grid
            self.image_proj = nn.Sequential(nn.Linear(self.image_encoder.dim, d), nn.LayerNorm(d))
        if m.get("freeze_image_encoder", False) and self.use_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(drop)
        if self.type == "mabsa":
            if self.use_contrastive:
                self.contrastive = AspectContrastiveLoss(d, m.get("proj_dim", 256), m.get("tau", 0.07))
            if self.use_cross_attention:
                self.cross_attn = AspectCrossAttention(d, m.get("attn_heads", 8), drop)
            self.fusion = GatedFusion(d) if m.get("fusion", "gated") == "gated" else ConcatFusion(d)
            self.classifier = nn.Linear(d, n_cls)
        elif self.type == "text_only":
            self.classifier = nn.Linear(d, n_cls)
        elif self.type == "image_only":
            self.cat_emb = nn.Embedding(len(ASPECT_CODES), d)
            self.classifier = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_cls))
        elif self.type == "early_fusion":
            self.classifier = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_cls))
        elif self.type == "late_fusion":
            self.cat_emb = nn.Embedding(len(ASPECT_CODES), d)
            self.text_head = nn.Linear(d, n_cls)
            self.image_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_cls))
            self.mix = nn.Parameter(torch.zeros(1))
        self.class_weight = None
        self.text_class_weight = None
        self.aspect_representation = m.get('aspect_representation', 'span')
        if self.use_text and self.aspect_representation == 'category_attention':
            self.aspect_query = nn.Embedding(len(ASPECT_CODES), d)
            self.text_attn = nn.MultiheadAttention(d, m.get('attn_heads', 8), dropout=drop, batch_first=True)
            self.aspect_norm = nn.LayerNorm(d)
        if self.type == 'mabsa' and m.get('text_auxiliary_head', False):
            self.text_classifier = nn.Linear(d, n_cls)
        if self.use_text and m.get('freeze_text_encoder', False):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.use_text and self.cfg['model'].get('freeze_text_encoder', False):
            self.text_encoder.eval()
        if self.use_image and self.cfg['model'].get('freeze_image_encoder', False):
            self.image_encoder.eval()
        return self

    # ------------------------------------------------------------ encoders
    def encode_aspects(self, batch):
        """Biểu diễn khía cạnh a_j = MeanPooling{h_i | i ∈ vị trí khía cạnh j} (mục 4.4.1)."""
        H = batch.get('text_features')
        if H is None:
            H = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if self.aspect_representation == 'category_attention':
            tokens = self.text_proj(H)
            queries = self.aspect_query(batch['categories'])
            context, _ = self.text_attn(queries, tokens, tokens,
                key_padding_mask=~batch['attention_mask'].bool(), need_weights=True)
            return self.aspect_norm(context + queries)
        span = batch["span_mask"]                                               # B×A×N
        pooled = torch.bmm(span, H) / span.sum(-1, keepdim=True).clamp_min(1.0)
        return self.text_proj(pooled)                                           # B×A×d

    def encode_regions(self, batch):
        """Biểu diễn vùng ảnh V = {v_1..v_k} của mọi ảnh trong mẫu (mục 4.4.2)."""
        imgs, img_mask = batch["images"], batch["image_mask"]                  # B×K×3×H×W, B×K
        B, K = img_mask.shape
        cached = batch.get('image_features')
        if cached is not None:
            P, dv = cached.shape[-2:]
            V = self.image_proj(cached.reshape(B, K * P, dv))
            region_mask = img_mask.unsqueeze(-1).expand(B, K, P).reshape(B, K * P)
            return V, region_mask
        flat = imgs.flatten(0, 1)
        sel = img_mask.flatten()
        if not sel.any():
            # Avoid sending an empty batch through image backbones. The model's
            # fusion path will use text features when every image is absent.
            P = int(self.cfg['model'].get('image_pool_grid', self.image_encoder.grid)) ** 2
            dv = self.image_encoder.dim
            V = flat.new_zeros(B, K * P, dv)
            V = self.image_proj(V)
            region_mask = img_mask.unsqueeze(-1).expand(B, K, P).reshape(B, K * P)
            return V, region_mask
        feats_valid = self.image_encoder(flat[sel])                             # M×P×d_v
        pool_grid = self.cfg['model'].get('image_pool_grid')
        if pool_grid and int(pool_grid) != self.image_encoder.grid:
            old_grid = self.image_encoder.grid
            fmap = feats_valid.transpose(1, 2).reshape(-1, feats_valid.shape[-1], old_grid, old_grid)
            feats_valid = F.adaptive_avg_pool2d(fmap, (int(pool_grid), int(pool_grid))).flatten(2).transpose(1, 2)
        P, dv = feats_valid.shape[1], feats_valid.shape[2]
        feats = feats_valid.new_zeros(B * K, P, dv)
        feats[sel] = feats_valid
        V = self.image_proj(feats.view(B, K * P, dv))                           # B×R×d
        region_mask = img_mask.unsqueeze(-1).expand(B, K, P).reshape(B, K * P)
        return V, region_mask

    # ------------------------------------------------------------ forward
    def forward(self, batch):
        out = {"attention": None}
        a = self.encode_aspects(batch) if self.use_text else None
        if self.use_image:
            V, region_mask = self.encode_regions(batch)
            v_mean = masked_mean(V, region_mask).unsqueeze(1)                   # B×1×d

        if self.type == "mabsa":
            if self.use_contrastive and self.training:
                out["loss_con"] = self.contrastive(a, batch["aspect_mask"], V, region_mask,
                                                   batch["region_pos"], batch["has_region"])
            if self.use_cross_attention:
                # MultiheadAttention returns NaNs when a sample has every image
                # region masked. Keep the text prediction usable for missing or
                # unreadable images by temporarily unmasking a zero-valued region,
                # then discard that sample's image context and attention map.
                has_regions = region_mask.any(dim=-1)
                if V.shape[1] == 0:
                    c = torch.zeros_like(a)
                else:
                    safe_regions = V
                    safe_mask = region_mask
                    missing = ~has_regions
                    if missing.any():
                        safe_regions = V.clone()
                        safe_mask = region_mask.clone()
                        safe_regions[missing, 0] = 0
                        safe_mask[missing, 0] = True
                    c, w = self.cross_attn(a, safe_regions, safe_mask)
                    c = c * has_regions[:, None, None].to(c.dtype)
                    out["attention"] = w * has_regions[:, None, None].to(w.dtype)  # B×A×R
                if V.shape[1] == 0:
                    out["attention"] = None
            else:
                c = v_mean.expand_as(a)
            logits = self.classifier(self.dropout(self.fusion(a, c)))
        elif self.type == "text_only":
            logits = self.classifier(self.dropout(a))
        elif self.type == "image_only":
            cat = self.cat_emb(batch["categories"])
            logits = self.classifier(torch.cat([v_mean.expand_as(cat), cat], -1))
        elif self.type == "early_fusion":
            logits = self.classifier(torch.cat([a, v_mean.expand_as(a)], -1))
        else:  # late_fusion
            cat = self.cat_emb(batch["categories"])
            w = torch.sigmoid(self.mix)
            logits = w * self.text_head(self.dropout(a)) + (1 - w) * self.image_head(
                torch.cat([v_mean.expand_as(cat), cat], -1))
        out["logits"] = logits                                                  # B×A×3
        if hasattr(self, 'text_classifier'):
            text_logits = self.text_classifier(self.dropout(a))
            out['text_logits'] = text_logits
            has_images = batch['image_mask'].any(-1)[:, None, None]
            out['logits'] = torch.where(has_images, logits, text_logits)
        return out

    def compute_loss(self, out, batch):
        """L_total = α·L_cls + β·L_contrastive (mục 4.8, Thuật toán 5.2)."""
        labels = batch["labels"]
        if hasattr(self, 'text_classifier'):
            labels = torch.where(batch['image_mask'].any(-1)[:, None], labels, batch['text_labels'])
        weight = self.class_weight.to(out["logits"].device) if self.class_weight is not None else None
        loss_cls = F.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]), labels.reshape(-1),
                                   weight=weight, ignore_index=-100) if (labels != -100).any() else out['logits'].sum() * 0
        if 'text_logits' in out:
            tw = self.text_class_weight.to(out['logits'].device) if self.text_class_weight is not None else weight
            text_labels = batch['text_labels']
            text_loss = F.cross_entropy(out['text_logits'].reshape(-1, out['text_logits'].shape[-1]),
                                       text_labels.reshape(-1), weight=tw, ignore_index=-100) \
                if (text_labels != -100).any() else out['text_logits'].sum() * 0
            loss_cls = loss_cls + self.cfg['model'].get('text_aux_loss_weight', 1.0) * text_loss
        if "loss_con" in out:
            total = self.alpha * loss_cls + self.beta * out["loss_con"]
            return total, {"loss_cls": loss_cls.item(), "loss_con": out["loss_con"].item()}
        return loss_cls, {"loss_cls": loss_cls.item()}
