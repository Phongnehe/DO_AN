"""Suy luận cho ứng dụng minh họa (mục 6.2.3): văn bản + ảnh → kết quả JSON theo từng khía cạnh."""
from pathlib import Path

import numpy as np
import torch

from mabsa.aspects import CODE2NAME, SENTIMENTS
from mabsa.builder import build_dataset, build_tokenizer_from_cfg, get_segmenter, model_grid
from mabsa.config import resolve
from mabsa.data.dataset import Collator
from mabsa.data.preprocess import normalize_text
from mabsa.engine import to_device
from mabsa.models.extractor import BIOAspectExtractor, LexiconAspectExtractor, merge_by_category
from mabsa.models.mabsa import MABSAModel
from mabsa.utils import get_device, get_logger, load_checkpoint

log = get_logger()


class MABSAPredictor:
    def __init__(self, checkpoint="checkpoints/mabsa_best.pt", extractor_checkpoint="checkpoints/extractor_best.pt",
                 device="auto"):
        self.device = get_device(device)
        ckpt = load_checkpoint(resolve(checkpoint))
        self.cfg = ckpt["config"]
        self.tokenizer = build_tokenizer_from_cfg(self.cfg)
        self.model = MABSAModel(self.cfg, vocab_size=len(self.tokenizer))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.segmenter = get_segmenter(self.cfg)
        self.grid = model_grid(self.cfg)
        self.collate = Collator(self.tokenizer, self.cfg["data"]["max_images"], self.grid)
        ext_path = resolve(extractor_checkpoint) if extractor_checkpoint else None
        if ext_path and Path(ext_path).exists():
            e = torch.load(ext_path, map_location="cpu", weights_only=False)
            self.extractor = BIOAspectExtractor(e["config"], vocab_size=len(self.tokenizer))
            self.extractor.load_state_dict(e["state_dict"])
            self.extractor.to(self.device).eval()
            self.extractor_name = "bio"
        else:
            log.warning("Chưa có checkpoint bộ nhận dạng khía cạnh → dùng từ điển từ khóa.")
            self.extractor = LexiconAspectExtractor()
            self.extractor_name = "lexicon"

    def extract_aspects(self, text):
        if self.extractor_name == "bio":
            spans = self.extractor.extract(text, self.tokenizer, self.segmenter, self.cfg["data"]["max_seq_len"])
        else:
            spans = self.extractor.extract(text)
        return merge_by_category(spans)

    @torch.no_grad()
    def predict(self, text: str, images: list, aspects: list = None) -> dict:
        """images: danh sách PIL.Image. aspects (tùy chọn): danh sách {category, term, start, end} đã biết."""
        text = normalize_text(text)
        aspects = aspects or self.extract_aspects(text)
        supported = self.cfg['model'].get('supported_aspects')
        unsupported = []
        if supported is not None:
            unsupported = [{'code': a['category'], 'aspect': CODE2NAME[a['category']],
                            'span': a.get('term', '')} for a in aspects if a['category'] not in supported]
            aspects = [a for a in aspects if a['category'] in supported]
        if not aspects:
            return {'text': text, 'aspects': [], 'unsupported_aspects': unsupported,
                    'extractor': self.extractor_name}
        row = {"id": "input", "text": text, "images": list(images)[: self.cfg["data"]["max_images"]],
               "aspects": [dict(a, sentiment=None) for a in aspects]}
        ds = build_dataset([row], self.cfg, self.tokenizer, with_labels=False)
        batch = to_device(self.collate([ds[0]]), self.device)
        # Trust the submitted files as the source of truth. Some older dataset
        # implementations mark the black placeholder image as present.
        has_images = bool(images) and bool(batch["image_mask"].any().item())
        if not images:
            batch["image_mask"].zero_()
        diagnostics = {"device": str(self.device), "has_images": has_images}
        if (self.model.type == "mabsa" and not has_images
                and self.cfg['model'].get('aspect_representation', 'span') == 'span'
                and not hasattr(self.model, 'text_classifier')):
            # With text-only input, use the learned text path and a zero visual
            # context directly. This avoids masked-image attention and keeps
            # prediction independent of the image backbone.
            H = self.model.text_encoder(batch["input_ids"], batch["attention_mask"])
            pooled = torch.bmm(batch["span_mask"], H) / batch["span_mask"].sum(-1, keepdim=True).clamp_min(1.0)
            aspects_repr = self.model.text_proj(pooled)
            fused = self.model.fusion(aspects_repr, torch.zeros_like(aspects_repr))
            logits = self.model.classifier(self.model.dropout(fused))
            diagnostics.update({
                "text_finite": bool(torch.isfinite(H).all().item()),
                "aspect_finite": bool(torch.isfinite(aspects_repr).all().item()),
                "fusion_finite": bool(torch.isfinite(fused).all().item()),
            })
            attention = None
        else:
            out = self.model(batch)
            logits = out["logits"]
            attention = out.get("attention")
        diagnostics["logits_finite"] = bool(torch.isfinite(logits).all().item())
        probs = logits[0].softmax(-1).cpu().numpy()
        attn = attention[0].detach().float().cpu().numpy() if attention is not None else None
        n_img = int(batch["image_mask"][0].sum().item())
        # Infer the patch grid from the attention tensor itself. This supports
        # ViT/CLIP/ResNet and safely skips heatmaps if an encoder returns no regions.
        regions_per_image = 0
        attention_grid = 0
        max_images = int(self.cfg["data"].get("max_images", 0))
        if attn is not None and attn.ndim == 2 and max_images > 0 and attn.shape[-1] % max_images == 0:
            regions_per_image = attn.shape[-1] // max_images
            attention_grid = int(np.sqrt(regions_per_image))
            if attention_grid * attention_grid != regions_per_image:
                attention_grid = 0
        results = []
        for j, a in enumerate(aspects):
            k = int(probs[j].argmax())
            item = {"aspect": CODE2NAME[a["category"]], "code": a["category"], "span": a.get("term", ""),
                    "sentiment": SENTIMENTS[k], "confidence": float(probs[j, k]),
                    "probs": {s: float(p) for s, p in zip(SENTIMENTS, probs[j])}}
            if (attn is not None and attn.ndim == 2 and j < attn.shape[0]
                    and attention_grid > 0 and n_img > 0
                    and attn.shape[-1] >= n_img * regions_per_image):
                item["attention"] = [
                    attn[j, i * regions_per_image:(i + 1) * regions_per_image]
                    .reshape(attention_grid, attention_grid).tolist()
                    for i in range(min(n_img, max_images))
                ]
            results.append(item)
        response = {"text": text, "aspects": results, "extractor": self.extractor_name}
        if unsupported:
            response['unsupported_aspects'] = unsupported
        if not diagnostics["logits_finite"] or not np.isfinite(probs).all():
            response["diagnostics"] = diagnostics
        return response
