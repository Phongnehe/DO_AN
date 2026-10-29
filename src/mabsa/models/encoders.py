"""Bộ mã hóa văn bản (mục 4.3.1) và bộ mã hóa hình ảnh (mục 4.3.2).

Văn bản: PhoBERT (mặc định), XLM-R, ViDeBERTa — hoặc "tiny" (BERT nhỏ khởi tạo ngẫu nhiên, để chạy thử).
Ảnh:     ViT (mặc định), CLIP visual encoder, ResNet-50 — hoặc "tiny".
Mỗi bộ mã hóa ảnh trả về biểu diễn của TỪNG vùng ảnh (patch), không gộp thành vector toàn cục.
"""
from pathlib import Path

import torch
import torch.nn as nn

TEXT_MODELS = {
    "phobert": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large",
    "xlmr": "xlm-roberta-base",
    "videberta": "Fsoft-AIC/videberta-base",
}
IMAGE_MODELS = {
    "vit": "google/vit-base-patch16-224-in21k",
    "clip": "openai/clip-vit-base-patch16",
    "resnet": "resnet50",
}


# ------------------------------------------------------------------ text
def build_tokenizer(cfg):
    name = cfg["text_encoder"]
    if name == "tiny":
        from transformers import BertTokenizer
        vocab = Path(cfg.get("tiny_vocab", "data/toy/vocab.txt"))
        from mabsa.config import resolve
        return BertTokenizer(str(resolve(vocab)), do_lower_case=True, strip_accents=False)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEXT_MODELS.get(name, name))


class TextEncoder(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        name = cfg["text_encoder"]
        if name == "tiny":
            from transformers import BertConfig, BertModel
            conf = BertConfig(vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=128, max_position_embeddings=cfg["max_seq_len"] + 8)
            self.model = BertModel(conf)
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(TEXT_MODELS.get(name, name),
                attn_implementation=cfg.get('attn_implementation', 'eager'))
        self.dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # B×N×d


# ------------------------------------------------------------------ image
class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = cfg["image_encoder"]
        size = cfg["image_size"]
        if self.name == "tiny":
            from transformers import ViTConfig, ViTModel
            conf = ViTConfig(image_size=size, patch_size=cfg.get("tiny_patch", 16), hidden_size=64,
                             num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
            self.model = ViTModel(conf, add_pooling_layer=False)
            self.dim, self.grid = 64, size // conf.patch_size
        elif self.name == "vit":
            from transformers import ViTModel
            self.model = ViTModel.from_pretrained(IMAGE_MODELS["vit"], add_pooling_layer=False,
                attn_implementation=cfg.get('attn_implementation', 'eager'))
            self.dim, self.grid = self.model.config.hidden_size, size // self.model.config.patch_size
        elif self.name == "clip":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(IMAGE_MODELS["clip"],
                attn_implementation=cfg.get('attn_implementation', 'eager'))
            self.dim, self.grid = self.model.config.hidden_size, size // self.model.config.patch_size
        elif self.name == "resnet":
            import torchvision
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            self.model = nn.Sequential(*list(net.children())[:-2])   # giữ bản đồ đặc trưng 7×7
            self.dim, self.grid = 2048, size // 32
        else:
            raise ValueError(f"image_encoder không hỗ trợ: {self.name}")

    def forward(self, pixel_values):
        """pixel_values: M×3×H×W → M×(grid·grid)×d (bỏ token [CLS])."""
        if self.name == "resnet":
            fmap = self.model(pixel_values)                      # M×2048×7×7
            return fmap.flatten(2).transpose(1, 2)
        out = self.model(pixel_values=pixel_values).last_hidden_state
        return out[:, 1:, :]


def grid_size(cfg) -> int:
    enc, size = cfg["image_encoder"], cfg["image_size"]
    if enc == "resnet":
        return size // 32
    if enc == "tiny":
        return size // cfg.get("tiny_patch", 16)
    return size // 16
