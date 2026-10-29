"""Dataset và collate cho mô hình MABSA (mục 4.2).

Định dạng một mẫu (một dòng JSONL, xem data/README.md):
{
  "id": "TA_000123", "source": "tripadvisor", "service_type": "restaurant",
  "text": "Món ăn ở đây rất ngon nhưng nhân viên phục vụ khá chậm",
  "images": ["data/raw/images/TA_000123_1.jpg"],
  "aspects": [
    {"category": "FOOD", "term": "Món ăn", "start": 0, "end": 6, "sentiment": "Tích cực",
     "regions": [{"image": 0, "bbox": [0.20, 0.15, 0.80, 0.90]}]},
    {"category": "SERVICE", "term": "nhân viên phục vụ", "start": 29, "end": 46, "sentiment": "Tiêu cực",
     "regions": []}
  ]
}
- start/end: vị trí ký tự của cụm từ khía cạnh trong `text` (khía cạnh ẩn: start = end = -1).
- bbox: tọa độ chuẩn hóa [x1, y1, x2, y2] trong [0, 1] (giá trị > 1 được hiểu là phần trăm).
"""
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from mabsa.aspects import ASPECT2ID, SENT2ID, normalize_aspect, normalize_sentiment
from mabsa.config import resolve
from mabsa.data.preprocess import image_to_tensor, load_image, normalize_text
from mabsa.utils import get_logger

log = get_logger()


def special_ids(tokenizer):
    cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return cls, sep, pad


def locate_aspect(text: str, asp: dict):
    """Kiểm tra/khôi phục vị trí ký tự của cụm từ khía cạnh; trả về (start, end) hoặc (-1, -1)."""
    s, e = int(asp.get("start", -1)), int(asp.get("end", -1))
    term = (asp.get("term") or "").strip()
    if 0 <= s < e <= len(text) and (not term or text[s:e].lower() == term.lower()):
        return s, e
    if term:
        pos = text.lower().find(term.lower())
        if pos >= 0:
            return pos, pos + len(term)
    return -1, -1


def tokenize_with_spans(text, spans, tokenizer, segmenter, max_len):
    """Tách từ + mã hóa theo từng đoạn để biết chính xác token nào thuộc cụm từ khía cạnh.

    Văn bản được cắt tại mọi ranh giới start/end, mỗi đoạn được tách từ và mã hóa riêng rồi nối lại.
    Trả về input_ids (có token đầu/cuối) và danh sách (tok_start, tok_end) cho từng span
    (span ẩn hoặc bị cắt cụt → (0, 1), tức token đầu câu <s>/[CLS]).
    """
    cls, sep, _ = special_ids(tokenizer)
    cuts = sorted({0, len(text)} | {p for s, e in spans if s >= 0 for p in (s, e)})
    seg_tok = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = text[a:b].strip()
        ids = tokenizer(segmenter(piece), add_special_tokens=False)["input_ids"] if piece else []
        seg_tok.append((a, b, ids))
    ids, char2tok = [cls], []
    for a, b, t in seg_tok:
        char2tok.append((a, b, len(ids), len(ids) + len(t)))
        ids.extend(t)
    limit = max_len - 1
    ids = ids[:limit] + [sep]
    tok_spans = []
    for s, e in spans:
        if s < 0:
            tok_spans.append((0, 1))
            continue
        covered = [(ts, te) for a, b, ts, te in char2tok if a >= s and b <= e and te > ts]
        if not covered:
            tok_spans.append((0, 1))
            continue
        ts, te = covered[0][0], covered[-1][1]
        te = min(te, limit)
        tok_spans.append((ts, te) if te > ts else (0, 1))
    return ids, tok_spans


def bbox_to_cells(bbox, grid: int, min_overlap: float = 0.3):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if max(x1, y1, x2, y2) > 1.0:
        x1, y1, x2, y2 = x1 / 100, y1 / 100, x2 / 100, y2 / 100
    x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
    cells, step = [], 1.0 / grid
    for r in range(grid):
        for c in range(grid):
            cx1, cy1 = c * step, r * step
            ow = max(0.0, min(x2, cx1 + step) - max(x1, cx1))
            oh = max(0.0, min(y2, cy1 + step) - max(y1, cy1))
            if ow * oh >= min_overlap * step * step:
                cells.append(r * grid + c)
    if not cells:  # bbox nhỏ hơn một ô: lấy ô chứa tâm bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cells.append(min(int(cy * grid), grid - 1) * grid + min(int(cx * grid), grid - 1))
    return cells


class MABSADataset(Dataset):
    def __init__(self, rows, tokenizer, segmenter, cfg, image_encoder: str, grid: int, with_labels: bool = True):
        self.rows = rows
        self.tok = tokenizer
        self.seg = segmenter
        self.max_len = cfg["max_seq_len"]
        self.max_images = cfg["max_images"]
        self.max_aspects = cfg.get("max_aspects", 12)
        self.image_size = cfg["image_size"]
        self.image_encoder = image_encoder
        self.grid = grid
        self.with_labels = with_labels
        self.aspect_representation = cfg.get('aspect_representation', 'span')
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _encode_text(self, i):
        if i not in self._cache:
            row = self.rows[i]
            raw = row["text"]
            text = normalize_text(raw)
            aspects = row.get("aspects", [])[: self.max_aspects]
            if self.aspect_representation == 'category_attention':
                # Category-conditioned attention uses the whole review in both
                # training and inference, independent of lexicon span boundaries.
                spans = [(-1, -1)] * len(aspects)
            elif text == raw:
                spans = [locate_aspect(text, a) for a in aspects]
            else:  # văn bản thay đổi sau chuẩn hóa → tìm lại cụm từ theo nội dung
                spans = [locate_aspect(text, {**a, "start": -1, "end": -1}) for a in aspects]
            self._cache[i] = tokenize_with_spans(text, spans, self.tok, self.seg, self.max_len)
        return self._cache[i]

    def __getitem__(self, i):
        row = self.rows[i]
        ids, tok_spans = self._encode_text(i)
        images, img_mask = [], []
        for p in row.get("images", [])[: self.max_images]:
            try:
                img = p if not isinstance(p, str) else load_image(resolve(p))
                images.append(image_to_tensor(img, self.image_size, self.image_encoder))
                img_mask.append(True)
            except Exception as e:
                log.warning("Bỏ qua ảnh lỗi %s (%s)", p, e)
        if not images:  # giữ tensor placeholder, nhưng đánh dấu ảnh vắng mặt
            images.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))
            img_mask.append(False)
        n_regions = self.max_images * self.grid * self.grid
        aspects = row.get("aspects", [])[: self.max_aspects]
        cats, labels, text_labels, pos, has_pos = [], [], [], [], []
        for a in aspects:
            cats.append(ASPECT2ID[normalize_aspect(a["category"])])
            if self.with_labels and a.get("sentiment") is not None:
                labels.append(SENT2ID[normalize_sentiment(a["sentiment"])])
            else:
                labels.append(-100)
            text_sentiment = a.get('text_sentiment', a.get('sentiment'))
            text_labels.append(SENT2ID[normalize_sentiment(text_sentiment)]
                               if self.with_labels and text_sentiment is not None else -100)
            m = np.zeros(n_regions, dtype=bool)
            for reg in a.get("regions", []) or []:
                k = int(reg.get("image", 0))
                if k >= len(images) or "bbox" not in reg:
                    continue
                for cell in bbox_to_cells(reg["bbox"], self.grid):
                    m[k * self.grid * self.grid + cell] = True
            pos.append(m)
            has_pos.append(bool(m.any()))
        return {
            "id": row.get("id", str(i)), "input_ids": ids, "aspect_spans": tok_spans,
            "images": np.stack(images), "image_mask": img_mask,
            "categories": cats, "labels": labels, "text_labels": text_labels,
            "region_pos": pos, "has_region": has_pos,
        }


class Collator:
    def __init__(self, tokenizer, max_images: int, grid: int):
        _, _, self.pad = special_ids(tokenizer)
        self.K = max_images
        self.G = grid

    def __call__(self, batch):
        B = len(batch)
        L = max(len(b["input_ids"]) for b in batch)
        A = max(1, max(len(b["categories"]) for b in batch))
        C, H, W = batch[0]["images"].shape[1:]
        R = self.K * self.G * self.G
        input_ids = torch.full((B, L), self.pad, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        images = torch.zeros((B, self.K, C, H, W))
        image_mask = torch.zeros((B, self.K), dtype=torch.bool)
        span_mask = torch.zeros((B, A, L))
        aspect_mask = torch.zeros((B, A), dtype=torch.bool)
        cats = torch.zeros((B, A), dtype=torch.long)
        labels = torch.full((B, A), -100, dtype=torch.long)
        text_labels = torch.full((B, A), -100, dtype=torch.long)
        region_pos = torch.zeros((B, A, R), dtype=torch.bool)
        has_region = torch.zeros((B, A), dtype=torch.bool)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"])
            attn[i, :n] = 1
            k = len(b["images"])
            images[i, :k] = torch.from_numpy(b["images"])
            image_mask[i, :k] = torch.tensor(b["image_mask"])
            for j, (s, e) in enumerate(b["aspect_spans"]):
                span_mask[i, j, s:e] = 1.0
                aspect_mask[i, j] = True
                cats[i, j] = b["categories"][j]
                labels[i, j] = b["labels"][j]
                text_labels[i, j] = b.get('text_labels', b['labels'])[j]
                region_pos[i, j] = torch.from_numpy(b["region_pos"][j])
                has_region[i, j] = b["has_region"][j]
        return {
            "ids": [b["id"] for b in batch], "input_ids": input_ids, "attention_mask": attn,
            "images": images, "image_mask": image_mask, "span_mask": span_mask,
            "aspect_mask": aspect_mask, "categories": cats, "labels": labels, "text_labels": text_labels,
            "region_pos": region_pos, "has_region": has_region,
        }


def class_weights(rows, num_classes: int = 3):
    """Trọng số nghịch với tần suất lớp cho weighted cross-entropy (mục 4.9)."""
    c = Counter(SENT2ID[normalize_sentiment(a["sentiment"])] for r in rows for a in r.get("aspects", []))
    total = sum(c.values())
    return torch.tensor([total / (num_classes * max(c.get(k, 0), 1)) for k in range(num_classes)], dtype=torch.float)


def oversampling_sampler(rows):
    """Oversampling: mẫu chứa lớp thiểu số được lấy với xác suất cao hơn (mục 4.9)."""
    w = class_weights(rows).numpy()
    weights = []
    for r in rows:
        ids = [SENT2ID[normalize_sentiment(a["sentiment"])] for a in r.get("aspects", [])]
        weights.append(float(max(w[i] for i in ids)) if ids else 1.0)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
