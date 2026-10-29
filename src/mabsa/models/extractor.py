"""Nhận dạng cụm từ khía cạnh khi suy luận (mục 4.4.1, 6.2.3).

- BIOAspectExtractor: gán nhãn chuỗi B-/I-<KHÍA CẠNH>/O trên từ (PhoBERT token classification),
  huấn luyện bằng scripts/train_extractor.py từ chính dữ liệu đã gán nhãn.
- LexiconAspectExtractor: dựa trên từ khóa (ASPECT_LEXICON) – dùng cho gán nhãn bán tự động
  và làm phương án dự phòng khi chưa có checkpoint của bộ nhận dạng.
"""
import re

import torch
import torch.nn as nn

from mabsa.aspects import ASPECT_CODES, ASPECT_LEXICON, normalize_aspect

BIO_LABELS = ["O"] + [f"{p}-{c}" for c in ASPECT_CODES for p in ("B", "I")]
BIO2ID = {l: i for i, l in enumerate(BIO_LABELS)}


def align_words(text: str, segmented: str):
    """Gióng các từ sau khi tách từ (vd. 'nhân_viên') về vị trí ký tự trong văn bản gốc."""
    words, pos, low = [], 0, text.lower()
    for w in segmented.split():
        surface = w.replace("_", " ")
        s = low.find(surface.lower(), pos)
        if s < 0:
            continue
        words.append((w, s, s + len(surface)))
        pos = s + len(surface)
    return words


def word_labels(words, aspects):
    labels = ["O"] * len(words)
    for a in aspects:
        s, e = int(a.get("start", -1)), int(a.get("end", -1))
        if s < 0:
            continue
        code = normalize_aspect(a["category"])
        first = True
        for k, (_, ws, we) in enumerate(words):
            if ws < e and we > s:
                labels[k] = ("B-" if first else "I-") + code
                first = False
    return labels


def encode_words(words, tokenizer, max_len):
    from mabsa.data.dataset import special_ids
    cls, sep, _ = special_ids(tokenizer)
    ids, first_idx = [cls], []
    for w, _, _ in words:
        t = tokenizer(w, add_special_tokens=False)["input_ids"] or [tokenizer.unk_token_id]
        if len(ids) + len(t) >= max_len - 1:
            break
        first_idx.append(len(ids))
        ids.extend(t)
    ids.append(sep)
    return ids, first_idx


class BIOAspectExtractor(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        name = cfg["model"]["text_encoder"]
        if name == "tiny":
            from transformers import BertConfig, BertForTokenClassification
            conf = BertConfig(vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=128, num_labels=len(BIO_LABELS),
                              max_position_embeddings=cfg["data"]["max_seq_len"] + 8)
            self.model = BertForTokenClassification(conf)
        else:
            from transformers import AutoModelForTokenClassification
            from mabsa.models.encoders import TEXT_MODELS
            self.model = AutoModelForTokenClassification.from_pretrained(
                TEXT_MODELS.get(name, name), num_labels=len(BIO_LABELS))

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def extract(self, text, tokenizer, segmenter, max_len=256):
        self.eval()
        words = align_words(text, segmenter(text))
        if not words:
            return []
        ids, first = encode_words(words, tokenizer, max_len)
        dev = next(self.parameters()).device
        x = torch.tensor([ids], device=dev)
        logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
        tags = [BIO_LABELS[int(logits[i].argmax())] for i in first]
        return decode_spans(text, words[: len(tags)], tags)


def decode_spans(text, words, tags):
    spans, cur = [], None
    for (w, s, e), tag in zip(words, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["category"] != tag[2:])):
            if cur:
                spans.append(cur)
            cur = {"category": tag[2:], "start": s, "end": e}
        elif tag.startswith("I-") and cur:
            cur["end"] = e
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    for sp in spans:
        sp["term"] = text[sp["start"]:sp["end"]]
    return spans


class LexiconAspectExtractor:
    def __init__(self, lexicon=None):
        items = [(term, code) for code, terms in (lexicon or ASPECT_LEXICON).items() for term in terms]
        items.sort(key=lambda x: -len(x[0]))  # ưu tiên cụm dài
        self.items = items

    def extract(self, text, *_, **__):
        low, taken, spans = text.lower(), [], []
        for term, code in self.items:
            for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", low):
                s, e = m.span()
                if any(s < te and e > ts for ts, te in taken):
                    continue
                taken.append((s, e))
                spans.append({"category": code, "term": text[s:e], "start": s, "end": e})
        return sorted(spans, key=lambda x: x["start"])


def merge_by_category(spans):
    """Mỗi khía cạnh giữ lần xuất hiện đầu tiên; nếu không tìm thấy khía cạnh nào → 'Tổng thể' (ẩn)."""
    seen, out = set(), []
    for sp in spans:
        if sp["category"] not in seen:
            seen.add(sp["category"])
            out.append(sp)
    if not out:
        out.append({"category": "GENERAL", "term": "", "start": -1, "end": -1})
    return out
