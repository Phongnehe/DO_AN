"""Huấn luyện bộ nhận dạng cụm từ khía cạnh (gán nhãn chuỗi BIO) dùng khi suy luận (mục 4.4.1, 6.2.3).

    python scripts/train_extractor.py --config configs/default.yaml
Checkpoint: checkpoints/extractor_best.pt (ứng dụng tự động dùng nếu tồn tại).
"""
import _bootstrap  # noqa: F401
import argparse
import random

import torch

from mabsa.builder import build_tokenizer_from_cfg, get_segmenter
from mabsa.config import load_config, resolve
from mabsa.data.dataset import special_ids
from mabsa.data.preprocess import normalize_text
from mabsa.models.extractor import BIO2ID, BIOAspectExtractor, align_words, decode_spans, encode_words, word_labels
from mabsa.utils import get_device, get_logger, read_jsonl, set_seed

log = get_logger()


def encode_rows(rows, tok, seg, max_len):
    data = []
    for r in rows:
        text = normalize_text(r["text"])
        words = align_words(text, seg(text))
        if not words:
            continue
        labels = word_labels(words, r.get("aspects", []))
        ids, first = encode_words(words, tok, max_len)
        y = [-100] * len(ids)
        for k, i in enumerate(first):
            y[i] = BIO2ID[labels[k]]
        gold = {(a["category"], int(a["start"]), int(a["end"])) for a in r.get("aspects", []) if int(a.get("start", -1)) >= 0}
        data.append({"ids": ids, "y": y, "words": words[: len(first)], "text": text, "gold": gold, "first": first})
    return data


def batches(data, bs, pad, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [data[i] for i in idx[s:s + bs]]
        L = max(len(d["ids"]) for d in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        y = torch.full((len(chunk), L), -100, dtype=torch.long)
        for i, d in enumerate(chunk):
            ids[i, :len(d["ids"])] = torch.tensor(d["ids"])
            y[i, :len(d["y"])] = torch.tensor(d["y"])
        yield chunk, ids, (ids != pad).long(), y


@torch.no_grad()
def span_f1(model, data, pad, device):
    from mabsa.models.extractor import BIO_LABELS
    model.eval()
    tp = fp = fn = 0
    for chunk, ids, mask, _ in batches(data, 32, pad, False):
        logits = model(ids.to(device), mask.to(device)).logits.cpu()
        for i, d in enumerate(chunk):
            tags = [BIO_LABELS[int(logits[i, j].argmax())] for j in d["first"]]
            pred = {(s["category"], s["start"], s["end"]) for s in decode_spans(d["text"], d["words"], tags)}
            tp += len(pred & d["gold"])
            fp += len(pred - d["gold"])
            fn += len(d["gold"] - pred)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", default="checkpoints/extractor_best.pt")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["train"].get("seed", 42))
    device = get_device(cfg["train"].get("device", "auto"))
    tok = build_tokenizer_from_cfg(cfg)
    seg = get_segmenter(cfg)
    _, _, pad = special_ids(tok)
    max_len = cfg["data"]["max_seq_len"]
    train = encode_rows(read_jsonl(resolve(cfg["data"]["train"])), tok, seg, max_len)
    val = encode_rows(read_jsonl(resolve(cfg["data"]["val"])), tok, seg, max_len)
    model = BIOAspectExtractor(cfg, vocab_size=len(tok)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for _, ids, mask, y in batches(train, args.batch_size, pad, True):
            loss = model(ids.to(device), mask.to(device), y.to(device)).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        f1 = span_f1(model, val, pad, device)
        log.info("Epoch %d | loss %.4f | val span-F1 %.4f", ep, tot, f1)
        if f1 > best:
            best, bad = f1, 0
            out = resolve(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "val_span_f1": f1}, out)
        else:
            bad += 1
            if bad >= 3:
                break
    log.info("Span-F1 tốt nhất trên validation: %.4f → %s", best, args.output)


if __name__ == "__main__":
    main()
