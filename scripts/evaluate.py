"""Đánh giá một checkpoint trên một tập bất kỳ (mục 5.1.3, 5.3.1, 5.3.2).

    python scripts/evaluate.py --checkpoint outputs/mabsa_default/best.pt --data data/processed/splits/test.jsonl
Xuất: metrics (Accuracy, Precision, Recall, Macro/Micro F1), kết quả theo khía cạnh (Bảng 5.4), confusion matrix.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import torch

from mabsa.builder import build_loader, build_model, build_tokenizer_from_cfg
from mabsa.config import resolve
from mabsa.engine import evaluate, save_eval
from mabsa.utils import get_device, read_jsonl, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed/splits/test.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()
    ck = load_checkpoint(resolve(args.checkpoint))
    cfg = ck["config"]
    device = get_device(cfg["train"].get("device", "auto"))
    tok = build_tokenizer_from_cfg(cfg)
    model = build_model(cfg, tok)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    loader = build_loader(read_jsonl(resolve(args.data)), cfg, tok)
    res, recs = evaluate(model, loader, device)
    out = Path(args.out) if args.out else resolve(args.checkpoint).parent
    full = save_eval(res, recs, resolve(out), args.name)
    print(f"Accuracy {full['accuracy']:.2%} | Macro P {full['macro_precision']:.2%} | Macro R {full['macro_recall']:.2%} "
          f"| Macro F1 {full['macro_f1']:.2%} | Micro F1 {full['micro_f1']:.2%}")
    print("\nKết quả theo khía cạnh:")
    for r in full["per_aspect"]:
        print(f"  {r['aspect']:<20} n={r['n_test']:<5} P={r['precision']:.2%}  R={r['recall']:.2%}  F1={r['f1']:.2%}")


if __name__ == "__main__":
    main()
