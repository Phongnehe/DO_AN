"""Bước 2 – Gán nhãn bán tự động (mục 3.3.4): gợi ý cụm từ khía cạnh để người gán nhãn kiểm tra, sửa.

Mặc định dùng từ điển từ khóa; nếu đã có bộ nhận dạng khía cạnh (--extractor), dùng mô hình đó.
Cảm xúc và vùng ảnh (bbox) để trống cho người gán nhãn hoàn thiện theo hướng dẫn gán nhãn.

    python scripts/pre_annotate.py --input data/processed/reviews_clean.jsonl --output data/annotations/draft.jsonl
"""
import _bootstrap  # noqa: F401
import argparse

from mabsa.config import resolve
from mabsa.models.extractor import LexiconAspectExtractor
from mabsa.utils import get_logger, read_jsonl, write_jsonl

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/reviews_clean.jsonl")
    ap.add_argument("--output", default="data/annotations/draft.jsonl")
    ap.add_argument("--extractor", default=None, help="checkpoints/extractor_best.pt (tùy chọn)")
    args = ap.parse_args()
    rows = read_jsonl(resolve(args.input))
    if args.extractor:
        import torch
        from mabsa.builder import build_tokenizer_from_cfg, get_segmenter
        from mabsa.models.extractor import BIOAspectExtractor
        ck = torch.load(resolve(args.extractor), map_location="cpu", weights_only=False)
        tok = build_tokenizer_from_cfg(ck["config"])
        ext = BIOAspectExtractor(ck["config"], vocab_size=len(tok))
        ext.load_state_dict(ck["state_dict"])
        seg = get_segmenter(ck["config"])
        extract = lambda t: ext.extract(t, tok, seg)  # noqa: E731
    else:
        lex = LexiconAspectExtractor()
        extract = lex.extract
    n = 0
    for r in rows:
        r["aspects"] = [dict(sp, sentiment=None, regions=[]) for sp in extract(r["text"])]
        n += len(r["aspects"])
    write_jsonl(rows, resolve(args.output))
    log.info("Đã gợi ý %d khía cạnh cho %d mẫu → %s", n, len(rows), args.output)


if __name__ == "__main__":
    main()
