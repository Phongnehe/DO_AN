"""Bước 1 – Làm sạch dữ liệu thô sau thu thập (mục 3.1.3, 3.1.4, 3.2).

Đầu vào: file CSV do người thu thập ghi (mỗi dòng một đánh giá), cột:
    id, source, service_type, place, province, rating, date, text, images
    (images: các đường dẫn ảnh cách nhau bởi dấu ';', tính từ thư mục gốc dự án)
Đầu ra: data/processed/reviews_clean.jsonl (sẵn sàng để gán nhãn) + báo cáo số mẫu bị loại theo lý do.

    python scripts/build_dataset.py --input data/raw/reviews.csv
"""
import _bootstrap  # noqa: F401
import argparse
import csv
from collections import Counter

from mabsa.aspects import SERVICE_TYPES
from mabsa.config import resolve
from mabsa.data.dedup import find_image_duplicates, find_text_duplicates
from mabsa.data.preprocess import MIN_WORDS, count_words, image_is_valid, looks_like_spam, normalize_text
from mabsa.utils import get_logger, write_json, write_jsonl

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/reviews.csv")
    ap.add_argument("--output", default="data/processed/reviews_clean.jsonl")
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--text-dup-threshold", type=float, default=0.9)
    args = ap.parse_args()

    with open(resolve(args.input), encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    log.info("Đọc %d đánh giá thô", len(raw))
    removed, kept = Counter(), []
    for r in raw:
        text = normalize_text(r.get("text", ""))
        if count_words(text) < MIN_WORDS:
            removed["văn bản < 10 từ"] += 1
            continue
        if looks_like_spam(text):
            removed["spam/quảng cáo"] += 1
            continue
        stype = (r.get("service_type") or "").strip().lower()
        if stype and stype not in SERVICE_TYPES:
            removed["loại hình dịch vụ không hợp lệ"] += 1
            continue
        imgs = [p.strip() for p in (r.get("images") or "").split(";") if p.strip()]
        valid = [p for p in imgs if image_is_valid(resolve(p))][: args.max_images]
        if not valid:
            removed["không có ảnh hợp lệ (≥224×224)"] += 1
            continue
        kept.append({"id": r["id"], "source": r.get("source", ""), "service_type": stype,
                     "place": r.get("place", ""), "province": r.get("province", ""),
                     "rating": r.get("rating", ""), "date": r.get("date", ""),
                     "text": text, "images": valid, "aspects": []})

    drop = find_text_duplicates([k["text"] for k in kept], args.text_dup_threshold)
    removed["trùng lặp văn bản (cosine > %.2f)" % args.text_dup_threshold] += len(drop)
    kept = [k for i, k in enumerate(kept) if i not in drop]
    drop = find_image_duplicates([k["images"] for k in kept], root=resolve("."))
    removed["trùng lặp ảnh (perceptual hash)"] += len(drop)
    kept = [k for i, k in enumerate(kept) if i not in drop]

    write_jsonl(kept, resolve(args.output))
    report = {"input": len(raw), "kept": len(kept), "removed": dict(removed),
              "by_source": dict(Counter(k["source"] for k in kept))}
    write_json(report, resolve(args.output).with_suffix(".report.json"))
    log.info("Giữ lại %d / %d mẫu. Loại bỏ: %s", len(kept), len(raw), dict(removed))


if __name__ == "__main__":
    main()
