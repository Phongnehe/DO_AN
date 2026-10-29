#!/usr/bin/env python
"""Chuyển ViMACSA (train/dev/test.json + roi_data.csv) sang JSONL của MABSA-Travel.

Dùng nguyên split chính thức: train -> train.jsonl, dev -> val.jsonl, test -> test.jsonl.
Nhãn là cấp danh mục khía cạnh nên start = end = -1 (khía cạnh ẩn, mô hình lấy token đầu câu).
Vùng ảnh (RoI) đổi từ pixel sang tọa độ chuẩn hóa [x1, y1, x2, y2] trong [0, 1].
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# ViMACSA -> mã khía cạnh của dự án (src/mabsa/aspects.py). Public_area không có mã riêng nên gán SPACE.
ASPECT_MAP = {"location": "LOCATION", "food": "FOOD", "room": "ROOM",
              "facilities": "FACILITY", "service": "SERVICE", "public_area": "SPACE"}
SENT_MAP = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
SPLITS = {"train": "train", "dev": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/vimacsa")
    ap.add_argument("--img-subdir", default="image/image")
    ap.add_argument("--out-dir", default="data/processed/vimacsa")
    ap.add_argument("--labels", default="text_img_label", choices=["text_img_label", "text_label"],
                    help="text_img_label: nhãn của bài toán đa phương thức (gồm cả khía cạnh chỉ thấy trong ảnh)")
    ap.add_argument("--lexicon-spans", action="store_true",
                    help="gán start/end bằng từ điển ASPECT_LEXICON khi tìm thấy từ khóa trong câu")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N mẫu mỗi tập (chạy thử nhanh)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--image-manifest', default=None,
                    help='manifest of resized images including original dimensions')
    ap.add_argument('--max-images', type=int, default=7)
    args = ap.parse_args()

    D, IMG = ROOT / args.data_dir, ROOT / args.data_dir / args.img_subdir
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.image_manifest).read_text()) if args.image_manifest else {}

    def image_path(fn):
        return Path(manifest[fn]['path']) if fn in manifest else IMG / fn

    roi = pd.read_csv(D / "roi_data.csv")
    rois = {}
    for r in roi.itertuples(index=False):
        rois.setdefault(str(r.file_name), []).append(
            (str(r.label).strip().lower(), float(r.x1), float(r.y1), float(r.x2), float(r.y2)))
    size_cache = {}

    def img_size(p):
        if p not in size_cache:
            with Image.open(p) as im:
                size_cache[p] = im.size
        return size_cache[p]

    lex = None
    if args.lexicon_spans:
        from mabsa.aspects import ASPECT_LEXICON
        lex = {c: sorted(v, key=len, reverse=True) for c, v in ASPECT_LEXICON.items()}

    rng = random.Random(args.seed)
    stat = Counter()
    for name, outname in SPLITS.items():
        data = json.load(open(D / f"{name}.json", encoding="utf-8"))
        if args.limit:
            data = rng.sample(data, min(args.limit, len(data)))
        rows = []
        for i, s in enumerate(data):
            text = " ".join(str(s["comment"]).split())
            text_sentiments = {}
            for text_lab in s.get('text_label', []):
                text_cat, text_sent = text_lab.split('#')
                text_sentiments[text_cat.strip().lower()] = SENT_MAP[text_sent.strip().lower()]
            kept = []
            for fn in s["list_img"][:args.max_images]:
                if image_path(fn).exists():
                    kept.append(fn)
                else:
                    stat["ảnh thiếu"] += 1
            seen, aspects = set(), []
            for lab in s[args.labels]:
                try:
                    cat, sent = lab.split("#")
                    cat, sent = cat.strip().lower(), sent.strip().lower()
                    code, vi = ASPECT_MAP[cat], SENT_MAP[sent]
                except (ValueError, KeyError):
                    stat["nhãn không nhận ra"] += 1
                    continue
                if cat in seen:
                    stat["nhãn trùng khía cạnh (bỏ)"] += 1
                    continue
                seen.add(cat)
                regions = []
                for k, fn in enumerate(kept):
                    if fn in manifest:
                        w, h = manifest[fn]['width'], manifest[fn]['height']
                    else:
                        w, h = img_size(IMG / fn)
                    for lb, x1, y1, x2, y2 in rois.get(Path(fn).stem, []):
                        if lb != cat:
                            continue
                        x1, x2 = sorted((min(max(x1 / w, 0.0), 1.0), min(max(x2 / w, 0.0), 1.0)))
                        y1, y2 = sorted((min(max(y1 / h, 0.0), 1.0), min(max(y2 / h, 0.0), 1.0)))
                        if x2 > x1 and y2 > y1:
                            regions.append({"image": k, "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
                term, st, en = "", -1, -1
                if lex:
                    low = text.lower()
                    for kw in lex[code]:
                        p = low.find(kw)
                        if p >= 0:
                            term, st, en = text[p:p + len(kw)], p, p + len(kw)
                            break
                aspects.append({"category": code, "term": term, "start": st, "end": en,
                                "sentiment": vi, 'text_sentiment': text_sentiments.get(cat),
                                "regions": regions})
                stat[f"{outname}:cảm xúc:{vi}"] += 1
                stat[f"{outname}:khía cạnh:{code}"] += 1
                stat["tổng khía cạnh"] += 1
                stat["khía cạnh có vùng ảnh"] += bool(regions)
                stat["khía cạnh có span"] += (st >= 0)
            if not aspects:
                stat["mẫu bỏ (không có nhãn)"] += 1
                continue
            rows.append({"id": f"VIMACSA_{name}_{i}", "source": "vimacsa", "service_type": "hotel",
                         "text": text, "images": [str(image_path(fn).resolve()) for fn in kept],
                         "aspects": aspects})
        with open(out / f"{outname}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stat[f"{outname}:mẫu"] = len(rows)
        print(f"{outname}: {len(rows)} mẫu -> {out / (outname + '.jsonl')}")
    print("\nThống kê:")
    for k in sorted(stat):
        print(f"  {k}: {stat[k]}")
    if stat["tổng khía cạnh"]:
        print(f"\nTỷ lệ khía cạnh có vùng ảnh: {stat['khía cạnh có vùng ảnh'] / stat['tổng khía cạnh']:.1%}")


if __name__ == "__main__":
    main()
