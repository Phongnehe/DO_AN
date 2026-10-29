"""Sinh bộ dữ liệu GIẢ đúng định dạng để chạy thử toàn bộ pipeline (không dùng cho báo cáo).

    python scripts/make_toy_data.py --n 400
Tạo: data/toy/dataset.jsonl, data/toy/images/*.jpg, data/toy/vocab.txt, data/toy/raw/reviews.csv
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw

from mabsa.config import ROOT
from mabsa.utils import write_jsonl

PHRASES = {
    "FOOD": (["món ăn", "đồ ăn", "cà phê"], {"Tích cực": ["rất ngon", "đậm đà", "tuyệt vời"],
             "Tiêu cực": ["hơi nhạt", "không ngon", "nguội lạnh"], "Trung lập": ["bình thường", "tạm được"]}),
    "SERVICE": (["nhân viên", "phục vụ"], {"Tích cực": ["nhiệt tình", "thân thiện"],
                "Tiêu cực": ["chậm chạp", "thờ ơ"], "Trung lập": ["bình thường"]}),
    "PRICE": (["giá", "giá cả"], {"Tích cực": ["hợp lý", "rẻ"], "Tiêu cực": ["hơi cao", "đắt đỏ"],
              "Trung lập": ["tầm trung"]}),
    "ROOM": (["phòng", "phòng ngủ"], {"Tích cực": ["rộng rãi", "thoáng mát"], "Tiêu cực": ["chật chội", "ẩm thấp"],
             "Trung lập": ["vừa đủ"]}),
    "SCENERY": (["view", "cảnh"], {"Tích cực": ["rất đẹp", "hùng vĩ"], "Tiêu cực": ["bị che khuất", "không đẹp"],
                "Trung lập": ["cũng được"]}),
    "SPACE": (["không gian"], {"Tích cực": ["ấm cúng", "yên tĩnh"], "Tiêu cực": ["ồn ào", "chật hẹp"],
              "Trung lập": ["bình thường"]}),
}
VISUAL = {"FOOD": "circle", "ROOM": "rect", "SCENERY": "triangle"}
FILLERS = ["chúng tôi đã ghé đây vào cuối tuần cùng gia đình", "đây là lần đầu tiên tôi đến địa điểm này",
           "chuyến đi kéo dài hai ngày một đêm", "tôi đặt chỗ trước qua ứng dụng"]
CONNECT = [", ", " và ", " nhưng ", ". "]


def draw_image(path, visual_aspects, sentiments, rng):
    img = Image.new("RGB", (256, 256), (rng.randint(180, 230),) * 3)
    d = ImageDraw.Draw(img)
    regions = {}
    for asp in visual_aspects:
        w = rng.randint(60, 110)
        x, y = rng.randint(0, 256 - w), rng.randint(0, 256 - w)
        bright = sentiments[asp] == "Tích cực"
        color = {"circle": (230, 140, 40), "rect": (60, 110, 200), "triangle": (40, 160, 70)}[VISUAL[asp]]
        if not bright:
            color = tuple(c // 2 for c in color)
        if VISUAL[asp] == "circle":
            d.ellipse([x, y, x + w, y + w], fill=color)
        elif VISUAL[asp] == "rect":
            d.rectangle([x, y, x + w, y + int(w * 0.7)], fill=color)
        else:
            d.polygon([(x, y + w), (x + w // 2, y), (x + w, y + w)], fill=color)
        regions[asp] = [round(x / 256, 3), round(y / 256, 3), round((x + w) / 256, 3), round((y + w) / 256, 3)]
    img.save(path, quality=90)
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rows, raw = [], []
    sources = ["tripadvisor", "booking", "traveloka", "googlemaps"]
    for i in range(args.n):
        k = rng.randint(1, 3)
        cats = rng.sample(list(PHRASES), k)
        sents = {c: rng.choices(["Tích cực", "Tiêu cực", "Trung lập"], [0.6, 0.27, 0.13])[0] for c in cats}
        text, aspects = rng.choice(FILLERS).capitalize() + ". ", []
        for n_, c in enumerate(cats):
            terms, ph = PHRASES[c]
            term = rng.choice(terms)
            start = len(text)
            text += term
            aspects.append({"category": c, "term": term, "start": start, "end": start + len(term),
                            "sentiment": sents[c], "regions": []})
            text += " " + rng.choice(ph[sents[c]])
            text += rng.choice(CONNECT) if n_ < len(cats) - 1 else "."
        vis = [c for c in cats if c in VISUAL]
        sid = f"TOY_{i:05d}"
        n_img = rng.choice([1, 1, 2])
        paths = []
        for j in range(n_img):
            p = f"{args.out}/images/{sid}_{j + 1}.jpg"
            regs = draw_image(ROOT / p, vis if j == 0 else [], sents, rng)
            for a in aspects:
                if a["category"] in regs:
                    a["regions"].append({"image": j, "bbox": regs[a["category"]]})
            paths.append(p)
        src = rng.choice(sources)
        rows.append({"id": sid, "source": src, "service_type": rng.choice(["hotel", "restaurant", "attraction"]),
                     "text": text, "images": paths, "aspects": aspects})
        raw.append({"id": sid, "source": src, "service_type": rows[-1]["service_type"], "place": f"Địa điểm {i % 40}",
                    "province": rng.choice(["Hà Nội", "Đà Nẵng", "Đà Lạt", "TP. Hồ Chí Minh"]), "rating": rng.randint(1, 5),
                    "date": "2025-06-01", "text": text, "images": ";".join(paths)})
    write_jsonl(rows, out / "dataset.jsonl")
    with open(out / "raw" / "reviews.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)
    words = sorted({w for r in rows for w in re.findall(r"\w+", r["text"].lower())})
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "!", "?", ":", ";", "(", ")", "-", "/", "%"]
    (out / "vocab.txt").write_text("\n".join(specials + words) + "\n", encoding="utf-8")
    print(f"Đã tạo {len(rows)} mẫu giả tại {out}")


if __name__ == "__main__":
    main()
