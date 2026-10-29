"""Thống kê mô tả bộ dữ liệu (mục 3.4): Bảng 3.5, 3.6, 3.7 và Hình 3.4."""
from collections import Counter
from pathlib import Path

import numpy as np

from mabsa.aspects import ASPECT_CODES, CODE2NAME, SENTIMENTS, normalize_aspect, normalize_sentiment
from mabsa.data.preprocess import count_words


def dataset_stats(rows, splits=None) -> dict:
    n = len(rows)
    n_aspects = [len(r.get("aspects", [])) for r in rows]
    n_images = [len(r.get("images", [])) for r in rows]
    lengths = [count_words(r["text"]) for r in rows]
    overview = {"Tổng số mẫu": n}
    for name, sub in (splits or {}).items():
        overview[f"Số mẫu {name} ({len(sub) / max(n, 1):.0%})"] = len(sub)
    overview.update({
        "Số khía cạnh trung bình/mẫu": round(float(np.mean(n_aspects)), 2) if n else 0,
        "Số ảnh trung bình/mẫu": round(float(np.mean(n_images)), 2) if n else 0,
        "Độ dài văn bản trung bình (từ)": round(float(np.mean(lengths)), 1) if n else 0,
        "Độ lệch chuẩn độ dài (từ)": round(float(np.std(lengths)), 1) if n else 0,
    })
    # Bảng 3.6: số mẫu có chứa từng khía cạnh
    per_aspect = Counter()
    for r in rows:
        for c in {normalize_aspect(a["category"]) for a in r.get("aspects", [])}:
            per_aspect[c] += 1
    aspect_table = [{"Khía cạnh": CODE2NAME[c], "Số mẫu": per_aspect.get(c, 0),
                     "Tỷ lệ (%)": round(100 * per_aspect.get(c, 0) / max(n, 1), 2)} for c in ASPECT_CODES]
    aspect_table.sort(key=lambda x: -x["Số mẫu"])
    # Bảng 3.7: phân bố nhãn cảm xúc (theo cặp khía cạnh–cảm xúc)
    sent = Counter(normalize_sentiment(a["sentiment"]) for r in rows for a in r.get("aspects", []))
    tot = sum(sent.values()) or 1
    sent_table = [{"Nhãn cảm xúc": s, "Số lượng": sent.get(s, 0), "Tỷ lệ (%)": round(100 * sent.get(s, 0) / tot, 2)}
                  for s in SENTIMENTS]
    return {"overview": overview, "aspects": aspect_table, "sentiments": sent_table,
            "_lengths": lengths, "_n_images": n_images}


def plot_length_and_images(stats: dict, path):
    """Hình 3.4: phân bố độ dài văn bản và số lượng ảnh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths, n_images = stats["_lengths"], stats["_n_images"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
    a1.hist(lengths, bins=range(0, max(lengths + [10]) + 10, 10), color="white", edgecolor="black")
    a1.axvline(np.mean(lengths), ls="--", color="black")
    a1.set_title("(a) Phân bố độ dài văn bản", fontweight="bold")
    a1.set_xlabel("Số từ trong văn bản")
    a1.set_ylabel("Số mẫu")
    c = Counter(n_images)
    xs = sorted(c)
    a2.bar(xs, [c[x] for x in xs], width=1, color="#E6E6E6", edgecolor="black")
    a2.axvline(np.mean(n_images), ls="--", color="black")
    a2.set_title("(b) Phân bố số lượng ảnh", fontweight="bold")
    a2.set_xlabel("Số lượng ảnh / đánh giá")
    a2.set_ylabel("Số mẫu")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
