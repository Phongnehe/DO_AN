"""Bước 3 – Đo độ đồng thuận giữa người gán nhãn (mục 3.3.5, Bảng 3.4).

Mỗi người gán nhãn nộp một file JSONL trên CÙNG tập mẫu chung (cùng id):
    python scripts/compute_kappa.py data/annotations/annotator_A.jsonl data/annotations/annotator_B.jsonl
2 người → Cohen's Kappa, ≥ 3 người → Fleiss' Kappa. Ngưỡng yêu cầu: Kappa ≥ 0.7.
"""
import _bootstrap  # noqa: F401
import argparse

from mabsa.aspects import ASPECT_CODES, normalize_aspect, normalize_sentiment
from mabsa.config import resolve
from mabsa.data.kappa import interpret, kappa
from mabsa.experiments import write_table
from mabsa.utils import read_jsonl


def by_category(row):
    out = {}
    for a in row.get("aspects", []):
        try:
            out.setdefault(normalize_aspect(a["category"]), a)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="outputs/stats/bang_3_4_kappa")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    anns = [{r["id"]: r for r in read_jsonl(resolve(f))} for f in args.files]
    ids = sorted(set.intersection(*[set(a) for a in anns]))
    print(f"{len(args.files)} người gán nhãn, {len(ids)} mẫu chung")

    # 1) Khía cạnh: với mỗi (mẫu, khía cạnh) – có/không được gán
    aspect_lists = [[int(c in by_category(a[i])) for i in ids for c in ASPECT_CODES] for a in anns]
    # 2) Cảm xúc: trên các khía cạnh mà mọi người cùng gán
    pos_neg, neutral = [[] for _ in anns], [[] for _ in anns]
    link = [[] for _ in anns]
    for i in ids:
        cats = [by_category(a[i]) for a in anns]
        shared = set.intersection(*[set(c) for c in cats])
        for c in sorted(shared):
            labs = [normalize_sentiment(cc[c]["sentiment"]) for cc in cats]
            target = neutral if "Trung lập" in labs else pos_neg
            for k, l in enumerate(labs):
                target[k].append(l)
            # 3) Liên kết ảnh–khía cạnh: khía cạnh có gắn với từng ảnh hay không
            n_img = len(anns[0][i].get("images", []))
            for img in range(n_img):
                for k, cc in enumerate(cats):
                    link[k].append(int(any(int(r.get("image", 0)) == img for r in cc[c].get("regions", []) or [])))
    rows = []
    for name, lists in [("Khía cạnh", aspect_lists), ("Cảm xúc tích cực/tiêu cực", pos_neg),
                        ("Cảm xúc trung lập", neutral), ("Liên kết ảnh-khía cạnh", link)]:
        if not lists[0]:
            continue
        k = kappa(lists)
        rows.append({"Tiêu chí": name, "Hệ số Kappa": round(k, 3), "Mức độ đồng thuận": interpret(k),
                     "Đạt ngưỡng": "Có" if k >= args.threshold else "Không", "Số mục": len(lists[0])})
    for r in rows:
        print(f"{r['Tiêu chí']:<28} {r['Hệ số Kappa']:.3f}  {r['Mức độ đồng thuận']:<10} đạt ngưỡng: {r['Đạt ngưỡng']}")
    write_table(rows, resolve(args.out))
    if any(r["Đạt ngưỡng"] == "Không" for r in rows):
        print("→ Có tiêu chí dưới ngưỡng: rà soát lại hướng dẫn gán nhãn và gán nhãn lại tập mẫu (mục 3.3.5).")


if __name__ == "__main__":
    main()
