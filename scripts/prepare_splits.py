"""Bước 4 – Kiểm tra nhãn, chia 80:10:10 và tạo tập con 2k–10k (mục 3.4.5), xuất thống kê (mục 3.4).

    python scripts/prepare_splits.py --input data/annotations/final.jsonl
Kết quả:
    data/processed/dataset.jsonl
    data/processed/splits/{train,val,test}.jsonl            (Bảng 3.8)
    data/processed/subsets/D2k..D10k/{train,val,test}.jsonl (Bảng 3.9)
    outputs/stats/*.json|md, outputs/stats/hinh_3_4.png     (Bảng 3.5–3.7, Hình 3.4)
"""
import _bootstrap  # noqa: F401
import argparse
import glob

from mabsa.aspects import normalize_aspect, normalize_sentiment
from mabsa.config import resolve
from mabsa.data.dataset import locate_aspect
from mabsa.data.preprocess import normalize_text
from mabsa.data.split import label_distribution, nested_subsets, stratified_split
from mabsa.data.stats import dataset_stats, plot_length_and_images
from mabsa.experiments import write_table
from mabsa.utils import get_logger, read_jsonl, write_json, write_jsonl

log = get_logger()


def validate(rows):
    """Chuẩn hóa nhãn và loại mẫu lỗi; trả về (mẫu hợp lệ, danh sách lỗi)."""
    good, errors = [], []
    for r in rows:
        r["text"] = normalize_text(r["text"])
        aspects = []
        for a in r.get("aspects", []):
            try:
                a["category"] = normalize_aspect(a["category"])
                a["sentiment"] = normalize_sentiment(a["sentiment"])
            except (ValueError, KeyError, TypeError) as e:
                errors.append(f"{r.get('id')}: {e}")
                continue
            s, e_ = locate_aspect(r["text"], a)
            if a.get("term") and s < 0:
                errors.append(f"{r.get('id')}: không tìm thấy cụm từ '{a['term']}' trong văn bản")
            a["start"], a["end"] = s, e_
            for reg in a.get("regions", []) or []:
                if int(reg.get("image", 0)) >= len(r.get("images", [])):
                    errors.append(f"{r.get('id')}: vùng ảnh trỏ tới ảnh không tồn tại")
            aspects.append(a)
        if not aspects or not r.get("images"):
            errors.append(f"{r.get('id')}: không có khía cạnh hợp lệ hoặc không có ảnh → loại")
            continue
        r["aspects"] = aspects
        good.append(r)
    return good, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/annotations/final.jsonl", help="có thể dùng glob, vd. data/annotations/*.jsonl")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sizes", default="2000,4000,6000,8000,10000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-dir", default="outputs/stats")
    args = ap.parse_args()

    files = sorted(glob.glob(str(resolve(args.input))))
    rows = [r for f in files for r in read_jsonl(f)]
    rows, errors = validate(rows)
    out = resolve(args.out)
    if errors:
        (out / "validation_errors.txt").parent.mkdir(parents=True, exist_ok=True)
        (out / "validation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        log.warning("%d cảnh báo khi kiểm tra nhãn → %s", len(errors), out / "validation_errors.txt")
    write_jsonl(rows, out / "dataset.jsonl")

    splits = stratified_split(rows, (0.8, 0.1, 0.1), args.seed)
    for name, sub in splits.items():
        write_jsonl(sub, out / "splits" / f"{name}.jsonl")
        log.info("%-5s %5d mẫu | phân bố cảm xúc %s", name, len(sub), label_distribution(sub))
    sizes = [int(s) for s in args.sizes.split(",") if int(s) <= len(rows)]
    subsets = nested_subsets(splits, sizes, (0.8, 0.1, 0.1), args.seed)
    table = []
    for name, sub in subsets.items():
        for part, rs in sub.items():
            write_jsonl(rs, out / "subsets" / name / f"{part}.jsonl")
        table.append({"Tập con": name, "Tổng số mẫu": sum(len(v) for v in sub.values()),
                      "Train (80%)": len(sub["train"]), "Validation (10%)": len(sub["val"]), "Test (10%)": len(sub["test"])})

    st = dataset_stats(rows, {"huấn luyện": splits["train"], "kiểm định": splits["val"], "kiểm tra": splits["test"]})
    sd = resolve(args.stats_dir)
    write_json(st["overview"], sd / "bang_3_5_tong_quan.json")
    write_table(st["aspects"], sd / "bang_3_6_phan_bo_khia_canh")
    write_table(st["sentiments"], sd / "bang_3_7_phan_bo_cam_xuc")
    write_table([{"Tập": k, "Số mẫu": len(v)} for k, v in splits.items()], sd / "bang_3_8_phan_chia")
    write_table(table, sd / "bang_3_9_tap_con")
    plot_length_and_images(st, sd / "hinh_3_4.png")
    log.info("Tổng quan: %s", st["overview"])


if __name__ == "__main__":
    main()
