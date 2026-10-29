"""Huấn luyện mô hình đề xuất (hoặc một baseline) và đánh giá trên tập test.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --set model.image_encoder=clip train.batch_size=8
    python scripts/train.py --config configs/toy.yaml            # chạy thử nhanh trên CPU
Checkpoint tốt nhất: <output_dir>/best.pt ; thêm --export để chép sang checkpoints/mabsa_best.pt cho app.
"""
import _bootstrap  # noqa: F401
import argparse
import shutil

from mabsa.config import load_config, resolve
from mabsa.engine import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè cấu hình, vd. train.epochs=10")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--export", action="store_true", help="chép best.pt sang checkpoints/mabsa_best.pt")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = args.output_dir or cfg["train"]["output_dir"]
    res = train(cfg, out)
    t = res["test"]
    print(f"\nTEST  Accuracy {t['accuracy']:.2%} | Macro F1 {t['macro_f1']:.2%} | Micro F1 {t['micro_f1']:.2%}")
    print(f"Kết quả chi tiết: {resolve(out)}")
    if args.export:
        dst = resolve("checkpoints/mabsa_best.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(resolve(out) / "best.pt", dst)
        print(f"Đã chép checkpoint cho ứng dụng: {dst}")


if __name__ == "__main__":
    main()
