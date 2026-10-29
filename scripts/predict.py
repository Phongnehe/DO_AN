"""Dự đoán nhanh từ dòng lệnh (không cần giao diện).

    python scripts/predict.py --text "Món ăn rất ngon nhưng giá hơi cao" --images data/raw/images/a.jpg
"""
import _bootstrap  # noqa: F401
import argparse
import json

from mabsa.config import resolve
from mabsa.data.preprocess import load_image
from mabsa.predict import MABSAPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--checkpoint", default="checkpoints/mabsa_best.pt")
    ap.add_argument("--extractor", default="checkpoints/extractor_best.pt")
    args = ap.parse_args()
    pred = MABSAPredictor(args.checkpoint, args.extractor)
    res = pred.predict(args.text, [load_image(resolve(p)) for p in args.images])
    for a in res["aspects"]:
        a.pop("attention", None)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
