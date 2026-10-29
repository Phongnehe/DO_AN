"""Các nhóm thực nghiệm của Chương 5: baseline (Bảng 5.3), ablation (Bảng 5.5), quy mô dữ liệu (mục 5.1.4)."""
import copy
import csv
from pathlib import Path

import numpy as np

from mabsa.config import load_config, resolve
from mabsa.engine import train
from mabsa.utils import get_logger, write_json

log = get_logger()

# Bảng 5.2 / 5.3 – các phương pháp so sánh
BASELINES = {
    "PhoBERT (chỉ văn bản)": {"model.type": "text_only"},
    "ViT (chỉ ảnh)": {"model.type": "image_only", "model.image_encoder": "vit"},
    "ResNet (chỉ ảnh)": {"model.type": "image_only", "model.image_encoder": "resnet"},
    "Early fusion": {"model.type": "early_fusion"},
    "Late fusion": {"model.type": "late_fusion"},
    "Cross-attention fusion (không contrastive)": {"model.type": "mabsa", "model.use_contrastive": False},
    "Đề xuất (Transformer + Contrastive)": {"model.type": "mabsa"},
}

# Bảng 5.5 – ablation study
ABLATIONS = {
    "Mô hình đầy đủ": {},
    "Loại bỏ contrastive loss": {"model.use_contrastive": False},
    "Loại bỏ cross-attention": {"model.use_cross_attention": False},
    "Thay PhoBERT bằng XLM-R": {"model.text_encoder": "xlmr", "data.segmenter": "none"},
    "Thay PhoBERT bằng ViDeBERTa": {"model.text_encoder": "videberta", "data.segmenter": "none"},
    "Thay ViT bằng ResNet": {"model.image_encoder": "resnet"},
    "Thay ViT bằng CLIP visual": {"model.image_encoder": "clip"},
}

# mục 5.1.4 – các mô hình dùng trong thực nghiệm quy mô dữ liệu
DATA_SCALE_MODELS = {
    "Đề xuất": {"model.type": "mabsa"},
    "PhoBERT (chỉ văn bản)": {"model.type": "text_only"},
    "Cross-attention fusion (không contrastive)": {"model.type": "mabsa", "model.use_contrastive": False},
}

SUMMARY_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1"]


def slug(name: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def make_cfg(config_path, overrides: dict, extra=None):
    base = load_config(config_path, extra)
    if base["model"]["text_encoder"] == "tiny":  # chế độ chạy thử: giữ bộ mã hóa tí hon
        overrides = {k: v for k, v in overrides.items()
                     if k not in ("model.text_encoder", "model.image_encoder", "data.segmenter")}
    cli = [f"{k}={v}" for k, v in overrides.items()] + list(extra or [])
    return load_config(config_path, cli)


def run_group(config_path, variants: dict, out_root, extra=None, only=None):
    """Huấn luyện + đánh giá lần lượt từng biến thể; trả về bảng kết quả trên tập test."""
    rows = []
    for name, ov in variants.items():
        if only and name not in only:
            continue
        cfg = make_cfg(config_path, ov, extra)
        out = Path(resolve(out_root)) / slug(name)
        log.info("===== %s =====", name)
        res = train(cfg, out)
        rows.append({"model": name, **{k: res["test"][k] for k in SUMMARY_KEYS}})
    write_table(rows, Path(resolve(out_root)) / "results")
    return rows


def run_data_scale(config_path, subsets_dir, full_test, out_root, seeds=(42, 43, 44),
                   sizes=("D2k", "D4k", "D6k", "D8k", "D10k"), models=None, extra=None):
    """Mục 5.1.4: huấn luyện trên từng tập con × mô hình × seed; đánh giá trên Test của tập con và Test đầy đủ."""
    from mabsa.utils import read_jsonl
    models = models or DATA_SCALE_MODELS
    rows = []
    curve = {m: {} for m in models}
    for size in sizes:
        sub = Path(resolve(subsets_dir)) / size
        n_train = len(read_jsonl(sub / "train.jsonl"))
        for mname, ov in models.items():
            scores = {"subset_test": [], "full_test": []}
            for seed in seeds:
                data_ov = {"data.train": str(sub / "train.jsonl"), "data.val": str(sub / "val.jsonl"),
                           "data.test": str(sub / "test.jsonl"), "train.seed": seed}
                cfg = make_cfg(config_path, {**ov, **data_ov}, extra)
                out = Path(resolve(out_root)) / size / slug(mname) / f"seed{seed}"
                log.info("===== %s | %s | seed %d =====", size, mname, seed)
                res = train(cfg, out, extra_eval={"full_test": full_test})
                scores["subset_test"].append(res["test"])
                scores["full_test"].append(res["full_test"])
            for split, lst in scores.items():
                row = {"subset": size, "n_train": n_train, "model": mname, "eval_on": split}
                for k in ("accuracy", "macro_f1"):
                    vals = [r[k] for r in lst]
                    row[f"{k}_mean"], row[f"{k}_std"] = float(np.mean(vals)), float(np.std(vals))
                rows.append(row)
                if split == "full_test":
                    curve[mname][n_train] = (row["macro_f1_mean"], row["macro_f1_std"])
    out_root = Path(resolve(out_root))
    write_table(rows, out_root / "data_scale_results")
    from mabsa.visualize import learning_curve
    learning_curve(curve, out_root / "learning_curve.png")
    return rows


def write_table(rows, stem: Path):
    if not rows:
        return
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_json(rows, stem.with_suffix(".json"))
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # bảng Markdown (%): tiện dán vào báo cáo
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    metric_words = ("accuracy", "precision", "recall", "f1")
    for r in rows:
        cells = [f"{v * 100:.2f}%" if isinstance(v, float) and any(w in k for w in metric_words) else str(v)
                 for k, v in r.items()]
        lines.append("| " + " | ".join(cells) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
