"""Chạy các nhóm thực nghiệm của Chương 5.

    python scripts/run_experiments.py baselines   # Bảng 5.3 – so sánh các phương pháp
    python scripts/run_experiments.py ablation    # Bảng 5.5 – ablation study
    python scripts/run_experiments.py data_scale  # mục 5.1.4 – tập con 2k..10k × 3 seed
Thêm --config configs/toy.yaml để chạy thử; --only "Tên 1" "Tên 2" để chạy một phần.
Kết quả: outputs/<nhóm>/results.{csv,md,json} (+ learning_curve.png cho data_scale).
"""
import _bootstrap  # noqa: F401
import argparse

from mabsa.config import resolve
from mabsa.experiments import ABLATIONS, BASELINES, run_data_scale, run_group
from mabsa.visualize import bar_compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["baselines", "ablation", "data_scale"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subsets-dir", default="data/processed/subsets")
    ap.add_argument("--full-test", default="data/processed/splits/test.jsonl")
    ap.add_argument("--sizes", nargs="*", default=["D2k", "D4k", "D6k", "D8k", "D10k"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    out = args.out or f"outputs/{args.group}"
    if args.group in ("baselines", "ablation"):
        variants = BASELINES if args.group == "baselines" else ABLATIONS
        rows = run_group(args.config, variants, out, args.set, args.only)
        bar_compare([r["model"] for r in rows], [r["macro_f1"] for r in rows], resolve(out) / "macro_f1.png",
                    title="Bảng 5.3 – So sánh các mô hình" if args.group == "baselines" else "Bảng 5.5 – Ablation study")
        base = rows[0]["macro_f1"] if args.group == "ablation" and rows else None
        print(f"\n{'Mô hình':<45} {'Accuracy':>9} {'Macro F1':>9} {'Micro F1':>9}" + ("   Δ Macro F1" if base else ""))
        for r in rows:
            delta = f"   {100 * (r['macro_f1'] - base):+.2f}%" if base is not None else ""
            print(f"{r['model']:<45} {r['accuracy']:>9.2%} {r['macro_f1']:>9.2%} {r['micro_f1']:>9.2%}{delta}")
    else:
        rows = run_data_scale(args.config, args.subsets_dir, args.full_test, out, args.seeds, args.sizes,
                              extra=args.set)
        for r in rows:
            print(f"{r['subset']:<5} {r['model']:<42} {r['eval_on']:<11} "
                  f"MacroF1 {r['macro_f1_mean']:.2%} ± {r['macro_f1_std']:.2%}")
    print(f"\nKết quả: {resolve(out)}")


if __name__ == "__main__":
    main()
