"""Chạy thử TOÀN BỘ pipeline trên dữ liệu giả (CPU, ~1–2 phút) để kiểm tra cài đặt môi trường.

    python scripts/smoke_test.py
Không tải mô hình tiền huấn luyện; kết quả số KHÔNG có ý nghĩa, chỉ để kiểm tra code chạy đúng.
"""
import _bootstrap  # noqa: F401
import subprocess
import sys

from mabsa.config import ROOT

STEPS = [
    ("Sinh dữ liệu giả", ["scripts/make_toy_data.py", "--n", "300"]),
    ("Làm sạch dữ liệu thô", ["scripts/build_dataset.py", "--input", "data/toy/raw/reviews.csv",
                              "--output", "data/toy/clean.jsonl"]),
    ("Gợi ý nhãn khía cạnh", ["scripts/pre_annotate.py", "--input", "data/toy/clean.jsonl",
                              "--output", "data/toy/draft.jsonl"]),
    ("Kappa (so file với chính nó)", ["scripts/compute_kappa.py", "data/toy/dataset.jsonl", "data/toy/dataset.jsonl",
                                      "--out", "outputs/toy_stats/kappa"]),
    ("Chia 80:10:10 + tập con", ["scripts/prepare_splits.py", "--input", "data/toy/dataset.jsonl", "--out", "data/toy",
                                 "--sizes", "100,200,300", "--stats-dir", "outputs/toy_stats"]),
    ("Huấn luyện mô hình đề xuất", ["scripts/train.py", "--config", "configs/toy.yaml", "--set", "train.epochs=2"]),
    ("Huấn luyện bộ nhận dạng khía cạnh", ["scripts/train_extractor.py", "--config", "configs/toy.yaml",
                                           "--epochs", "2", "--lr", "1e-3", "--output", "outputs/toy/extractor.pt"]),
    ("Đánh giá", ["scripts/evaluate.py", "--checkpoint", "outputs/toy/best.pt", "--data", "data/toy/splits/test.jsonl"]),
    ("Dự đoán 1 câu", ["scripts/predict.py", "--checkpoint", "outputs/toy/best.pt", "--extractor",
                       "outputs/toy/extractor.pt", "--text", "Cà phê rất ngon nhưng giá hơi cao so với mặt bằng chung.",
                       "--images", "data/toy/images/TOY_00000_1.jpg"]),
    ("Baseline (2 mô hình)", ["scripts/run_experiments.py", "baselines", "--config", "configs/toy.yaml",
                              "--set", "train.epochs=1", "--out", "outputs/toy_baselines",
                              "--only", "PhoBERT (chỉ văn bản)", "Đề xuất (Transformer + Contrastive)"]),
]


def main():
    for i, (name, cmd) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {name}\n" + "-" * 60, flush=True)
        r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"\n✗ Lỗi ở bước: {name}")
            sys.exit(r.returncode)
    print("\n✓ Pipeline chạy thành công. Môi trường đã sẵn sàng cho dữ liệu thật (configs/default.yaml).")


if __name__ == "__main__":
    main()
