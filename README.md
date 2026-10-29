# MABSA-Travel – Phân tích cảm xúc đa phương thức theo khía cạnh cho đánh giá du lịch tiếng Việt

Mã nguồn đi kèm báo cáo: **PhoBERT + ViT/CLIP + Contrastive Learning cấp khía cạnh + Cross-Attention**,
gồm toàn bộ quy trình từ xây dựng dữ liệu (Chương 3), mô hình (Chương 4), thực nghiệm (Chương 5)
đến ứng dụng minh họa Streamlit (Chương 6). Dự án được cấu hình sẵn để chạy và debug trong **Visual Studio Code**.

---

## 1. Cài đặt trên VS Code

**Yêu cầu:** Python 3.9–3.11 (khuyến nghị 3.10), Git, Java 8+ (cho VnCoreNLP), GPU NVIDIA ≥ 12 GB nếu huấn luyện mô hình thật.

1. Giải nén thư mục `mabsa_travel`, mở VS Code → **File › Open Folder…** → chọn `mabsa_travel`.
   Cài các extension được gợi ý (Python, Pylance, Python Debugger).
2. Mở Terminal (**Ctrl + `**) và tạo môi trường ảo:
   ```powershell
   py -3.10 -m venv .venv            # macOS/Linux: python3 -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   ```
   Nếu PowerShell báo lỗi quyền chạy script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. (Có GPU) cài PyTorch bản CUDA **trước**, ví dụ CUDA 12.1:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
4. Cài thư viện và cài dự án ở chế độ phát triển:
   ```powershell
   pip install -r requirements.txt
   pip install -e .
   ```
5. **Ctrl+Shift+P → Python: Select Interpreter** → chọn `.venv`.
   (File `.vscode/settings.json` đặt sẵn đường dẫn Windows `.venv/Scripts/python.exe`.)
6. Kiểm tra môi trường: tab **Run and Debug** (Ctrl+Shift+D) → chọn **“0. Chạy thử toàn bộ (dữ liệu giả, CPU)”** → F5,
   hoặc `python scripts/smoke_test.py`. Lệnh này sinh dữ liệu giả và chạy toàn bộ pipeline trong ~1 phút
   với mô hình tí hon, **không tải mô hình tiền huấn luyện** – số liệu in ra không có ý nghĩa.

Tất cả các bước bên dưới đều có sẵn cấu hình chạy/debug trong `.vscode/launch.json` (đánh số 1 → 8).

## 2. Quy trình với dữ liệu thật

| Bước | Lệnh | Tương ứng báo cáo |
|---|---|---|
| 1. Làm sạch dữ liệu thô | `python scripts/build_dataset.py --input data/raw/reviews.csv` | 3.1.3, 3.1.4, 3.2 (NFC, ≥ 10 từ, ảnh ≥ 224×224, loại spam, trùng lặp TF-IDF > 0.9 + perceptual hash) |
| 2. Gợi ý nhãn khía cạnh | `python scripts/pre_annotate.py` | 3.3.4 gán nhãn bán tự động |
| 3. Đo độ đồng thuận | `python scripts/compute_kappa.py A.jsonl B.jsonl` | 3.3.5, Bảng 3.4 (Kappa ≥ 0.7) |
| 4. Chia dữ liệu + thống kê | `python scripts/prepare_splits.py --input data/annotations/final.jsonl` | 3.4, Bảng 3.5–3.9, Hình 3.4 |
| 5. Huấn luyện mô hình đề xuất | `python scripts/train.py --config configs/default.yaml --export` | Chương 4, Bảng 5.1 |
| 5b. Bộ nhận dạng khía cạnh | `python scripts/train_extractor.py` | 4.4.1, 6.2.3 |
| 6. Đánh giá | `python scripts/evaluate.py --checkpoint checkpoints/mabsa_best.pt` | 5.1.3, Bảng 5.4 |
| 7a. So sánh baseline | `python scripts/run_experiments.py baselines` | Bảng 5.2, 5.3 |
| 7b. Ablation study | `python scripts/run_experiments.py ablation` | Bảng 5.5 |
| 7c. Quy mô dữ liệu | `python scripts/run_experiments.py data_scale` | 5.1.4 (2k→10k × 3 seed, đường cong học) |
| 8. Ứng dụng minh họa | `streamlit run app/app.py` → http://localhost:8501 | Chương 6 |

Định dạng file dữ liệu: xem **`data/README.md`**.

Kết quả mỗi lần huấn luyện nằm trong `outputs/<tên>/`: `best.pt`, `config.yaml`, `train.log`, `history.json`,
`test_metrics.json` (Accuracy, Precision, Recall, Macro/Micro F1, theo lớp, confusion matrix),
`test_per_aspect.csv` (Bảng 5.4), `test_confusion_matrix.png`, `test_predictions.jsonl`.
Các nhóm thực nghiệm xuất `results.csv / results.md` (bảng dán thẳng vào báo cáo) và biểu đồ.

## 3. Cấu hình

`configs/default.yaml` chứa đúng siêu tham số của Bảng 5.1 (Adam, lr 2e-5, warm-up 10%, batch 16, 20 epoch,
early stopping 3, τ = 0.07, α = 0.7, β = 0.3, max length 256, ảnh 224×224). Ghi đè nhanh từ dòng lệnh:

```powershell
python scripts/train.py --config configs/default.yaml --set model.image_encoder=clip train.batch_size=8
python scripts/train.py --set model.type=text_only                 # baseline PhoBERT chỉ văn bản
python scripts/train.py --set model.use_contrastive=false          # cross-attention không contrastive
python scripts/train.py --set train.imbalance=oversample           # xử lý mất cân bằng bằng oversampling
```

| Khóa | Giá trị |
|---|---|
| `model.type` | `mabsa` (đề xuất) · `text_only` · `image_only` · `early_fusion` · `late_fusion` |
| `model.text_encoder` | `phobert` · `phobert-large` · `xlmr` · `videberta` |
| `model.image_encoder` | `vit` · `clip` · `resnet` |
| `model.fusion` | `gated` · `concat` |
| `train.imbalance` | `weighted_ce` · `oversample` · `none` |

**Thiếu bộ nhớ GPU (CUDA out of memory):** giảm `train.batch_size` (8) và tăng `train.grad_accum_steps` (2),
giảm `data.max_images` (3), hoặc đặt `model.freeze_image_encoder=true`.

## 4. Cấu trúc mã nguồn

```
mabsa_travel/
├── .vscode/                 settings, launch (chạy/debug từng bước), tasks
├── configs/                 default.yaml (Bảng 5.1), toy.yaml (chạy thử)
├── app/app.py               ứng dụng Streamlit (Chương 6)
├── scripts/                 các bước 1 → 8 ở trên + make_toy_data.py, smoke_test.py, predict.py
├── src/mabsa/
│   ├── aspects.py           10 khía cạnh (Bảng 3.3), 3 nhãn cảm xúc, từ điển từ khóa
│   ├── data/
│   │   ├── preprocess.py    chuẩn hóa văn bản, tách từ VnCoreNLP/underthesea, xử lý ảnh (3.2)
│   │   ├── dedup.py         trùng lặp TF-IDF cosine + perceptual hash (3.1.4.6)
│   │   ├── kappa.py         Cohen's / Fleiss' Kappa (3.3.5)
│   │   ├── split.py         chia phân tầng 80:10:10, tập con lồng nhau (3.4.5)
│   │   ├── stats.py         thống kê mô tả (3.4)
│   │   └── dataset.py       Dataset/Collator: vị trí khía cạnh → token, bbox → vùng ảnh dương
│   ├── models/
│   │   ├── encoders.py      PhoBERT/XLM-R/ViDeBERTa, ViT/CLIP/ResNet (4.3)
│   │   ├── contrastive.py   InfoNCE cấp khía cạnh – Thuật toán 5.1 (4.5)
│   │   ├── fusion.py        Cross-Attention + Gated fusion (4.6)
│   │   ├── mabsa.py         mô hình đề xuất + các baseline, hàm mất mát tổng hợp (4.7, 4.8, Thuật toán 5.2)
│   │   └── extractor.py     nhận dạng cụm từ khía cạnh BIO + từ điển (4.4.1)
│   ├── engine.py            vòng huấn luyện/đánh giá (4.9, 5.1)
│   ├── metrics.py           Accuracy, P, R, Macro/Micro F1, theo khía cạnh (5.1.3)
│   ├── experiments.py       baseline, ablation, quy mô dữ liệu (5.2, 5.3, 5.1.4)
│   ├── predict.py           suy luận cho ứng dụng (6.2.3)
│   └── visualize.py         attention map, biểu đồ, đường cong học
├── data/                    README.md (định dạng), raw/, annotations/, processed/
├── checkpoints/             mabsa_best.pt, extractor_best.pt (ứng dụng đọc từ đây)
└── tests/                   pytest
```

## 5. Ghi chú

- Lần chạy đầu, HuggingFace tự tải `vinai/phobert-base`, `google/vit-base-patch16-224-in21k`
  (hoặc `openai/clip-vit-base-patch16`); VnCoreNLP được tải vào thư mục `vncorenlp/` (cần Java).
  Nếu không có Java, chương trình tự chuyển sang `underthesea` và in cảnh báo.
- XLM-R và ViDeBERTa không dùng văn bản đã tách từ: thực nghiệm ablation tự đặt `data.segmenter=none`.
- VLP-MABSA (Ling và cộng sự, 2022) dùng mã nguồn chính thức của tác giả; kết quả điền tay vào Bảng 5.3.
- Kết quả trong báo cáo cần được thay bằng số liệu thực khi chạy các lệnh ở mục 2 trên bộ dữ liệu thật.


## Dataset – Nguồn dữ liệu

### MACSAA Dataset

Dự án sử dụng bộ dữ liệu MACSAA làm nguồn dữ liệu phục vụ quá trình xây dựng, huấn luyện và thực nghiệm mô hình phân tích cảm xúc đa phương thức theo khía cạnh (Multimodal Aspect-Based Sentiment Analysis – MABSA).

- **Dataset name:** MACSAA
- **Original source:** [MACSAA Dataset on Kaggle](https://www.kaggle.com/datasets/nguynquy/macsaa)
- **Dataset download:** [Download MACSAA from Google Drive](https://drive.google.com/file/d/1qbChChzBHYLk61_bBYFKmAJFLbO4FIsP/view?usp=sharing)
- **Purpose:** Data preparation, model training and experimentation for the MABSA-Travel project.
- **Preprocessing:** The data should be checked, cleaned, annotated where necessary, and split into training, validation and test sets before model training, following the workflow described in Chapter 3.
- **Storage policy:** The dataset files are hosted externally and are not included in this GitHub repository. Download the data from the links above and follow the applicable dataset license and sharing terms.

**Reproducibility note:** Experimental results reported in the project should be based on actual runs using the processed dataset. The presence of a dataset link does not by itself establish that training has been completed or that reported metrics have been reproduced.
