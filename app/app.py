"""Ứng dụng minh họa MABSA (Chương 6).

Chạy từ thư mục gốc dự án:  streamlit run app/app.py   →  http://localhost:8501
Cần checkpoint: checkpoints/mabsa_best.pt  (tạo bằng: python scripts/train.py --config ... --export)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from mabsa.data.preprocess import count_words, load_image  # noqa: E402
from mabsa.visualize import SENTIMENT_COLORS, overlay_attention  # noqa: E402

CKPT = ROOT / "checkpoints" / "mabsa_best.pt"
EXTRACTOR = ROOT / "checkpoints" / "extractor_best.pt"
MODEL_CODE_FILES = (
    ROOT / "app" / "app.py",
    ROOT / "src" / "mabsa" / "predict.py",
    ROOT / "src" / "mabsa" / "models" / "mabsa.py",
    ROOT / "src" / "mabsa" / "models" / "encoders.py",
    ROOT / "src" / "mabsa" / "models" / "fusion.py",
    ROOT / "src" / "mabsa" / "builder.py",
    ROOT / "src" / "mabsa" / "engine.py",
    ROOT / "src" / "mabsa" / "utils.py",
    ROOT / "src" / "mabsa" / "data" / "dataset.py",
)


def model_code_stamp():
    """Invalidate the cached predictor when inference code changes."""
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in MODEL_CODE_FILES)

st.set_page_config(page_title="MABSA Du lịch", page_icon="🧭", layout="wide")

# Streamlit ≥ 1.50 dùng width="stretch"; bản cũ hơn dùng use_container_width=True
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}


def show_evaluation():
    """Display saved held-out results independently of the review being entered."""
    with st.expander("📊 Kết quả đánh giá mô hình trên tập test", expanded=True):
        st.caption(
            "Báo cáo test đã lưu, không phải điểm của câu đang nhập. "
            "Chấm cảm xúc trên các khía cạnh có nhãn chuẩn; chưa chấm bước nhận diện khía cạnh."
        )
        mode = st.radio("Chế độ đánh giá", ["Văn bản", "Văn bản + ảnh"],
                        horizontal=True, key="evaluation_mode")
        name = "text_only" if mode == "Văn bản" else "multimodal"
        path = ROOT / "outputs" / "retrained" / f"test_{name}_metrics.json"
        try:
            scores = json.loads(path.read_text(encoding="utf-8-sig"))
            keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
            if not all(isinstance(scores[k], (int, float)) and np.isfinite(scores[k])
                       and 0 <= scores[k] <= 1 for k in keys):
                raise ValueError("Chỉ số cần nằm trong khoảng 0–1.")
            n = scores["n"]
            if not isinstance(n, int) or n <= 0:
                raise ValueError("Số mẫu không hợp lệ.")
        except FileNotFoundError:
            st.info("Chưa có báo cáo test cho chế độ này. Cần giữ các file metrics trong outputs/retrained.")
            return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(f"Không đọc được báo cáo đánh giá: {exc}")
            return

        definitions = [
            ("Accuracy", "accuracy", "Tỷ lệ dự đoán đúng trên toàn bộ mẫu test."),
            ("Precision (Macro)", "macro_precision", "Trung bình Precision của ba nhãn, mỗi nhãn trọng số như nhau."),
            ("Recall (Macro)", "macro_recall", "Trung bình Recall của ba nhãn, mỗi nhãn trọng số như nhau."),
            ("F1 (Macro)", "macro_f1", "Trung bình F1 của ba nhãn; phù hợp để theo dõi dữ liệu lệch lớp."),
        ]
        for column, (label, key, help_text) in zip(st.columns(4), definitions):
            column.metric(label, f"{scores[key]:.2%}", help=help_text)
        st.caption(f"Số mẫu được chấm: {n:,} cặp khía cạnh–nhãn · Tích cực / Tiêu cực / Trung lập.")

        labels = ["Tích cực", "Tiêu cực", "Trung lập"]
        summary_tab, matrix_tab, aspect_tab = st.tabs(
            ["Theo nhãn cảm xúc", "Ma trận nhầm lẫn", "Theo khía cạnh"])
        with summary_tab:
            rows = scores.get("per_class", [])
            if rows:
                frame = pd.DataFrame(rows).rename(columns={
                    "label": "Nhãn", "precision": "Precision", "recall": "Recall",
                    "f1": "F1", "support": "Số mẫu"})
                st.dataframe(frame.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}",
                                                 "F1": "{:.2%}"}), hide_index=True, **FULL)
            else:
                st.info("Báo cáo chưa có kết quả riêng cho từng nhãn.")
        with matrix_tab:
            cm = np.asarray(scores.get("confusion_matrix", []))
            if (cm.shape == (3, 3) and np.issubdtype(cm.dtype, np.number)
                    and np.isfinite(cm).all() and (cm >= 0).all()
                    and (cm == np.floor(cm)).all() and cm.sum() == n):
                normalized = st.checkbox("Hiển thị tỷ lệ theo nhãn đúng", key="normalized_confusion")
                values = cm.astype(float) if normalized else cm.astype(int)
                if normalized:
                    totals = values.sum(axis=1, keepdims=True)
                    values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
                matrix = pd.DataFrame(values, index=labels, columns=labels)
                matrix.index.name = "Nhãn đúng ↓ / Dự đoán →"
                st.caption("Hàng là nhãn đúng; cột là nhãn dự đoán. Các ô trên đường chéo là dự đoán đúng.")
                cells = []
                row_totals = cm.sum(axis=1)
                for i, true_label in enumerate(labels):
                    for j, predicted_label in enumerate(labels):
                        ratio = float(cm[i, j] / row_totals[i]) if row_totals[i] else 0.0
                        cells.append({
                            "Nhãn đúng": true_label, "Dự đoán": predicted_label,
                            "Số mẫu": int(cm[i, j]), "Tỷ lệ theo nhãn đúng": ratio,
                            "Giá trị": float(values[i, j]),
                            "Nội dung": f"{ratio:.1%}" if normalized else f"{int(cm[i, j]):,}",
                        })
                maximum = 1.0 if normalized else max(1.0, float(cm.max()))
                base = alt.Chart(pd.DataFrame(cells)).encode(
                    x=alt.X("Dự đoán:N", sort=labels,
                            axis=alt.Axis(labelAngle=0, title="Nhãn dự đoán")),
                    y=alt.Y("Nhãn đúng:N", sort=labels,
                            axis=alt.Axis(title="Nhãn đúng")),
                    tooltip=[alt.Tooltip("Nhãn đúng:N"), alt.Tooltip("Dự đoán:N"),
                             alt.Tooltip("Số mẫu:Q", format=","),
                             alt.Tooltip("Tỷ lệ theo nhãn đúng:Q", format=".1%")],
                )
                heatmap = base.mark_rect(stroke="white", strokeWidth=2).encode(
                    color=alt.Color("Giá trị:Q", title="Tỷ lệ" if normalized else "Số mẫu",
                                    scale=alt.Scale(scheme="blues", domain=[0, maximum]),
                                    legend=alt.Legend(format=".0%" if normalized else ",")),
                )
                annotations = base.mark_text(fontSize=18, fontWeight="bold").encode(
                    text="Nội dung:N",
                    color=alt.condition(alt.datum["Giá trị"] > maximum * 0.5,
                                        alt.value("white"), alt.value("#17365D")),
                )
                st.altair_chart((heatmap + annotations).properties(
                    title=f"Ma trận nhầm lẫn — {mode}", height=330), use_container_width=True)
                with st.expander("Xem bảng số liệu ma trận"):
                    st.dataframe(matrix.style.format("{:.1%}" if normalized else "{:,.0f}"), **FULL)
            else:
                st.info("Báo cáo chưa có ma trận nhầm lẫn hợp lệ.")
        with aspect_tab:
            rows = scores.get("per_aspect", [])
            if rows:
                frame = pd.DataFrame(rows).rename(columns={
                    "aspect": "Khía cạnh", "n_test": "Số mẫu", "accuracy": "Accuracy",
                    "precision": "Precision (Macro)", "recall": "Recall (Macro)", "f1": "F1 (Macro)"})
                st.dataframe(frame.style.format({column: "{:.2%}" for column in
                             ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]}),
                             hide_index=True, **FULL)
            else:
                st.info("Báo cáo chưa có kết quả riêng cho từng khía cạnh.")
        st.caption("Nếu thay checkpoint hoặc train lại mô hình, cần cập nhật báo cáo test tương ứng.")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_predictor(ckpt_mtime: int, extractor_mtime: int, code_stamp: tuple):
    # Streamlit reruns the script but Python keeps imported modules in
    # sys.modules. Reload the inference dependency chain when the cache key
    # changes so a running server cannot keep using pre-edit model code.
    import importlib
    import mabsa.utils as utils_module
    import mabsa.data.dataset as dataset_module
    import mabsa.models.encoders as encoder_module
    import mabsa.models.fusion as fusion_module
    import mabsa.models.mabsa as model_module
    import mabsa.builder as builder_module
    import mabsa.engine as engine_module
    import mabsa.predict as predict_module

    importlib.reload(utils_module)
    importlib.reload(encoder_module)
    importlib.reload(fusion_module)
    importlib.reload(dataset_module)
    importlib.reload(model_module)
    importlib.reload(builder_module)
    importlib.reload(engine_module)
    importlib.reload(predict_module)
    # The trained checkpoint was validated on CPU. This also avoids backend
    # specific NaNs on Apple MPS during this app's inference path.
    return predict_module.MABSAPredictor(CKPT, EXTRACTOR, device="cpu")


def current_model_stamp():
    """Identify the exact model and inference code used for session results."""
    return (
        CKPT.stat().st_mtime_ns if CKPT.exists() else 0,
        EXTRACTOR.stat().st_mtime_ns if EXTRACTOR.exists() else 0,
        model_code_stamp(),
    )


# ----------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    if CKPT.exists():
        stamp = current_model_stamp()
        predictor = get_predictor(*stamp)
        m = predictor.cfg["model"]
        enc_t = {"phobert": "PhoBERT-base", "phobert-large": "PhoBERT-large", "xlmr": "XLM-R",
                 "videberta": "ViDeBERTa"}.get(m["text_encoder"], m["text_encoder"])
        enc_v = {"vit": "ViT-B/16", "clip": "CLIP ViT-B/16", "resnet": "ResNet-50"}.get(m["image_encoder"], m["image_encoder"])
        st.markdown(
            f"**Mô hình:** {enc_t} + {enc_v}  \n"
            f"**Căn chỉnh:** {'Contrastive Learning cấp khía cạnh' if m.get('use_contrastive', True) else 'không'}  \n"
            f"**Kết hợp:** {'Cross-Attention' if m.get('use_cross_attention', True) else 'trung bình vùng ảnh'}"
        )
    else:
        predictor = None
    threshold = st.slider("Ngưỡng hiển thị", 0.0, 1.0, 0.50, 0.05,
                          help="Lọc theo xác suất mô hình gán cho nhãn dự đoán. Điểm này chưa được hiệu chỉnh xác suất.")
    detail = st.radio("Chế độ hiển thị", ["Cơ bản", "Chi tiết"], horizontal=True)
    st.divider()
    st.caption("Nhãn cảm xúc: Tích cực · Tiêu cực · Trung lập")
    if predictor and predictor.cfg['model'].get('supported_aspects'):
        from mabsa.aspects import CODE2NAME
        st.caption('Khía cạnh đã huấn luyện: ' + ', '.join(CODE2NAME[c] for c in predictor.cfg['model']['supported_aspects']))

# ----------------------------------------------------------------- Nhập liệu
st.title("Phân tích cảm xúc đa phương thức theo khía cạnh")
st.caption("Đánh giá du lịch tiếng Việt · văn bản, có thể kèm hình ảnh")
show_evaluation()

if predictor is None:
    st.error("Chưa có checkpoint mô hình tại `checkpoints/mabsa_best.pt`.\n\n"
             "Huấn luyện và xuất checkpoint: `python scripts/train.py --config configs/default.yaml --export` "
             "(hoặc chạy thử: `python scripts/make_toy_data.py` → `python scripts/prepare_splits.py "
             "--input data/toy/dataset.jsonl --out data/toy --sizes 100,200` → "
             "`python scripts/train.py --config configs/toy.yaml --export`).")
    st.stop()

col_text, col_img = st.columns([3, 2], gap="large")
with col_text:
    review = st.text_area("Văn bản đánh giá", height=180,
                          placeholder="Nhập đánh giá về khách sạn, nhà hàng, quán cà phê hoặc điểm tham quan...")
with col_img:
    max_img = predictor.cfg["data"]["max_images"]
    files = st.file_uploader(f"Hình ảnh đính kèm (tùy chọn, tối đa {max_img} ảnh)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if files:
        st.image([load_image(f) for f in files[:max_img]], width=150)

if st.button("🔍 Phân tích", type="primary", **FULL):
    if not review.strip():
        st.warning("Vui lòng nhập văn bản đánh giá.")
        st.stop()
    if count_words(review) < 10:
        st.info("Đánh giá nên có ít nhất 10 từ để mô hình nhận diện khía cạnh tốt hơn.")
    images = [load_image(f) for f in files[:max_img]] if files else []
    with st.spinner("Đang phân tích..."):
        result = predictor.predict(review, images)
        confidences = [a.get("confidence", np.nan) for a in result.get("aspects", [])]
        if any(not np.isfinite(score) for score in confidences):
            st.session_state.pop("result", None)
            st.session_state.pop("result_stamp", None)
            details = result.get("diagnostics", {})
            st.error(f"Mô hình sinh ra điểm không hợp lệ. Thông tin kiểm tra: {details}")
            st.stop()
        st.session_state["result"] = result
        st.session_state["result_stamp"] = current_model_stamp()
        st.session_state["images"] = images

result = st.session_state.get("result")
if result and st.session_state.get("result_stamp") != current_model_stamp():
    # Streamlit preserves session_state across source/model reloads. Never show
    # old predictions under a newly loaded checkpoint or inference code.
    st.session_state.pop("result", None)
    st.session_state.pop("result_stamp", None)
    st.session_state.pop("images", None)
    result = None
    st.info("Mô hình vừa được cập nhật. Nhập đánh giá rồi bấm **Phân tích** để lấy kết quả mới.")
if result:
    unsupported = result.get('unsupported_aspects', [])
    if unsupported:
        st.info('Chưa có nhãn huấn luyện riêng cho: ' + ', '.join(a['aspect'] for a in unsupported)
                + '. Cần bổ sung dữ liệu gán nhãn cho các khía cạnh này.')
    all_aspects = result.get("aspects", [])
    valid_aspects = [a for a in all_aspects if np.isfinite(a.get("confidence", np.nan))]
    aspects = [a for a in valid_aspects if a["confidence"] >= threshold]
    st.divider()
    st.subheader("📋 Kết quả phân tích theo khía cạnh")
    if not aspects:
        if not all_aspects:
            if unsupported:
                st.info("Các khía cạnh nhận diện được trong câu này chưa có dữ liệu huấn luyện riêng.")
            else:
                st.info("Chưa nhận diện được khía cạnh trong đánh giá này.")
            st.stop()
        elif not valid_aspects:
            st.warning("Mô hình chưa trả được điểm tin cậy hợp lệ cho các khía cạnh.")
            st.stop()
        else:
            best_score = max(a["confidence"] for a in valid_aspects)
            st.info(
                f"Không có khía cạnh nào đạt ngưỡng {threshold:.0%}; "
                f"đang hiển thị các dự đoán hợp lệ (điểm cao nhất {best_score:.0%})."
            )
            # Do not hide every prediction just because the selected threshold
            # is higher than the model's current confidence.
            aspects = valid_aspects
    col_tab, col_chart = st.columns([3, 2], gap="large")
    with col_tab:
        rows = [{"Khía cạnh": a["aspect"], "Cảm xúc": a["sentiment"], "Xác suất dự đoán": f'{a["confidence"]:.0%}'}
                for a in aspects]
        if detail == "Chi tiết":
            for r, a in zip(rows, aspects):
                r["Cụm từ trong văn bản"] = a["span"] or "(ẩn)"
        styled = pd.DataFrame(rows).style.map(
            lambda v: f"color: white; background-color: {SENTIMENT_COLORS[v]}; font-weight: 600"
            if v in SENTIMENT_COLORS else "", subset=["Cảm xúc"])
        st.dataframe(styled, hide_index=True, **FULL)
        cols = st.columns(3)
        for col, lab in zip(cols, SENTIMENT_COLORS):
            col.metric(lab, sum(a["sentiment"] == lab for a in aspects))
        if result.get("extractor") == "lexicon":
            st.caption("Khía cạnh được nhận dạng bằng từ điển từ khóa (chưa có checkpoints/extractor_best.pt).")
    with col_chart:
        dist = pd.DataFrame({"Cảm xúc": list(SENTIMENT_COLORS),
                             "Số khía cạnh": [sum(a["sentiment"] == s for a in aspects) for s in SENTIMENT_COLORS]})
        chart = alt.Chart(dist[dist["Số khía cạnh"] > 0]).mark_arc(innerRadius=55).encode(
            theta="Số khía cạnh:Q",
            color=alt.Color("Cảm xúc:N", scale=alt.Scale(domain=list(SENTIMENT_COLORS),
                                                         range=list(SENTIMENT_COLORS.values()))),
            tooltip=["Cảm xúc", "Số khía cạnh"],
        ).properties(title="Phân bố cảm xúc của đánh giá", height=260)
    st.altair_chart(chart, use_container_width=True)

    if aspects[0].get("attention") is not None:
        st.divider()
        st.subheader("🔥 Trực quan hóa Attention Map")
        choice = st.selectbox("Chọn khía cạnh", [a["aspect"] for a in aspects])
        asp = next(a for a in aspects if a["aspect"] == choice)
        imgs = st.session_state["images"]
        idx = st.radio("Ảnh", list(range(1, len(imgs) + 1)), horizontal=True, format_func=lambda i: f"Ảnh {i}") - 1
        c1, c2 = st.columns(2)
        c1.image(imgs[idx], caption="Ảnh gốc", **FULL)
        c2.image(overlay_attention(imgs[idx], np.array(asp["attention"][idx])), **FULL,
                 caption=f"Attention cho khía cạnh “{asp['aspect']}” ({asp['sentiment']}, {asp['confidence']:.0%})")
        if detail == "Chi tiết":
            st.bar_chart(pd.Series(asp["probs"], name="Xác suất"), height=180)
