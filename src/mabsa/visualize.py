"""Trực quan hóa: attention map (mục 5.3.5, 6.2.4), biểu đồ kết quả và đường cong học (mục 5.1.4)."""
import numpy as np
from PIL import Image

SENTIMENT_COLORS = {"Tích cực": "#2E7D32", "Tiêu cực": "#C62828", "Trung lập": "#F9A825"}


def overlay_attention(image: Image.Image, attention, alpha: float = 0.45) -> Image.Image:
    """Nội suy lưới attention (vd. 14×14) về kích thước ảnh và phủ heatmap (xanh → đỏ)."""
    from matplotlib import colormaps
    attn = np.asarray(attention, dtype=np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heat = Image.fromarray(np.uint8(attn * 255)).resize(image.size, Image.BICUBIC)
    colored = Image.fromarray(np.uint8(colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255))
    return Image.blend(image.convert("RGB"), colored, alpha)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bar_compare(names, values, path, ylabel="Macro F1 (%)", title=""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bars = ax.bar(range(len(names)), [v * 100 for v in values], color="#1565C0")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v * 100:.2f}", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def learning_curve(table, path, metric="macro_f1", title="Đường cong học theo quy mô dữ liệu"):
    """table: {model_name: {n_train: (mean, std)}}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for name, pts in table.items():
        xs = sorted(pts)
        m = np.array([pts[x][0] for x in xs]) * 100
        s = np.array([pts[x][1] for x in xs]) * 100
        ax.plot(xs, m, marker="o", label=name)
        ax.fill_between(xs, m - s, m + s, alpha=0.15)
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
