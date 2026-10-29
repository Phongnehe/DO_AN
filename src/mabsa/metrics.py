"""Độ đo đánh giá (mục 2.2.4, 5.1.3): Accuracy, Precision, Recall, Macro/Micro F1, kết quả theo khía cạnh."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from mabsa.aspects import ASPECT_CODES, CODE2NAME, SENTIMENTS

LABELS = list(range(len(SENTIMENTS)))


def compute_metrics(y_true, y_pred) -> dict:
    if len(y_true) == 0:
        return {}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "micro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0)),
        "n": int(len(y_true)),
    }


def per_class_report(y_true, y_pred) -> list:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return [{"label": SENTIMENTS[k], "precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
             "support": int(s[k])} for k in LABELS]


def per_aspect_report(records) -> list:
    """Bảng 5.4: số mẫu test, Precision, Recall, F1 (macro theo 3 nhãn cảm xúc) cho từng khía cạnh."""
    rows = []
    for code in ASPECT_CODES:
        sub = [r for r in records if r["category"] == code]
        if not sub:
            continue
        yt = [r["label"] for r in sub]
        yp = [r["pred"] for r in sub]
        p, rc, f, _ = precision_recall_fscore_support(yt, yp, labels=LABELS, average="macro", zero_division=0)
        rows.append({"aspect": CODE2NAME[code], "n_test": len(sub), "precision": float(p), "recall": float(rc),
                     "f1": float(f), "accuracy": float(np.mean(np.array(yt) == np.array(yp)))})
    return rows


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABELS).tolist()


def plot_confusion(cm, path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(LABELS, SENTIMENTS)
    ax.set_yticks(LABELS, SENTIMENTS)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn đúng")
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
