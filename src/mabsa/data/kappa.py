"""Độ đồng thuận giữa người gán nhãn (mục 3.3.5, Bảng 3.4): Cohen's Kappa và Fleiss' Kappa."""
from collections import Counter

import numpy as np


def cohen_kappa(a, b) -> float:
    assert len(a) == len(b) and len(a) > 0
    labels = sorted(set(a) | set(b), key=str)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / (n * n)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def fleiss_kappa(ratings) -> float:
    """ratings: danh sách theo mục, mỗi mục là danh sách nhãn của các người gán nhãn (≥ 3 người)."""
    labels = sorted({l for r in ratings for l in r}, key=str)
    n_items, n_raters = len(ratings), len(ratings[0])
    mat = np.zeros((n_items, len(labels)))
    for i, r in enumerate(ratings):
        c = Counter(r)
        for j, l in enumerate(labels):
            mat[i, j] = c.get(l, 0)
    p_j = mat.sum(0) / (n_items * n_raters)
    P_i = ((mat ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar, Pe = P_i.mean(), (p_j ** 2).sum()
    return 1.0 if Pe == 1 else float((P_bar - Pe) / (1 - Pe))


def interpret(k: float) -> str:
    """Thang Landis & Koch (1977)."""
    if k >= 0.81:
        return "Rất tốt"
    if k >= 0.61:
        return "Tốt"
    if k >= 0.41:
        return "Trung bình"
    if k >= 0.21:
        return "Yếu"
    return "Kém"


def kappa(label_lists) -> float:
    """label_lists: danh sách theo người gán nhãn (cùng thứ tự mục). 2 người → Cohen, ≥3 → Fleiss."""
    if len(label_lists) == 2:
        return cohen_kappa(*label_lists)
    return fleiss_kappa([list(x) for x in zip(*label_lists)])
