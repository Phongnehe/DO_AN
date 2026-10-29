"""Chia dữ liệu 80:10:10 phân tầng và tạo các tập con lồng nhau 2k/4k/6k/8k/10k (mục 3.4.5)."""
from collections import Counter, defaultdict

import numpy as np

from mabsa.aspects import SENTIMENTS


def strata_key(sample) -> str:
    """Khóa phân tầng: cảm xúc chiếm ưu thế + khía cạnh đầu tiên của mẫu."""
    sents = [a["sentiment"] for a in sample.get("aspects", [])]
    dom = Counter(sents).most_common(1)[0][0] if sents else "none"
    first = sample["aspects"][0]["category"] if sample.get("aspects") else "none"
    return f"{dom}|{first}"


def _stratified_order(samples, rng, min_group: int = 5):
    """Sắp thứ tự mẫu sao cho MỌI tiền tố của danh sách đều giữ tỷ lệ các tầng.

    Mỗi mẫu thứ k trong tầng có kích thước n nhận hạng (k + u)/n, u ~ U(0,1); sắp theo hạng.
    Nhờ đó lấy n phần tử đầu tiên cho ra một mẫu phân tầng, và các tập con lấy theo tiền tố
    luôn lồng nhau (D2k ⊂ D4k ⊂ ... ⊂ D10k).
    """
    keys = [strata_key(s) for s in samples]
    cnt = Counter(keys)
    keys = [k if cnt[k] >= min_group else k.split("|")[0] for k in keys]
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    ranks = np.empty(len(samples))
    for idx in groups.values():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        for k, i in enumerate(idx):
            ranks[i] = (k + rng.random()) / n
    return list(np.argsort(ranks, kind="stable"))


def stratified_split(samples, ratios=(0.8, 0.1, 0.1), seed: int = 42):
    rng = np.random.default_rng(seed)
    order = _stratified_order(samples, rng)
    n = len(samples)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # xen kẽ để cả 3 tập cùng giữ tỷ lệ tầng: gán theo vị trí trong chuỗi phân tầng
    pattern = (["train"] * int(ratios[0] * 10) + ["val"] * int(ratios[1] * 10) + ["test"] * int(ratios[2] * 10))
    buckets = {"train": [], "val": [], "test": []}
    for pos, i in enumerate(order):
        buckets[pattern[pos % len(pattern)]].append(samples[i])
    # hiệu chỉnh kích thước chính xác
    target = {"train": n_train, "val": n_val, "test": n - n_train - n_val}
    for src in ("train", "val", "test"):
        for dst in ("train", "val", "test"):
            while len(buckets[src]) > target[src] and len(buckets[dst]) < target[dst]:
                buckets[dst].append(buckets[src].pop())
    return buckets


def nested_subsets(splits, sizes=(2000, 4000, 6000, 8000, 10000), ratios=(0.8, 0.1, 0.1), seed: int = 42):
    """Tạo tập con lồng nhau; phần train/val/test của tập con lấy từ train/val/test của tập đầy đủ."""
    rng = np.random.default_rng(seed + 1)
    orders = {name: _stratified_order(rows, rng) for name, rows in splits.items()}
    subsets = {}
    for size in sizes:
        want = {"train": int(round(size * ratios[0])), "val": int(round(size * ratios[1]))}
        want["test"] = size - want["train"] - want["val"]
        sub = {}
        for name in ("train", "val", "test"):
            k = min(want[name], len(splits[name]))
            sub[name] = [splits[name][i] for i in orders[name][:k]]
        subsets[f"D{size // 1000}k" if size % 1000 == 0 else f"D{size}"] = sub
    return subsets


def label_distribution(rows):
    c = Counter(a["sentiment"] for r in rows for a in r.get("aspects", []))
    total = sum(c.values()) or 1
    return {s: round(c.get(s, 0) / total, 4) for s in SENTIMENTS}
