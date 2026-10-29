"""Kiểm tra trùng lặp (mục 3.1.4.6): TF-IDF cosine > 0.9 cho văn bản và perceptual hash cho ảnh."""
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_text_duplicates(texts, threshold: float = 0.9, chunk: int = 2000):
    """Trả về tập chỉ số cần loại: với mỗi cặp trùng, giữ mẫu xuất hiện trước."""
    if len(texts) < 2:
        return set()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    drop = set()
    for start in range(0, X.shape[0], chunk):
        sims = cosine_similarity(X[start:start + chunk], X)
        for r in range(sims.shape[0]):
            i = start + r
            if i in drop:
                continue
            dup = np.where(sims[r, i + 1:] > threshold)[0] + i + 1
            drop.update(int(j) for j in dup)
    return drop


def image_hash(path):
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def find_image_duplicates(image_lists, root=".", max_distance: int = 4):
    """image_lists: danh sách (theo mẫu) các đường dẫn ảnh. Mẫu có ảnh trùng mẫu trước bị loại."""
    seen, drop = [], set()
    for i, paths in enumerate(image_lists):
        hashes = []
        for p in paths:
            try:
                hashes.append(image_hash(Path(root) / p))
            except Exception:
                continue
        if any(h - s <= max_distance for h in hashes for s in seen):
            drop.add(i)
            continue
        seen.extend(hashes)
    return drop
