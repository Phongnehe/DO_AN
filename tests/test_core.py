import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mabsa.data.dataset import bbox_to_cells, locate_aspect  # noqa: E402
from mabsa.data.kappa import cohen_kappa, fleiss_kappa  # noqa: E402
from mabsa.data.preprocess import count_words, normalize_text  # noqa: E402
from mabsa.data.split import nested_subsets, stratified_split  # noqa: E402
from mabsa.models.contrastive import AspectContrastiveLoss  # noqa: E402
from mabsa.models.extractor import LexiconAspectExtractor  # noqa: E402


def _rows(n):
    sents = ["Tích cực", "Tiêu cực", "Trung lập"]
    return [{"id": str(i), "text": "x", "aspects": [{"category": "FOOD", "sentiment": sents[i % 3]}]} for i in range(n)]


def test_normalize_text():
    t = normalize_text("Ngonnnnn quá!!!  😍 http://abc.com ★★★")
    assert "http" not in t and "!!" not in t and "😍" in t and "★" not in t
    assert count_words("Món ăn ở đây rất ngon") == 6


def test_split_ratio_and_nested():
    rows = _rows(1000)
    sp = stratified_split(rows, (0.8, 0.1, 0.1), 0)
    assert [len(sp[k]) for k in ("train", "val", "test")] == [800, 100, 100]
    ids = [set(r["id"] for r in sp[k]) for k in sp]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    sub = nested_subsets(sp, (200, 400), (0.8, 0.1, 0.1), 0)
    for part in ("train", "val", "test"):
        small = {r["id"] for r in sub["D200"][part]}
        big = {r["id"] for r in sub["D400"][part]}
        assert small <= big


def test_kappa():
    assert abs(cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) - 1.0) < 1e-9
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0], [1, 1, 1]]) > 0.99


def test_locate_and_bbox():
    text = "Món ăn ở đây rất ngon"
    assert locate_aspect(text, {"term": "món ăn", "start": 5, "end": 9}) == (0, 6)
    cells = bbox_to_cells([0.0, 0.0, 0.5, 0.5], 4)
    assert set(cells) == {0, 1, 4, 5}


def test_contrastive_prefers_positive():
    torch.manual_seed(0)
    loss = AspectContrastiveLoss(8, 8, tau=0.1)
    a = torch.randn(2, 1, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    pos = torch.zeros(2, 1, 4, dtype=torch.bool)
    pos[0, 0, 1] = pos[1, 0, 2] = True
    val = loss(a, torch.ones(2, 1, dtype=torch.bool), v, mask, pos, pos.any(-1))
    val.backward()
    assert torch.isfinite(val) and val.item() > 0


def test_lexicon_extractor():
    spans = LexiconAspectExtractor().extract("Nhân viên nhiệt tình, giá hơi cao")
    assert {s["category"] for s in spans} == {"SERVICE", "PRICE"}
