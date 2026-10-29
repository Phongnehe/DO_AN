import torch
from torch import nn

from mabsa.data.dataset import MABSADataset, Collator
from mabsa.models.mabsa import MABSAModel
from mabsa.predict import MABSAPredictor


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    bos_token_id = eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [3 + sum(map(ord, w)) % 20 for w in text.split()]}


def config():
    return {'data': {'max_seq_len': 32, 'image_size': 4, 'max_images': 1,
                     'aspect_representation': 'category_attention'},
            'model': {'type': 'mabsa', 'text_encoder': 'tiny', 'image_encoder': 'tiny',
                      'hidden_dim': 8, 'attn_heads': 2, 'proj_dim': 4,
                      'aspect_representation': 'category_attention', 'text_auxiliary_head': True,
                      'use_contrastive': False, 'dropout': 0.0}}


class TextEncoder(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        self.dim = 8
        self.embedding = nn.Embedding(30, 8)

    def forward(self, ids, mask):
        return self.embedding(ids)


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim, self.grid = 8, 2

    def forward(self, pixels):
        return pixels.new_zeros(len(pixels), 4, self.dim)


def dataset(rows):
    return MABSADataset(rows, Tokenizer(), lambda s: s, config()['data'], 'tiny', 2)


def model(monkeypatch):
    monkeypatch.setattr('mabsa.models.mabsa.TextEncoder', TextEncoder)
    monkeypatch.setattr('mabsa.models.mabsa.ImageEncoder', ImageEncoder)
    torch.manual_seed(3)
    return MABSAModel(config()).eval()


def rows():
    return [{'id': 'x', 'text': 'Phòng bẩn nhưng nhân viên nhiệt tình', 'images': [],
             'aspects': [{'category': 'ROOM', 'term': 'Phòng', 'start': 0, 'end': 5,
                          'sentiment': 'Tiêu cực', 'text_sentiment': 'Tiêu cực'},
                         {'category': 'SERVICE', 'term': 'nhân viên', 'start': -1, 'end': -1,
                          'sentiment': 'Tích cực', 'text_sentiment': 'Tích cực'}]}]


def test_whole_review_tokenization_independent_of_spans():
    explicit = rows()
    implicit = rows()
    for a in implicit[0]['aspects']:
        a.update(term='', start=-1, end=-1)
    assert dataset(explicit)[0]['input_ids'] == dataset(implicit)[0]['input_ids']


def test_implicit_aspects_have_different_representations(monkeypatch):
    m = model(monkeypatch)
    batch = Collator(Tokenizer(), 1, 2)([dataset(rows())[0]])
    representation = m.encode_aspects(batch)
    assert not torch.allclose(representation[0, 0], representation[0, 1])
    out = m(batch)
    assert torch.isfinite(out['logits']).all()
    assert torch.allclose(out['logits'], out['text_logits'])


def test_image_only_annotation_does_not_supervise_text_head(monkeypatch):
    m = model(monkeypatch)
    data = rows()
    for a in data[0]['aspects']:
        a['text_sentiment'] = None
    batch = Collator(Tokenizer(), 1, 2)([dataset(data)[0]])
    assert (batch['text_labels'] == -100).all()
    out = m(batch)
    loss, _ = m.compute_loss(out, batch)
    assert torch.isfinite(loss)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(m.text_classifier.weight.grad) == 0


def test_unsupported_aspect_has_no_invented_prediction():
    predictor = MABSAPredictor.__new__(MABSAPredictor)
    predictor.cfg = {'model': {'supported_aspects': ['ROOM']}}
    predictor.extractor_name = 'lexicon'
    predictor.extract_aspects = lambda text: [{'category': 'PRICE', 'term': 'giá', 'start': 0, 'end': 3}]
    result = predictor.predict('giá quá đắt', [])
    assert result['aspects'] == []
    assert result['unsupported_aspects'][0]['code'] == 'PRICE'
    assert 'confidence' not in result['unsupported_aspects'][0]


def test_cached_features_match_live_forward(monkeypatch):
    m = model(monkeypatch)
    batch = Collator(Tokenizer(), 1, 2)([dataset(rows())[0]])
    batch['image_mask'].fill_(True)
    live = m(batch)['logits']
    batch['text_features'] = m.text_encoder(batch['input_ids'], batch['attention_mask'])
    batch['image_features'] = torch.zeros(1, 1, 4, 8)
    cached = m(batch)['logits']
    assert torch.allclose(live, cached, atol=1e-6)


def test_per_aspect_macro_uses_all_three_sentiment_labels():
    from mabsa.metrics import per_aspect_report
    report = per_aspect_report([{'category': 'ROOM', 'label': 0, 'pred': 0}])
    assert abs(report[0]['f1'] - 1 / 3) < 1e-8


def test_mps_transfer_keeps_temporary_token_buffers_intact():
    import pytest
    from mabsa.engine import to_device
    if not torch.backends.mps.is_available():
        pytest.skip('MPS device not available')
    expected = torch.arange(64_000, dtype=torch.long).reshape(64, 1000)
    for _ in range(5):
        transferred = to_device({'input_ids': expected.clone()}, torch.device('mps'))
        scratch = torch.full_like(expected, 1)
        assert torch.equal(transferred['input_ids'].cpu(), expected)
        assert scratch.sum().item() == 64_000


def test_compact_checkpoint_preserves_backbone_and_updated_weights(tmp_path):
    from mabsa.utils import load_checkpoint
    base = tmp_path / 'before.pt'
    delta = tmp_path / 'after.pt'
    torch.save({'config': {}, 'state_dict': {
        'text_encoder.embedding': torch.tensor([1.0]),
        'text_encoder.model.encoder.layer.10.weight': torch.tensor([2.0]),
        'image_encoder.embedding': torch.tensor([3.0]),
        'classifier.weight': torch.tensor([99.0]),
    }}, base)
    torch.save({'format': 'mabsa_delta_v1', 'config': {},
                'backbone_checkpoint': str(base), 'backbone_filename': base.name,
                'state_dict': {'text_encoder.model.encoder.layer.10.weight': torch.tensor([7.0]),
                               'classifier.weight': torch.tensor([8.0])}}, delta)
    loaded = load_checkpoint(delta)['state_dict']
    assert loaded['text_encoder.embedding'].item() == 1
    assert loaded['image_encoder.embedding'].item() == 3
    assert loaded['text_encoder.model.encoder.layer.10.weight'].item() == 7
    assert loaded['classifier.weight'].item() == 8


def test_compact_checkpoint_reports_missing_backbone(tmp_path):
    import pytest
    from mabsa.utils import load_checkpoint
    path = tmp_path / 'after.pt'
    torch.save({'format': 'mabsa_delta_v1', 'config': {}, 'state_dict': {},
                'backbone_checkpoint': str(tmp_path / 'absent.pt'), 'backbone_filename': 'absent.pt'}, path)
    with pytest.raises(FileNotFoundError, match='retained backbone'):
        load_checkpoint(path)
