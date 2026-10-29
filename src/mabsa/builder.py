"""Khởi tạo các thành phần dùng chung: tokenizer, bộ tách từ, dataset, dataloader, mô hình."""
import torch
from torch.utils.data import DataLoader

from mabsa.config import resolve
from mabsa.data.dataset import Collator, MABSADataset, class_weights, oversampling_sampler
from mabsa.data.preprocess import WordSegmenter
from mabsa.models.encoders import build_tokenizer, grid_size
from mabsa.models.mabsa import MABSAModel
from mabsa.utils import read_jsonl

_SEGMENTERS = {}


def get_segmenter(cfg) -> WordSegmenter:
    backend = cfg["data"].get("segmenter", "vncorenlp")
    if cfg["model"]["text_encoder"] == "tiny":
        backend = "none"
    if backend not in _SEGMENTERS:
        _SEGMENTERS[backend] = WordSegmenter(backend, str(resolve(cfg["data"].get("vncorenlp_dir", "vncorenlp"))))
    return _SEGMENTERS[backend]


def model_grid(cfg) -> int:
    if cfg['model'].get('image_pool_grid'):
        return int(cfg['model']['image_pool_grid'])
    enc_cfg = {**cfg["model"], "image_size": cfg["data"]["image_size"]}
    return grid_size(enc_cfg)


def build_tokenizer_from_cfg(cfg):
    return build_tokenizer(cfg["model"])


def build_dataset(rows, cfg, tokenizer, with_labels=True):
    data_cfg = {**cfg['data'], 'aspect_representation': cfg['model'].get('aspect_representation', 'span')}
    return MABSADataset(rows, tokenizer, get_segmenter(cfg), data_cfg,
                        image_encoder=cfg["model"]["image_encoder"], grid=model_grid(cfg), with_labels=with_labels)


def build_loader(rows, cfg, tokenizer, train=False, with_labels=True):
    ds = build_dataset(rows, cfg, tokenizer, with_labels)
    collate = Collator(tokenizer, cfg["data"]["max_images"], model_grid(cfg))
    t = cfg["train"]
    sampler = oversampling_sampler(rows) if (train and t.get("imbalance") == "oversample") else None
    return DataLoader(ds, batch_size=t["batch_size"] if train else t.get("eval_batch_size", t["batch_size"]),
                      shuffle=train and sampler is None, sampler=sampler, collate_fn=collate,
                      num_workers=t.get("num_workers", 0), pin_memory=torch.cuda.is_available())


def load_split(cfg, name):
    return read_jsonl(resolve(cfg["data"][name]))


def build_model(cfg, tokenizer, train_rows=None):
    model = MABSAModel(cfg, vocab_size=len(tokenizer))
    if train_rows is not None and cfg["train"].get("imbalance") == "weighted_ce":
        model.class_weight = class_weights(train_rows)
    return model
