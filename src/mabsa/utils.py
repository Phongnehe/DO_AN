import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def get_logger(name: str = "mabsa", log_file=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
            logger.removeHandler(h)
            h.close()
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_checkpoint(path, _seen=None):
    """Load a full checkpoint or a compact update with a retained backbone.

    A delta carries all new heads and the final two updated PhoBERT layers.
    The original frozen backbone must remain alongside the app checkpoint.
    """
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f'Circular backbone checkpoint reference: {path}')
    seen.add(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('format') != 'mabsa_delta_v1':
        checkpoint['_backbone_path'] = str(path)
        return checkpoint
    local_base = path.parent / checkpoint['backbone_filename']
    base_path = local_base if local_base.exists() else Path(checkpoint['backbone_checkpoint'])
    if not base_path.exists():
        raise FileNotFoundError(f'This compact checkpoint needs its retained backbone: {base_path}')
    base = load_checkpoint(base_path, seen)
    state = {name: value for name, value in base['state_dict'].items()
             if name.startswith(('text_encoder.', 'image_encoder.'))}
    state.update(checkpoint['state_dict'])
    checkpoint['state_dict'] = state
    checkpoint['_backbone_path'] = base['_backbone_path']
    return checkpoint


def write_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
