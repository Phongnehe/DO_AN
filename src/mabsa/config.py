"""Đọc cấu hình YAML, hỗ trợ kế thừa (`base:`) và ghi đè từ dòng lệnh (`--set a.b=1`)."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path, overrides=None) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(path.parent / base)
        cfg = _merge(base_cfg, cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def resolve(path: str) -> Path:
    """Đường dẫn tương đối trong cấu hình được hiểu là tính từ thư mục gốc dự án."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def save_config(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
