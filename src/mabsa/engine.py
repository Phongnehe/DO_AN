"""Huấn luyện (mục 4.9, 5.1.2) và đánh giá (mục 5.1.3).

Adam, lr 2e-5, warm-up tuyến tính 10% số bước, batch 16, tối đa 20 epoch,
early stopping theo validation loss (patience 3), τ = 0.07, α = 0.7, β = 0.3.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from mabsa import metrics as M
from mabsa.builder import build_loader, build_model, build_tokenizer_from_cfg, load_split
from mabsa.config import resolve, save_config
from mabsa.utils import count_parameters, get_device, get_logger, set_seed, write_json, write_jsonl

log = get_logger()


def to_device(batch, device):
    # MPS asynchronous copies can outlive temporary CPU collator buffers.
    # Keep MPS copies synchronous; CUDA's pinned loader buffers can overlap.
    non_blocking = torch.device(device).type == 'cuda'
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v) for k, v in batch.items()}


def linear_warmup(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def build_optimizer(model, t):
    enc, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith(("text_encoder.", "image_encoder.")) else head).append(p)
    groups = [{"params": enc, "lr": t["lr"]}, {"params": head, "lr": t.get("head_lr", t["lr"])}]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    records, losses = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = to_device(batch, device)
        out = model(batch)
        logits, labels = out["logits"], batch["labels"]
        if 'text_logits' in out and 'text_labels' in batch:
            labels = torch.where(batch['image_mask'].any(-1)[:, None], labels, batch['text_labels'])
        if (labels != -100).any():
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                          ignore_index=-100).item())
        probs = logits.softmax(-1).cpu().numpy()
        preds = probs.argmax(-1)
        mask = batch["aspect_mask"].cpu().numpy()
        cats, labs = batch["categories"].cpu().numpy(), labels.cpu().numpy()
        for i, sid in enumerate(batch["ids"]):
            for j in np.nonzero(mask[i])[0]:
                if labs[i, j] == -100:
                    continue
                from mabsa.aspects import ASPECT_CODES
                records.append({"id": sid, "aspect_index": int(j), "category": ASPECT_CODES[cats[i, j]],
                                "label": int(labs[i, j]), "pred": int(preds[i, j]),
                                "probs": [round(float(x), 4) for x in probs[i, j]]})
    y_true = [r["label"] for r in records]
    y_pred = [r["pred"] for r in records]
    res = M.compute_metrics(y_true, y_pred)
    res["loss"] = float(np.mean(losses)) if losses else math.nan
    return res, records


def save_eval(res, records, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [r["label"] for r in records]
    y_pred = [r["pred"] for r in records]
    full = {**res, "per_class": M.per_class_report(y_true, y_pred), "per_aspect": M.per_aspect_report(records),
            "confusion_matrix": M.confusion(y_true, y_pred)}
    write_json(full, out_dir / f"{prefix}_metrics.json")
    write_jsonl(records, out_dir / f"{prefix}_predictions.jsonl")
    M.plot_confusion(full["confusion_matrix"], out_dir / f"{prefix}_confusion_matrix.png")
    import csv
    with open(out_dir / f"{prefix}_per_aspect.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["aspect", "n_test", "precision", "recall", "f1", "accuracy"])
        w.writeheader()
        w.writerows(full["per_aspect"])
    return full


def train(cfg: dict, output_dir, extra_eval=None) -> dict:
    """Huấn luyện theo cfg, lưu checkpoint tốt nhất và đánh giá trên tập test.

    extra_eval: {tên: đường_dẫn_jsonl} – các tập đánh giá bổ sung (vd. tập Test đầy đủ ở mục 5.1.4).
    """
    t = cfg["train"]
    out = Path(resolve(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    log_ = get_logger("mabsa", out / "train.log")
    save_config(cfg, out / "config.yaml")
    set_seed(t.get("seed", 42))
    device = get_device(t.get("device", "auto"))

    tokenizer = build_tokenizer_from_cfg(cfg)
    train_rows, val_rows, test_rows = load_split(cfg, "train"), load_split(cfg, "val"), load_split(cfg, "test")
    log_.info("Dữ liệu: train=%d, val=%d, test=%d | thiết bị: %s", len(train_rows), len(val_rows), len(test_rows), device)
    train_loader = build_loader(train_rows, cfg, tokenizer, train=True)
    val_loader = build_loader(val_rows, cfg, tokenizer)
    test_loader = build_loader(test_rows, cfg, tokenizer)

    model = build_model(cfg, tokenizer, train_rows).to(device)
    log_.info("Mô hình: %s | tham số huấn luyện: %s", cfg["model"]["type"], f"{count_parameters(model):,}")

    accum = t.get("grad_accum_steps", 1)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total = steps_per_epoch * t["epochs"]
    optimizer = build_optimizer(model, t)
    scheduler = linear_warmup(optimizer, int(t.get("warmup_ratio", 0.1) * total), total)
    use_amp = t.get("fp16", True) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best, patience, history = math.inf, 0, []
    ckpt_path = out / "best.pt"
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, run = time.time(), {"loss": 0.0, "n": 0}
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False)
        for step, batch in enumerate(bar, 1):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outp = model(batch)
                loss, parts = model.compute_loss(outp, batch)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            run["loss"] += loss.item()
            run["n"] += 1
            bar.set_postfix(loss=f"{run['loss'] / run['n']:.4f}", **{k: f"{v:.3f}" for k, v in parts.items()})
        val, _ = evaluate(model, val_loader, device, "val")
        rec = {"epoch": epoch, "train_loss": run["loss"] / max(1, run["n"]), **{f"val_{k}": v for k, v in val.items()},
               "time_s": round(time.time() - t0, 1)}
        history.append(rec)
        log_.info("Epoch %d | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_macro_f1 %.4f",
                  epoch, rec["train_loss"], val["loss"], val.get("accuracy", 0), val.get("macro_f1", 0))
        if val["loss"] < best - 1e-4:
            best, patience = val["loss"], 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, ckpt_path)
        else:
            patience += 1
            if patience >= t.get("early_stopping_patience", 3):
                log_.info("Early stopping sau %d epoch không cải thiện validation loss.", patience)
                break
    write_json(history, out / "history.json")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    results = {"best_epoch": ckpt["epoch"], "val": ckpt["val"]}
    res, recs = evaluate(model, test_loader, device, "test")
    results["test"] = save_eval(res, recs, out, "test")
    for name, path in (extra_eval or {}).items():
        from mabsa.utils import read_jsonl
        loader = build_loader(read_jsonl(resolve(path)), cfg, tokenizer)
        r, rc = evaluate(model, loader, device, name)
        results[name] = save_eval(r, rc, out, name)
    write_json({k: ({kk: vv for kk, vv in v.items() if kk not in ("per_aspect", "per_class", "confusion_matrix")}
                    if isinstance(v, dict) else v) for k, v in results.items()}, out / "summary.json")
    log_.info("TEST | acc %.4f | macro_f1 %.4f | micro_f1 %.4f", results["test"]["accuracy"],
              results["test"]["macro_f1"], results["test"]["micro_f1"])
    return results
