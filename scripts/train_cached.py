"""Retrain category-conditioned MABSA on the official ViMACSA splits.

Stage 1 caches frozen encoder features in RAM and trains the new heads.
Stage 2 fine-tunes the last two PhoBERT layers with fixed visual features.
Validation selects checkpoints; the held-out test split is scored only at end.
"""
import _bootstrap
import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from mabsa import metrics as M
from mabsa.aspects import ASPECT_CODES, ASPECT2ID, CODE2NAME, SENT2ID
from mabsa.builder import build_dataset, build_model, build_tokenizer_from_cfg, get_segmenter, model_grid
from mabsa.config import load_config, resolve, save_config
from mabsa.data.dataset import Collator, class_weights, special_ids, bbox_to_cells
from mabsa.data.preprocess import image_to_tensor, load_image
from mabsa.engine import save_eval, to_device
from mabsa.predict import MABSAPredictor
from mabsa.utils import read_jsonl, set_seed, write_json, load_checkpoint


class FeatureCollator:
    def __init__(self, tokenizer, cfg, image_cache, with_text_features=True):
        self.base = Collator(tokenizer, cfg['data']['max_images'], model_grid(cfg))
        self.image_cache = image_cache
        self.with_text_features = with_text_features
        self.grid = model_grid(cfg)

    def __call__(self, items):
        batch = self.base(items)
        B, L = batch['input_ids'].shape
        K = batch['image_mask'].shape[1]
        if self.with_text_features:
            batch['text_features'] = torch.zeros(B, L, 768)
            for i, item in enumerate(items):
                feature = item['text_features']
                batch['text_features'][i, :len(feature)] = feature.float()
        batch['image_features'] = torch.zeros(B, K, self.grid ** 2, 768)
        for i, item in enumerate(items):
            for j, path in enumerate(item['image_paths'][:K]):
                batch['image_features'][i, j] = self.image_cache[path].float()
        return batch


@torch.no_grad()
def evaluate_cached(model, items, collate, device, batch_size):
    model.eval()
    multimodal, text = [], []
    for start in range(0, len(items), batch_size):
        batch = to_device(collate(items[start:start + batch_size]), device)
        out = model(batch)
        if not torch.isfinite(out['logits']).all() or not torch.isfinite(out['text_logits']).all():
            raise RuntimeError('Non-finite validation logits')
        for logits, labels, destination in [(out['logits'], batch['labels'], multimodal),
                                             (out['text_logits'], batch['text_labels'], text)]:
            probabilities = logits.softmax(-1).cpu()
            predictions = probabilities.argmax(-1)
            for i, sid in enumerate(batch['ids']):
                for j in batch['aspect_mask'][i].nonzero().flatten().tolist():
                    label = int(labels[i, j])
                    if label == -100:
                        continue
                    destination.append({'id': sid, 'aspect_index': j,
                        'category': ASPECT_CODES[int(batch['categories'][i, j])],
                        'label': label, 'pred': int(predictions[i, j]),
                        'probs': probabilities[i, j].tolist()})
    scores = {}
    for name, records in [('multimodal', multimodal), ('text_only', text)]:
        scores[name] = M.compute_metrics([r['label'] for r in records], [r['pred'] for r in records])
    return scores, {'multimodal': multimodal, 'text_only': text}


def train_stage(model, splits, collate, cfg, device, output, stage, epochs, patience, encoder_lr=None):
    groups = []
    head = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('text_encoder.')]
    groups.append({'params': head, 'lr': cfg['train']['head_lr'] if stage == 'heads' else 5e-5})
    encoder = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('text_encoder.')]
    if encoder:
        groups.append({'params': encoder, 'lr': encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['train']['weight_decay'])
    batch_size = cfg['train']['batch_size'] if stage == 'heads' else 12
    n_steps = math.ceil(len(splits['train']) / batch_size) * epochs
    warmup = max(1, int(n_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (n_steps - step) / max(1, n_steps - warmup))))
    history = []
    checkpoint = output / 'best.pt'
    best_score = float('-inf')
    if checkpoint.exists():
        previous = load_checkpoint(checkpoint)
        best_score = float(previous['selection_score'])
        del previous
    stalled = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        order = list(range(len(splits['train'])))
        random.shuffle(order)
        total_loss, steps = 0.0, 0
        for offset in range(0, len(order), batch_size):
            items = [splits['train'][i] for i in order[offset:offset + batch_size]]
            batch = to_device(collate(items), device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, _ = model.compute_loss(out, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in {stage}, epoch {epoch}, offset {offset}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        validation, _ = evaluate_cached(model, splits['val'], collate, device, cfg['train']['eval_batch_size'])
        score = (validation['text_only']['macro_f1'] + validation['multimodal']['macro_f1']) / 2
        row = {'stage': stage, 'epoch': epoch, 'train_loss': total_loss / steps,
               'validation': validation, 'selection_score': score, 'seconds': round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_json(history, output / f'{stage}_history.json')
        if score > best_score + 0.0001:
            best_score = score
            stalled = 0
            # Preserve updated encoder layers without cloning/writing every
            # frozen backbone tensor, which caused memory/disk pressure.
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                     if not k.startswith(('text_encoder.', 'image_encoder.'))
                     or k.startswith(('text_encoder.model.encoder.layer.10.',
                                      'text_encoder.model.encoder.layer.11.'))}
            saved_cfg = copy.deepcopy(cfg)
            saved_cfg['train']['completed_stage'] = stage
            saved_cfg['train']['completed_epochs_in_stage'] = epoch
            temporary = output / 'best.tmp.pt'
            torch.save({'state_dict': state, 'config': saved_cfg, 'epoch': epoch,
                        'val': validation['multimodal'], 'val_text_only': validation['text_only'],
                        'selection_score': score, 'stage': stage, 'repair_version': 2,
                        'format': 'mabsa_delta_v1',
                        'backbone_checkpoint': cfg['model']['backbone_checkpoint'],
                        'backbone_filename': Path(cfg['model']['backbone_checkpoint']).name}, temporary)
            os.replace(temporary, checkpoint)
            del state
        else:
            stalled += 1
            if stalled >= patience:
                print(f'Early stopping {stage} after {epoch} epochs.', flush=True)
                break
    ck = load_checkpoint(checkpoint)
    model.load_state_dict(ck['state_dict'])
    del ck
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/retrain.yaml')
    ap.add_argument('--initialize', default='/Users/dlb/Downloads/mabsa_travel/checkpoints/mabsa_before_retrain_20261012.pt')
    ap.add_argument('--finetune-epochs', type=int, default=5)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        for name in ['train', 'val', 'test']:
            cfg['data'][name] = str(Path(args.data_root).resolve() / (name + '.jsonl'))
    if args.output_dir:
        cfg['train']['output_dir'] = str(Path(args.output_dir).resolve())
    set_seed(cfg['train']['seed'])
    torch.set_num_threads(4)
    device = torch.device(cfg['train']['device'])
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available. Run with GPU access or change train.device to cpu.')
    output = resolve(cfg['train']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    log_file = open(output / 'train.log', 'w', encoding='utf-8')
    class ConsoleLog:
        def write(self, message):
            sys.__stdout__.write(message)
            log_file.write(message)
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()
    sys.stdout = ConsoleLog()
    save_config(cfg, output / 'config.yaml')
    raw = {name: read_jsonl(resolve(cfg['data'][name])) for name in ['train', 'val', 'test']}
    stats = {name: {'reviews': len(rows),
        'multimodal_labels': dict(Counter(a['sentiment'] for row in rows for a in row['aspects'])),
        'text_labels': dict(Counter(a['text_sentiment'] for row in rows for a in row['aspects'] if a.get('text_sentiment')))}
        for name, rows in raw.items()}
    write_json(stats, output / 'data_stats.json')
    print('Official split counts: ' + json.dumps(stats, ensure_ascii=False), flush=True)
    tokenizer = build_tokenizer_from_cfg(cfg)
    model = build_model(cfg, tokenizer, raw['train'])
    text_rows = [{'aspects': [dict(a, sentiment=a['text_sentiment']) for a in row['aspects'] if a.get('text_sentiment')]}
                 for row in raw['train']]
    model.text_class_weight = class_weights(text_rows)
    initial = load_checkpoint(args.initialize)
    cfg['model']['backbone_checkpoint'] = initial['_backbone_path']
    save_config(cfg, output / 'config.yaml')
    backbone = {name: value for name, value in initial['state_dict'].items()
                if name.startswith(('text_encoder.', 'image_encoder.'))}
    model.load_state_dict(backbone, strict=False)
    del initial, backbone
    model.to(device).eval()
    print(f'Loaded previous PhoBERT/ViT backbones. Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} new head parameters.', flush=True)

    # Initialize category queries with encoded category names, not identical
    # first-token vectors. The queries are then learned using training labels.
    segmenter = get_segmenter(cfg)
    cls, sep, pad = special_ids(tokenizer)
    prompts = [[cls] + tokenizer(segmenter(CODE2NAME[c]), add_special_tokens=False)['input_ids'] + [sep]
               for c in ASPECT_CODES]
    ids = torch.full((len(prompts), max(map(len, prompts))), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(prompts):
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, :len(sequence)] = 1
    with torch.no_grad():
        features = model.text_encoder(ids, mask)
        if not torch.isfinite(features).all():
            raise RuntimeError('Non-finite PhoBERT features on chosen device')
        pooled = (features * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        model.aspect_query.weight.copy_(model.text_proj(pooled))
    print('Category query initialization is finite.', flush=True)

    datasets = {name: build_dataset(rows, cfg, tokenizer) for name, rows in raw.items()}
    splits = {}
    for name, ds in datasets.items():
        items = []
        for i in range(len(ds)):
            row = raw[name][i]
            ids, spans = ds._encode_text(i)
            paths = row['images'][:cfg['data']['max_images']]
            aspects = row['aspects'][:ds.max_aspects]
            positions = []
            for aspect in aspects:
                positive = np.zeros(cfg['data']['max_images'] * model_grid(cfg) ** 2, dtype=bool)
                for region in aspect.get('regions', []):
                    image_index = int(region['image'])
                    if image_index >= len(paths):
                        continue
                    cells = bbox_to_cells(region['bbox'], model_grid(cfg))
                    positive[image_index * model_grid(cfg) ** 2 + np.array(cells)] = True
                positions.append(positive)
            image_mask = [True] * len(paths) if paths else [False]
            item = {'id': row['id'], 'input_ids': ids, 'aspect_spans': spans,
                    'image_paths': paths, 'image_mask': image_mask,
                    'images': np.zeros((len(image_mask), 3, 1, 1), dtype=np.float32),
                    'categories': [ASPECT2ID[a['category']] for a in aspects],
                    'labels': [SENT2ID[a['sentiment']] for a in aspects],
                    'text_labels': [SENT2ID[a['text_sentiment']] if a.get('text_sentiment') else -100 for a in aspects],
                    'region_pos': positions, 'has_region': [bool(p.any()) for p in positions]}
            items.append(item)
            if i % 500 == 0:
                print(f'Input metadata {name}: {i + 1}/{len(ds)}', flush=True)
        splits[name] = items
        print(f'Prepared input metadata {name}: {len(items)} reviews.', flush=True)

    base_collator = Collator(tokenizer, cfg['data']['max_images'], model_grid(cfg))
    with torch.no_grad():
        for name, items in splits.items():
            for start in range(0, len(items), 16):
                group = items[start:start + 16]
                cpu_batch = base_collator(group)
                batch = to_device(cpu_batch, device)
                features = model.text_encoder(batch['input_ids'], batch['attention_mask'])
                if not torch.isfinite(features).all():
                    observed = features.cpu()
                    active = batch['attention_mask'].cpu().bool()
                    print('NON-FINITE DIAGNOSTIC ' + json.dumps({
                        'ids': batch['ids'], 'lengths': active.sum(1).tolist(),
                        'nan_per_row': torch.isnan(observed).sum((1, 2)).tolist(),
                        'nan_active_tokens': int(torch.isnan(observed[active]).sum()),
                        'nan_padding_tokens': int(torch.isnan(observed[~active]).sum()),
                        'attention': model.text_encoder.model.config._attn_implementation,
                        'gpu_input_matches_cpu': torch.equal(batch['input_ids'].cpu(), cpu_batch['input_ids']),
                        'gpu_input_min': int(batch['input_ids'].min()),
                        'gpu_input_max': int(batch['input_ids'].max()),
                    }), flush=True)
                    model.text_encoder.cpu()
                    reference = model.text_encoder(cpu_batch['input_ids'], cpu_batch['attention_mask'])
                    print('CPU reference finite:', bool(torch.isfinite(reference).all()), flush=True)
                    raise RuntimeError('Non-finite cached text features; see GPU/CPU diagnostic above')
                features = features.cpu().half()
                for i, item in enumerate(group):
                    item['text_features'] = features[i, :len(item['input_ids'])].clone()
                if start % 320 == 0:
                    print(f'Text features {name}: {min(start + 16, len(items))}/{len(items)}', flush=True)
    image_cache = {}
    paths = sorted({path for items in splits.values() for item in items for path in item['image_paths']})
    grid = model_grid(cfg)
    with torch.no_grad():
        for start in range(0, len(paths), 24):
            group = paths[start:start + 24]
            pixels = torch.stack([torch.from_numpy(image_to_tensor(load_image(path), cfg['data']['image_size'],
                                                                   cfg['model']['image_encoder'])) for path in group]).to(device)
            features = model.image_encoder(pixels)
            old_grid = model.image_encoder.grid
            fmap = features.transpose(1, 2).reshape(len(group), 768, old_grid, old_grid)
            features = F.adaptive_avg_pool2d(fmap, (grid, grid)).flatten(2).transpose(1, 2)
            if not torch.isfinite(features).all():
                raise RuntimeError('Non-finite cached visual features')
            features = features.cpu().half()
            for i, path in enumerate(group):
                image_cache[path] = features[i].clone()
            if start % 240 == 0:
                print(f'Image features: {min(start + 24, len(paths))}/{len(paths)}', flush=True)
    print('Frozen features cached in RAM. Starting head training.', flush=True)
    collate = FeatureCollator(tokenizer, cfg, image_cache)
    head_history = train_stage(model, splits, collate, cfg, device, output, 'heads',
                               cfg['train']['epochs'], cfg['train']['early_stopping_patience'])

    # Updating only the final two PhoBERT layers saves memory while allowing
    # the encoder to learn sentiment cues from the full real training split.
    model.cfg['model']['freeze_text_encoder'] = False
    for name, parameter in model.text_encoder.named_parameters():
        parameter.requires_grad = name.startswith(('model.encoder.layer.10.', 'model.encoder.layer.11.'))
    collate.with_text_features = False
    for items in splits.values():
        for item in items:
            item.pop('text_features', None)
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    print('Fine-tuning the final two PhoBERT layers.', flush=True)
    fine_history = train_stage(model, splits, collate, cfg, device, output, 'finetune',
                               args.finetune_epochs, 2, encoder_lr=1e-5)
    # Cache cannot be used after encoder fine-tuning; evaluate with live text.
    scores, records = evaluate_cached(model, splits['test'], collate, device, cfg['train']['eval_batch_size'])
    for name, recs in records.items():
        full = dict(scores[name], loss=None)
        save_eval(full, recs, output, 'test_' + name)
    report = {'held_out_test': scores, 'head_epochs_run': len(head_history),
              'finetune_epochs_run': len(fine_history), 'supported_aspects': cfg['model']['supported_aspects'],
              'split_source': 'Kaggle nguynquy/macsaa v13, original train/dev/test',
              'test_used_for_selection': False}
    selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=True, mmap=True)
    report.update(selected_stage=selected['stage'], selected_epoch=selected['epoch'],
                  validation_selection_score=selected['selection_score'])
    save_config(selected['config'], output / 'config.yaml')
    del selected
    write_json(report, output / 'training_report.json')
    del splits, image_cache, datasets, raw, collate
    model.cpu()
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    predictor = MABSAPredictor(output / 'best.pt', extractor_checkpoint=None, device='cpu')
    texts = {
        'user_negative': 'Phòng ốc rất bẩn và hôi, nhà vệ sinh không được dọn, tiện nghi thường xuyên hỏng, nhân viên phục vụ tệ và giá cả quá đắt.',
        'positive': 'Phòng ốc rất sạch và thơm, nhà vệ sinh được dọn kỹ, tiện nghi hoạt động tốt, nhân viên phục vụ tận tình và giá cả hợp lý.',
        'neutral': 'Phòng có hai giường, nhà vệ sinh ở cạnh cửa, tiện nghi gồm tivi và điều hòa, nhân viên trực ở quầy lễ tân, giá phòng là 500 nghìn đồng.',
        'mixed': 'Phòng rất bẩn nhưng nhân viên phục vụ tận tình và thân thiện.',
    }
    examples = {name: predictor.predict(text, []) for name, text in texts.items()}
    write_json(examples, output / 'sanity_predictions.json')
    print('FINAL REPORT ' + json.dumps(report, ensure_ascii=False), flush=True)
    print('SANITY PREDICTIONS ' + json.dumps(examples, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
