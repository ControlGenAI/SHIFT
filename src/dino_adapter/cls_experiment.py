"""CLS extraction, train-only mean differences, and real-feature guidance in SHIFT."""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from .features import DinoFeatures
from .runtime import load_pipeline, selected_blocks, save_json, digest, generate
from .cls_guidance import CLSActivationGuidance


def read_cls_config(path):
    config = json.loads(Path(path).read_text())
    if config['width'] <= 0 or config['height'] <= 0 or config['width'] % 16 or config['height'] % 16:
        raise ValueError('FLUX image dimensions must be positive multiples of 16')
    if config['dino_size'] <= 0 or config['inference_steps'] < 1:
        raise ValueError('Invalid DINO size or inference step count')
    alphas = config['alphas']
    if (not alphas or 0 not in alphas or len(set(alphas)) != len(alphas)
            or any(not math.isfinite(a) for a in alphas)):
        raise ValueError('Use distinct finite alphas including the zero control')
    blocks = config['blocks']
    if blocks != 'all' and (not isinstance(blocks, list) or not blocks or
            len(set(blocks)) != len(blocks) or any(not isinstance(b, int) or b < 0 for b in blocks)):
        raise ValueError('blocks must be all or distinct nonnegative indices')
    return config


def cls_signature(config):
    return dict(model=config['dino_model'], revision=config.get('dino_revision'),
                size=config['dino_size'], preprocessing='full_frame_bicubic_square_imagenet')


def validate_records(rows):
    if not rows:
        raise ValueError('Empty image manifest')
    pairs, seeds = {}, {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or row['label'] not in (0, 1):
            raise ValueError('Need split=train/val/test and label=0/1')
        group = pairs.setdefault(row['pair_id'], {})
        if row['label'] in group or any(r['split'] != row['split'] for r in group.values()):
            raise ValueError('Duplicate pair label or pair crossing splits')
        group[row['label']] = row
        if 'seed' in row:
            if row['seed'] in seeds and seeds[row['seed']] != row['split']:
                raise ValueError('Seed crosses splits')
            seeds[row['seed']] = row['split']
    if any(set(group) != {0, 1} for group in pairs.values() if next(iter(group.values()))['split'] == 'train'):
        raise ValueError('Need complete train pairs')


def paired_mean(rows, vectors):
    validate_records(rows)
    vectors = torch.as_tensor(vectors) if isinstance(vectors, torch.Tensor) else torch.stack(vectors)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not torch.isfinite(vectors).all():
        raise ValueError('CLS tensors must be finite [number_of_records, D]')
    if (vectors.norm(dim=-1) < 1e-8).any():
        raise ValueError('Zero CLS feature')
    vectors = F.normalize(vectors.detach().float(), dim=-1)
    pairs = {}
    for row, vector in zip(rows, vectors):
        if row['split'] == 'train':
            pairs.setdefault(row['pair_id'], {})[row['label']] = vector
    if not pairs:
        raise ValueError('No train pairs')
    positives = torch.stack([p[1] for p in pairs.values()])
    negatives = torch.stack([p[0] for p in pairs.values()])
    pos, neg = positives.mean(0), negatives.mean(0)
    direction = pos - neg
    if direction.norm() < 1e-8:
        raise ValueError('Degenerate CLS direction')
    return dict(mean_positive=pos, mean_negative=neg, direction=direction,
                train_pair_ids=sorted(pairs), n_train_pairs=len(pairs),
                train_seeds=sorted({r['seed'] for r in rows if r['split'] == 'train' and 'seed' in r}),
                paired_delta_norm_mean=float((positives - negatives).norm(dim=-1).mean()))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def extract(config, manifest, dataset, output, device, cached=False):
    data = None
    if dataset is not None:
        root_images = Path(dataset)
        data = json.loads((root_images / 'dataset.json').read_text())
        rows = data['samples']
        if cached and cls_signature(data['config']) != cls_signature(config):
            raise ValueError('Cached CLS has different DINO model/preprocessing')
    else:
        rows, root_images = read_rows(manifest), Path(manifest).parent
        if cached:
            raise ValueError('--cached-cls requires --dataset')
    validate_records(rows)
    # Detect missing inputs before any model loads or output-directory creation.
    for row in rows:
        file = root_images / row['features' if cached else 'image']
        if not file.is_file():
            raise FileNotFoundError(file)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    dino = None if cached else DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    vectors = []
    for row in rows:
        if cached:
            cls = torch.load(root_images / row['features'], map_location='cpu', weights_only=True)['cls']
        else:
            with Image.open(root_images / row['image']) as image:
                _, batch_cls = dino(image, None)
            cls = batch_cls[0]
        vectors.append(cls)
    statistics = paired_mean(rows, vectors)
    torch.save(dict(version=1, rows=rows, cls=torch.stack(vectors), config=config,
                    cls_signature=cls_signature(config)), root / 'cls_features.pt')
    torch.save(dict(version=1, **statistics, cls_signature=cls_signature(config)), root / 'cls_direction.pt')
    save_json(root / 'summary.json', dict(n_images=len(rows), n_train_pairs=statistics['n_train_pairs'],
                                         direction_norm=float(statistics['direction'].norm()),
                                         paired_delta_norm_mean=statistics['paired_delta_norm_mean'],
                                         cached_cls=cached, cls_signature=cls_signature(config)))


def mean_from_features(features, output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    payload = torch.load(features, map_location='cpu', weights_only=True)
    statistics = paired_mean(payload['rows'], payload['cls'])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, **statistics, cls_signature=payload['cls_signature']), path)


def optimize(config, rows, direction_path, output, device):
    payload = torch.load(direction_path, map_location='cpu', weights_only=True)
    if cls_signature(config) != payload['cls_signature']:
        raise ValueError('CLS direction and guidance DINO model/preprocessing differ')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty prompts with unique IDs')
    for row in rows:
        if row['pair_id'] in payload['train_pair_ids'] or row['seed'] in payload.get('train_seeds', []):
            raise ValueError('Use held-out pairs and seeds for comparisons')
        if not row['prompt'] or not isinstance(row['seed'], int):
            raise ValueError('Need a prompt and integer seed')
        if not row['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['id']):
            raise ValueError('Unsafe output ID')
    settings = config['cls_optimization']
    space = settings.get('space', 'activation')
    if not settings['steps'] or any(s < 0 or s >= config['inference_steps'] for s in settings['steps']):
        raise ValueError('Invalid optimization timestep')
    # Validate optimization settings without model loading.
    CLSActivationGuidance(None, payload['direction'], 0, steps=settings['steps'], alpha=config['alphas'][0],
        iterations=settings['iterations'], learning_rate=settings['learning_rate'],
        preservation_weight=settings['preservation_weight'], max_relative_rms=settings['max_relative_rms'],
        optimization_space=space)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'config.json', config)
    import diffusers, transformers
    save_json(root / 'provenance.json', dict(direction_sha256=digest(direction_path),
              cls_signature=payload['cls_signature'], torch_version=str(torch.__version__),
              diffusers_version=diffusers.__version__, transformers_version=transformers.__version__))
    pipe = load_pipeline(config, device)
    if settings.get('gradient_checkpointing', True):
        pipe.transformer.enable_gradient_checkpointing()
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    blocks = selected_blocks(config, len(pipe.transformer.transformer_blocks)) if space == 'activation' else [0]
    entries = []
    for row in rows:
        baseline = generate(pipe, config, row['prompt'], row['seed'])
        baseline.save(root / f"{row['id']}_baseline.png")
        with torch.no_grad():
            _, baseline_cls = dino(baseline, None)
        entries.append(dict(sample_id=row['id'], mode='baseline', image=f"{row['id']}_baseline.png"))
        for block in blocks:
            for index, alpha in enumerate(config['alphas']):
                guidance = CLSActivationGuidance(dino, payload['direction'], block,
                    steps=settings['steps'], alpha=alpha, iterations=settings['iterations'],
                    learning_rate=settings['learning_rate'], preservation_weight=settings['preservation_weight'],
                    max_relative_rms=settings['max_relative_rms'], resolution=(config['height'], config['width']),
                    optimization_space=space)
                image = pipe(row['prompt'], width=config['width'], height=config['height'],
                    num_inference_steps=config['inference_steps'], guidance_scale=config['guidance_scale'],
                    max_sequence_length=256, generator=torch.Generator('cpu').manual_seed(row['seed']),
                    structure_strength=0., txt_steering={'vector': None}, activation_guidance=guidance).images[0]
                name = f"{row['id']}_{space}_block{block}_alpha{index}"
                image.save(root / f'{name}.png')
                torch.save(guidance.references, root / f'{name}_cls_targets.pt')
                _, final_cls = dino(image, None)
                direction = payload['direction'].cpu()
                # Post-generation check: does optimizing x0 CLS affect the final image?
                projected_removal = float(((baseline_cls - final_cls) * direction).sum() / direction.square().sum())
                import numpy as np
                delta = np.asarray(image, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
                metrics = dict(final_cls_removal_proxy=projected_removal,
                               baseline_pixel_mae=float(abs(delta).mean()), baseline_pixel_max_abs=float(abs(delta).max()))
                save_json(root / f'{name}.json', dict(sample=row, block=block if space == 'activation' else None,
                          alpha=alpha, space=space, logs=guidance.logs, **metrics))
                entries.append(dict(sample_id=row['id'], mode=space, block=block if space == 'activation' else None,
                                    alpha=alpha, image=f'{name}.png', **metrics))
                save_json(root / 'generations.json', entries)
    save_json(root / 'generations.json', entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cls_guidance.json')
    parser.add_argument('--device', default='cuda:0')
    sub = parser.add_subparsers(dest='command', required=True)
    ex = sub.add_parser('extract', help='Actual CLS from images or cached DINO outputs; also save mean diff')
    source = ex.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--dataset', help='Existing SHIFT dataset with dataset.json')
    ex.add_argument('--cached-cls', action='store_true', help='Use existing features.pt CLS; no models loaded')
    ex.add_argument('--output', required=True)
    mean = sub.add_parser('mean', help='Recompute paired mean from saved cls_features.pt, no model loading')
    mean.add_argument('--features', required=True)
    mean.add_argument('--output', required=True)
    opt = sub.add_parser('optimize')
    source = opt.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompts')
    source.add_argument('--dataset', help='Use positive held-out prompts from existing dataset.json')
    opt.add_argument('--split', choices=['val', 'test'], default='test')
    opt.add_argument('--num-samples', type=int, default=1)
    opt.add_argument('--direction', required=True)
    opt.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'mean':
        mean_from_features(args.features, args.output)
        return
    config = read_cls_config(args.config)
    if args.command == 'extract':
        extract(config, args.manifest, args.dataset, args.output, args.device, args.cached_cls)
    else:
        if args.num_samples < 1:
            raise ValueError('--num-samples must be positive')
        if args.prompts:
            rows = read_rows(args.prompts)
        else:
            data = json.loads((Path(args.dataset) / 'dataset.json').read_text())
            rows = [r for r in data['samples'] if r['split'] == args.split and r['label'] == 1]
        optimize(config, rows[:args.num_samples], args.direction, args.output, args.device)


if __name__ == '__main__':
    main()
