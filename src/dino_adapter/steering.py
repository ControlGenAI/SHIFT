"""Image-only interventions and matched-activation-RMS constant-direction control."""
from pathlib import Path
import csv
import json
import torch
from .data import load_dataset
from .directions import region_mask
from .hooks import ImageBlockHook
from .runtime import (load_adapter, load_pipeline, generate, require_compatible,
                      save_json, digest)


def rms(x):
    return float(x.float().square().mean().sqrt())


class AdapterEdit:
    def __init__(self, adapter, direction, alpha):
        self.adapter, self.direction, self.alpha = adapter, direction, alpha
        self.stats = {}

    @torch.no_grad()
    def __call__(self, h):
        z, r = self.adapter.encode(h)
        reconstructed = self.adapter.decode(z, r)
        changed = self.adapter.decode(z - self.alpha * self.direction.to(h.device), r)
        if not torch.isfinite(changed).all():
            raise RuntimeError('Nonfinite inverse after intervention')
        self.stats = dict(roundtrip_max_abs=float((reconstructed - h.float()).abs().max()),
                          roundtrip_rms=rms(reconstructed - h.float()),
                          edit_rms_fp32=rms(changed - h.float()),
                          edit_rms=rms(changed.to(h.dtype).float() - h.float()))
        return changed


class ConstantEdit:
    def __init__(self, direction, target_rms, alpha):
        self.direction, self.target_rms, self.alpha = direction, target_rms, alpha
        self.stats = {}

    @torch.no_grad()
    def __call__(self, h):
        # Direction is fixed across samples. Scalar calibrates intervention size.
        direction = self.direction.to(h.device).float()
        norm = rms(direction)
        if norm <= 1e-12:
            raise ValueError('Zero constant direction')
        sign = 1 if self.alpha > 0 else -1 if self.alpha < 0 else 0
        changed = h.float() - sign * self.target_rms * direction / norm
        self.stats = dict(edit_rms=rms(changed.to(h.dtype).float() - h.float()))
        return changed


class DirectImageEdit:
    """Raw paired post-block direction: h' = h - alpha * mean(h_pos-h_neg)."""
    def __init__(self, direction, alpha):
        self.direction, self.alpha, self.stats = direction, alpha, {}

    @torch.no_grad()
    def __call__(self, h):
        direction = self.direction.to(device=h.device, dtype=torch.float32)
        if direction.shape != h.shape[-2:]:
            raise ValueError('Image-token direction must match [tokens, channels]')
        changed = h.float() - self.alpha * direction
        self.stats = dict(edit_rms=rms(changed.to(h.dtype).float() - h.float()))
        return changed


def steer(config, dataset, adapters, directions, output, device, split='test'):
    mode = config.get('steering_mode', 'adapter_comparison')
    if mode not in ('adapter_comparison', 'image_tokens'):
        raise ValueError('Unknown steering_mode')
    direct = mode == 'image_tokens'
    if not direct and adapters is None:
        raise ValueError('Adapter comparison requires --adapters')
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    direction_data = torch.load(directions, map_location='cpu', weights_only=True)
    require_compatible(config, direction_data['config'])
    if config['roi'] != direction_data['config']['roi']:
        raise ValueError('ROI differs from the saved directions')
    if list(data['grid']) != list(direction_data['grid']):
        raise ValueError('Direction spatial grid differs')
    blocks = data['blocks'] if config['blocks'] == 'all' else config['blocks']
    if not blocks or any(b not in data['blocks'] for b in blocks):
        raise ValueError('Requested block is not in the dataset')
    samples = [s for s in data['samples'] if s['split'] == split and s['label'] == 1]
    if not samples:
        raise ValueError(f'No positive samples in {split} split')
    training_ids = set(direction_data['train_pair_ids'])
    if any(s['pair_id'] in training_ids for s in samples):
        raise ValueError('Evaluation must use held-out pairs')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'config.json', config)
    # Check every artifact before loading the large model.
    for block in blocks:
        if str(block) not in direction_data['vectors']:
            raise ValueError('No direction for block')
        if direct:
            continue
        adapter, payload = load_adapter(Path(adapters) / f'block_{block}.pt', 'cpu')
        require_compatible(config, payload['config'])
        if payload['block'] != block or payload['step'] != config['step'] or payload['grid'] != data['grid']:
            raise ValueError('Adapter block/step/grid mismatch')
        if any(s['pair_id'] in payload['train_pair_ids'] for s in samples):
            raise ValueError('Adapter evaluation leaks training pairs')
        if str(block) not in direction_data['vectors']:
            raise ValueError('No direction for block')
        if direction_data['vectors'][str(block)]['z'].shape[-1] != adapter.z_dim:
            raise ValueError('Direction dimension differs from adapter')
    if not direct:
        del adapter
    pipe = load_pipeline(config, device)
    entries = []
    for sample in samples:
        folder = root / sample['id']
        folder.mkdir()
        baseline = generate(pipe, config, sample['prompt'], sample['seed'])
        baseline_path = folder / 'baseline.png'
        baseline.save(baseline_path)
        entries.append(dict(sample_id=sample['id'], mode='baseline', block=None, alpha=0.,
                            image=str(baseline_path.relative_to(root)), seed=sample['seed']))
        for block in blocks:
            if direct:
                direction = direction_data['vectors'][str(block)]['h']
                for index, alpha in enumerate(config['alphas']):
                    edit = DirectImageEdit(direction, alpha)
                    with ImageBlockHook(pipe.transformer, block, config['step'], edit) as hook:
                        image = generate(pipe, config, sample['prompt'], sample['seed'], hook)
                    path = folder / f'block_{block}_alpha_{index}_image_tokens.png'
                    image.save(path)
                    entries.append(dict(sample_id=sample['id'], mode='image_tokens', block=block,
                                        alpha=alpha, image=str(path.relative_to(root)), seed=sample['seed'],
                                        **edit.stats))
                    save_json(root / 'generations.json', entries)
                continue
            checkpoint = Path(adapters) / f'block_{block}.pt'
            adapter, _ = load_adapter(checkpoint, device)
            vectors = direction_data['vectors'][str(block)]
            for index, alpha in enumerate(config['alphas']):
                edit = AdapterEdit(adapter, vectors['z'], alpha)
                with ImageBlockHook(pipe.transformer, block, config['step'], edit) as hook:
                    image = generate(pipe, config, sample['prompt'], sample['seed'], hook)
                path = folder / f'block_{block}_alpha_{index}_adapter.png'
                image.save(path)
                entries.append(dict(sample_id=sample['id'], mode='adapter', block=block, alpha=alpha,
                                    image=str(path.relative_to(root)), seed=sample['seed'],
                                    checkpoint_sha256=digest(checkpoint), **edit.stats))
                control = ConstantEdit(vectors['h'], edit.stats['edit_rms'], alpha)
                with ImageBlockHook(pipe.transformer, block, config['step'], control) as hook:
                    image = generate(pipe, config, sample['prompt'], sample['seed'], hook)
                path = folder / f'block_{block}_alpha_{index}_constant.png'
                image.save(path)
                entries.append(dict(sample_id=sample['id'], mode='constant', block=block, alpha=alpha,
                                    image=str(path.relative_to(root)), seed=sample['seed'], **control.stats))
                save_json(root / 'generations.json', entries)
            del adapter
    save_json(root / 'generations.json', entries)
    with (root / 'visual_review.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'glasses_present', 'same_person', 'other_changes', 'notes'])
        for entry in entries:
            writer.writerow([entry['image'], '', '', '', ''])
    # Quantitative image evaluation is a separate command; it loads DINO without FLUX.


def evaluate(config, directions, results, device):
    """DINO proxies and outside-ROI pixels. Not an independent identity benchmark."""
    from PIL import Image
    import numpy as np
    from .features import DinoFeatures
    root = Path(results)
    require_compatible(config, json.loads((root / 'config.json').read_text()))
    direction_data = torch.load(directions, map_location='cpu', weights_only=True)
    require_compatible(config, direction_data['config'])
    if config['roi'] != direction_data['config']['roi']:
        raise ValueError('ROI differs from the saved directions')
    entries = json.loads((root / 'generations.json').read_text())
    grid = direction_data['grid']
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    baselines = {}
    for entry in entries:
        if entry['mode'] == 'baseline':
            with Image.open(root / entry['image']) as image:
                patches, cls = dino(image, grid)
                pixels = torch.from_numpy(np.array(image.convert('RGB'))).float() / 255
            baselines[entry['sample_id']] = (patches, cls, pixels)
    rows = []
    first_direction = next(iter(direction_data['vectors'].values()))['z']
    for entry in entries:
        with Image.open(root / entry['image']) as image:
            patches, cls = dino(image, grid)
            pixels = torch.from_numpy(np.array(image.convert('RGB'))).float() / 255
        bp, bc, bx = baselines[entry['sample_id']]
        # Positive score means removal along the trained paired direction.
        direction = first_direction if entry['block'] is None else direction_data['vectors'][str(entry['block'])]['z']
        removal = float(((bp - patches) * direction).sum() / direction.square().sum().clamp_min(1e-12))
        outside = ~region_mask((pixels.shape[0], pixels.shape[1]), config['roi'])
        if not outside.any():
            raise ValueError('Evaluation requires a non-full-frame ROI for outside-change metric')
        outside_mse = float((pixels - bx).square().reshape(-1, 3)[outside].mean())
        rows.append(dict(**entry, dino_removal_proxy=removal, dino_cls_similarity=float((cls * bc).sum()),
                         outside_roi_mse=outside_mse, image_mae=float((pixels - bx).abs().mean()),
                         image_max_abs=float((pixels - bx).abs().max())))
    save_json(root / 'metrics.json', rows)
    # Report nearest constant-control attribute effect in each alpha sweep, plus mismatch.
    # A large mismatch explicitly invalidates a claim of equal attribute strength.
    matched = []
    for row in rows:
        if row['mode'] != 'adapter' or row['alpha'] == 0:
            continue
        candidates = [r for r in rows if r['mode'] == 'constant' and
                      r['sample_id'] == row['sample_id'] and r['block'] == row['block']]
        closest = min(candidates, key=lambda r: abs(r['dino_removal_proxy'] - row['dino_removal_proxy']))
        matched.append(dict(adapter_image=row['image'], constant_image=closest['image'],
                            removal_proxy_gap=abs(row['dino_removal_proxy'] - closest['dino_removal_proxy']),
                            adapter_outside_mse=row['outside_roi_mse'], constant_outside_mse=closest['outside_roi_mse'],
                            adapter_cls=row['dino_cls_similarity'], constant_cls=closest['dino_cls_similarity']))
    save_json(root / 'matched_effect_comparisons.json', matched)
