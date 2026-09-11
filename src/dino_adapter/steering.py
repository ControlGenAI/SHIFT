"""Image-only interventions and matched-activation-RMS constant-direction control."""
from pathlib import Path
import csv
import json
import numpy as np
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
        if self.direction.shape[-1] != self.adapter.z_dim:
            raise ValueError('Direction dimension differs from adapter z')
        if self.direction.ndim > 1 and self.direction.shape[-2] != h.shape[-2]:
            raise ValueError('Direction token count differs from the image-token grid')
        z, r = self.adapter.encode(h)
        reconstructed = self.adapter.decode(z, r)
        changed = self.adapter.decode(z - self.alpha * self.direction.to(h.device), r)
        if not torch.isfinite(changed).all():
            raise RuntimeError('Nonfinite inverse after intervention')
        self.stats = dict(roundtrip_max_abs=float((reconstructed - h.float()).abs().max()),
                          roundtrip_rms=rms(reconstructed - h.float()),
                          roundtrip_rms_after_cast=rms(reconstructed.to(h.dtype).float() - h.float()),
                          activation_rms=rms(h),
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
        if direction.shape != h.shape[-2:]:
            raise ValueError('Constant direction must match [tokens, channels]')
        norm = rms(direction)
        if norm <= 1e-12:
            raise ValueError('Zero constant direction')
        sign = 1 if self.alpha > 0 else -1 if self.alpha < 0 else 0
        changed = h.float() - sign * self.target_rms * direction / norm
        self.stats = dict(edit_rms=rms(changed.to(h.dtype).float() - h.float()),
                          activation_rms=rms(h))
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
        self.stats = dict(edit_rms=rms(changed.to(h.dtype).float() - h.float()),
                          activation_rms=rms(h))
        return changed


def _unit(vector, eps=1e-6):
    return vector / vector.norm(dim=-1, keepdim=True).clamp(min=eps)


def _renormalize(edited, original_norm, eps=1e-6):
    """Match apply_steering_with_injection_sd35: unit direction, restore token norm."""
    return _unit(edited, eps) * original_norm


class RenormImageEdit:
    """h' = renorm(h - alpha * |h_token| * unit(direction)).

    Alpha is a fraction of the per-token norm, so a shared grid is comparable
    across per-token and mean-vector directions.
    """

    def __init__(self, direction, alpha, renorm=True):
        self.direction, self.alpha, self.renorm = direction, alpha, renorm
        self.stats = {}

    @torch.no_grad()
    def __call__(self, h):
        x = h.float()
        direction = self.direction.to(device=x.device, dtype=x.dtype)
        if direction.shape != x.shape[-2:]:
            raise ValueError('Direction must match [tokens, channels]')
        norms = x.norm(dim=-1, keepdim=True)
        changed = x - self.alpha * norms * _unit(direction)
        if self.renorm:
            changed = _renormalize(changed, norms)
        self.stats = dict(edit_rms=rms(changed - x), activation_rms=rms(x),
                          norm_drift=float((changed.norm(dim=-1) - norms.squeeze(-1)).abs().max()))
        return changed


class RenormAdapterEdit:
    """z' = renorm(z - alpha * |z_token| * unit(direction)); h' = F^-1(z', r)."""

    def __init__(self, adapter, direction, alpha, renorm=True):
        self.adapter, self.direction, self.alpha, self.renorm = adapter, direction, alpha, renorm
        self.stats = {}

    @torch.no_grad()
    def __call__(self, h):
        z, r = self.adapter.encode(h)
        direction = self.direction.to(device=z.device, dtype=z.dtype)
        if direction.shape[-1] != self.adapter.z_dim:
            raise ValueError('Direction dimension differs from adapter z')
        if direction.ndim > 1 and direction.shape[-2] != z.shape[-2]:
            raise ValueError('Direction token count differs from the image-token grid')
        norms = z.norm(dim=-1, keepdim=True)
        changed_z = z - self.alpha * norms * _unit(direction)
        if self.renorm:
            changed_z = _renormalize(changed_z, norms)
        changed = self.adapter.decode(changed_z, r)
        if not torch.isfinite(changed).all():
            raise RuntimeError('Nonfinite inverse after intervention')
        self.stats = dict(
            edit_rms=rms(changed - h.float()), activation_rms=rms(h),
            z_norm_drift=float((changed_z.norm(dim=-1) - norms.squeeze(-1)).abs().max()),
            z_cosine_shift=float(torch.nn.functional.cosine_similarity(changed_z, z, dim=-1).mean()))
        return changed


def _pixel_diff(a, b):
    return int(np.abs(np.asarray(a.convert('RGB'), dtype=np.int16)
                      - np.asarray(b.convert('RGB'), dtype=np.int16)).max())


def steer(config, dataset, adapters, directions, output, device, split='test', resume=False):
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
    tokens = data['grid'][0] * data['grid'][1]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=resume)
    save_json(root / 'config.json', config)
    entries, done = [], {}
    if resume and (root / 'generations.json').exists():
        entries = json.loads((root / 'generations.json').read_text())
        entries = [e for e in entries if (root / e['image']).exists()]
        done = {e['image']: e for e in entries}
        print(f'resuming with {len(entries)} existing generations', flush=True)

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
        if direction_data['vectors'][str(block)]['z'].shape[-1] != adapter.z_dim:
            raise ValueError('Direction dimension differs from adapter')
        del adapter, payload

    pipe = load_pipeline(config, device)

    def run(name, folder, sample, edit, extra):
        """Generate one condition, or reuse it when resuming."""
        path = folder / name
        relative = str(path.relative_to(root))
        if relative in done:
            return done[relative]
        if edit is None:
            image = generate(pipe, config, sample['prompt'], sample['seed'])
            stats = {}
        else:
            with ImageBlockHook(pipe.transformer, extra['block'], config['step'], edit,
                                capture=False, expected_tokens=tokens) as hook:
                image = generate(pipe, config, sample['prompt'], sample['seed'], hook)
            stats = edit.stats
        image.save(path)
        entry = dict(sample_id=sample['id'], seed=sample['seed'], image=relative, **extra, **stats)
        entries.append(entry)
        done[relative] = entry
        save_json(root / 'generations.json', entries)
        return entry

    for sample in samples:
        folder = root / sample['id']
        folder.mkdir(exist_ok=True)
        baseline = run('baseline.png', folder, sample, None,
                       dict(mode='baseline', block=None, alpha=0.))
        # The dataset image used the same prompt, seed and schedule, so an
        # unhooked baseline must reproduce it exactly. Cheap end-to-end check.
        reference_path = Path(dataset) / sample['image']
        if 'baseline_matches_dataset_max_abs' not in baseline and reference_path.exists():
            from PIL import Image
            with Image.open(reference_path) as reference, \
                    Image.open(root / baseline['image']) as produced:
                baseline['baseline_matches_dataset_max_abs'] = _pixel_diff(reference, produced)
            save_json(root / 'generations.json', entries)
        for block in blocks:
            vectors = direction_data['vectors'][str(block)]
            if direct:
                for index, alpha in enumerate(config['alphas']):
                    run(f'block_{block}_alpha_{index}_image_tokens.png', folder, sample,
                        DirectImageEdit(vectors['h'], alpha),
                        dict(mode='image_tokens', block=block, alpha=alpha))
                continue
            checkpoint = Path(adapters) / f'block_{block}.pt'
            adapter, _ = load_adapter(checkpoint, device)
            sha = digest(checkpoint)
            for index, alpha in enumerate(config['alphas']):
                edit = AdapterEdit(adapter, vectors['z'], alpha)
                entry = run(f'block_{block}_alpha_{index}_adapter.png', folder, sample, edit,
                            dict(mode='adapter', block=block, alpha=alpha, checkpoint_sha256=sha))
                # Match the control to the adapter's actual activation change.
                target = entry.get('edit_rms', edit.stats.get('edit_rms', 0.0))
                run(f'block_{block}_alpha_{index}_constant.png', folder, sample,
                    ConstantEdit(vectors['h'], target, alpha),
                    dict(mode='constant', block=block, alpha=alpha, matched_to_rms=target))
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"{sample['id']}: {len(entries)} generations so far", flush=True)
    save_json(root / 'generations.json', entries)
    with (root / 'visual_review.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'mode', 'block', 'alpha', 'glasses_present', 'same_person',
                         'other_changes', 'notes'])
        for entry in entries:
            writer.writerow([entry['image'], entry['mode'], entry['block'], entry['alpha'], '', '', '', ''])
    # Quantitative image evaluation is a separate command; it loads DINO without FLUX.


def evaluate(config, directions, results, device):
    """DINO proxies plus an independent glasses classifier and identity score."""
    from PIL import Image
    from .features import DinoFeatures
    from .metrics import GlassesScorer, IdentityScorer
    root = Path(results)
    require_compatible(config, json.loads((root / 'config.json').read_text()))
    direction_data = torch.load(directions, map_location='cpu', weights_only=True)
    require_compatible(config, direction_data['config'])
    if config['roi'] != direction_data['config']['roi']:
        raise ValueError('ROI differs from the saved directions')
    entries = json.loads((root / 'generations.json').read_text())
    grid = direction_data['grid']
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    glasses = GlassesScorer(device=device)
    identity = IdentityScorer(device=device)

    def read(path):
        with Image.open(root / path) as image:
            image = image.convert('RGB')
            patches, cls = dino(image, grid)
            pixels = torch.from_numpy(np.asarray(image)).float() / 255
            return patches, cls, pixels, glasses(image), identity.embed(image)

    baselines = {e['sample_id']: read(e['image']) for e in entries if e['mode'] == 'baseline'}
    rows = []
    first_direction = next(iter(direction_data['vectors'].values()))['z']
    for index, entry in enumerate(entries):
        if entry['mode'] == 'baseline':
            patches, cls, pixels, glasses_prob, face = baselines[entry['sample_id']]
        else:
            patches, cls, pixels, glasses_prob, face = read(entry['image'])
        bp, bc, bx, b_glasses, b_face = baselines[entry['sample_id']]
        # Positive score means removal along the trained paired direction.
        direction = first_direction if entry['block'] is None else direction_data['vectors'][str(entry['block'])]['z']
        removal = float(((bp - patches) * direction).sum() / direction.square().sum().clamp_min(1e-12))
        inside = region_mask((pixels.shape[0], pixels.shape[1]), config['roi'])
        outside = ~inside
        if not outside.any():
            raise ValueError('Evaluation requires a non-full-frame ROI for outside-change metric')
        flat = (pixels - bx).square().reshape(-1, 3)
        rows.append(dict(**entry,
                         dino_removal_proxy=removal,
                         dino_cls_similarity=float((cls * bc).sum()),
                         inside_roi_mse=float(flat[inside].mean()),
                         outside_roi_mse=float(flat[outside].mean()),
                         image_mae=float((pixels - bx).abs().mean()),
                         image_max_abs=float((pixels - bx).abs().max()),
                         glasses_prob=glasses_prob,
                         baseline_glasses_prob=b_glasses,
                         glasses_prob_drop=b_glasses - glasses_prob,
                         glasses_removed=bool(b_glasses >= glasses.threshold > glasses_prob),
                         identity_similarity=IdentityScorer.similarity(face, b_face),
                         face_detected=face is not None))
        if (index + 1) % 200 == 0:
            print(f'  scored {index + 1}/{len(entries)}', flush=True)
    save_json(root / 'metrics.json', dict(
        detector=dict(glasses_model=glasses.model_name, glasses_threshold=glasses.threshold,
                      identity_model=identity.model_name, face_detection_failures=identity.failures),
        rows=rows))

    # Report nearest constant-control attribute effect in each alpha sweep, plus mismatch.
    # A large mismatch explicitly invalidates a claim of equal attribute strength.
    matched = []
    for row in rows:
        if row['mode'] != 'adapter' or row['alpha'] == 0:
            continue
        # alpha=0 constant control is the identity, so it is not a matched effect.
        candidates = [r for r in rows if r['mode'] == 'constant' and r['alpha'] != 0 and
                      r['sample_id'] == row['sample_id'] and r['block'] == row['block']]
        if not candidates:
            continue
        closest = min(candidates, key=lambda r: abs(r['dino_removal_proxy'] - row['dino_removal_proxy']))
        matched.append(dict(adapter_image=row['image'], constant_image=closest['image'],
                            block=row['block'], adapter_alpha=row['alpha'], constant_alpha=closest['alpha'],
                            removal_proxy_gap=abs(row['dino_removal_proxy'] - closest['dino_removal_proxy']),
                            adapter_outside_mse=row['outside_roi_mse'], constant_outside_mse=closest['outside_roi_mse'],
                            adapter_cls=row['dino_cls_similarity'], constant_cls=closest['dino_cls_similarity'],
                            adapter_identity=row['identity_similarity'], constant_identity=closest['identity_similarity'],
                            adapter_glasses_prob=row['glasses_prob'], constant_glasses_prob=closest['glasses_prob']))
    save_json(root / 'matched_effect_comparisons.json', matched)
    print(f'metrics.json: {len(rows)} rows; matched_effect_comparisons.json: {len(matched)} pairs', flush=True)
