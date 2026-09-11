"""Paired train-only directions: DINO features, or adapter z = F(h).z."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from .data import load_dataset, sample_tensors
from .runtime import load_adapter, require_compatible


def region_mask(grid, roi):
    """ROI is [left, top, right, bottom] in full-frame normalized coordinates."""
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('ROI must be a nonempty rectangle inside [0,1]')
    mask = torch.zeros(grid, dtype=torch.bool)
    mask[math.floor(y0 * grid[0]):math.ceil(y1 * grid[0]),
         math.floor(x0 * grid[1]):math.ceil(x1 * grid[1])] = True
    return mask.flatten()


def _train_pairs(data, max_pairs=None):
    pairs = {}
    for sample in data['samples']:
        if sample['split'] == 'train':
            pairs.setdefault(sample['pair_id'], {})[sample['label']] = sample
    pairs = {k: v for k, v in sorted(pairs.items()) if 0 in v and 1 in v}
    if not pairs:
        raise ValueError('No train pairs to build a direction from')
    if max_pairs is not None:
        pairs = dict(list(pairs.items())[:max_pairs])
    return pairs


def build_directions(config, dataset, output):
    """Legacy: z from DINO patch differences, h from post-block activations."""
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    mask = region_mask(data['grid'], config['roi'])
    pairs = _train_pairs(data)
    blocks = data['blocks'] if config['blocks'] == 'all' else config['blocks']
    if not blocks or any(b not in data['blocks'] for b in blocks):
        raise ValueError('Missing requested block')
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)

    # The DINO direction comes from the final images, so it does not depend on
    # the block; only the activation direction does.
    delta_z = None
    for pair in pairs.values():
        _, yp = sample_tensors(dataset, pair[1], blocks[0])
        _, yn = sample_tensors(dataset, pair[0], blocks[0])
        delta_z = (yp - yn) if delta_z is None else delta_z + (yp - yn)
    delta_z = delta_z / len(pairs)
    delta_z[~mask] = 0
    if delta_z.norm() < 1e-8:
        raise ValueError('Degenerate paired DINO direction; check prompts/data')

    vectors = {}
    for block in blocks:
        delta_h = None
        for pair in pairs.values():
            hp, _ = sample_tensors(dataset, pair[1], block)
            hn, _ = sample_tensors(dataset, pair[0], block)
            delta_h = (hp - hn) if delta_h is None else delta_h + (hp - hn)
        delta_h = delta_h / len(pairs)
        delta_h[~mask] = 0
        if delta_h.norm() < 1e-8:
            raise ValueError(f'Degenerate paired activation direction for block {block}')
        # Keep mean-difference units: alpha=1 removes one average DINO displacement.
        vectors[str(block)] = dict(z=delta_z.clone(), h=delta_h)
        print(f'block={block} |delta_h|={float(delta_h.norm()):.4g} '
              f'rms={float(delta_h.square().mean().sqrt()):.4g}', flush=True)
    print(f'|delta_z|={float(delta_z.norm()):.4g} over {len(pairs)} train pairs', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, config=config, grid=data['grid'], mask=mask,
                    direction_space='dino_features', train_pair_ids=sorted(pairs),
                    vectors=vectors), path)


def build_adapter_z_directions(config, dataset, adapters, output, device='cuda:0',
                               roi='full', max_pairs=100, workers=16, blocks=None):
    """Directions in adapter z: mean(F(h_with).z - F(h_without).z) over train pairs.

    Also stores a single mean vector broadcast to every token (z_mean / h_mean).
    Unit variants rescale so mean per-token norm equals the block's mean |z_token|.
    """
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    grid = data['grid']
    blocks = blocks if blocks is not None else (
        data['blocks'] if config['blocks'] == 'all' else config['blocks'])
    if not blocks or any(b not in data['blocks'] for b in blocks):
        raise ValueError('Missing requested block')
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    pairs = _train_pairs(data, max_pairs=max_pairs)
    pair_list = list(pairs.values())
    mask = (torch.ones(grid[0] * grid[1], dtype=torch.bool) if roi == 'full'
            else region_mask(grid, config['roi']))
    print(f'{len(pair_list)} train pairs, mask {int(mask.sum())}/{mask.numel()} tokens, '
          f'{len(blocks)} blocks', flush=True)

    vectors, summary = {}, []
    pool = ThreadPoolExecutor(max_workers=workers)
    for block in blocks:
        adapter, _ = load_adapter(Path(adapters) / f'block_{block}.pt', device)

        def read(pair, block=block):
            hp, _ = sample_tensors(dataset, pair[1], block)
            hn, _ = sample_tensors(dataset, pair[0], block)
            return hp, hn

        dz = dh = None
        z_norm_sum, count = 0.0, 0
        with torch.no_grad():
            for hp, hn in pool.map(read, pair_list):
                hp, hn = hp.to(device), hn.to(device)
                zp, _ = adapter.encode(hp)
                zn, _ = adapter.encode(hn)
                d = (zp - zn).cpu()
                dz = d if dz is None else dz + d
                d = (hp - hn).cpu()
                dh = d if dh is None else dh + d
                z_norm_sum += float(zp.norm(dim=-1).mean())
                count += 1
        dz, dh = dz / count, dh / count
        dz[~mask] = 0
        dh[~mask] = 0

        dz_mean = torch.zeros_like(dz)
        dh_mean = torch.zeros_like(dh)
        dz_mean[mask] = dz[mask].mean(dim=0, keepdim=True)
        dh_mean[mask] = dh[mask].mean(dim=0, keepdim=True)
        z_token_norm = z_norm_sum / count

        def renormalize(vector):
            per_token = float(vector[mask].norm(dim=-1).mean())
            if per_token < 1e-12:
                raise ValueError(f'Degenerate direction for block {block}')
            return vector * (z_token_norm / per_token), per_token

        dz_unit, dz_per_token = renormalize(dz)
        dz_mean_unit, dz_mean_per_token = renormalize(dz_mean)
        vectors[str(block)] = dict(
            z=dz, z_mean=dz_mean, z_unit=dz_unit, z_mean_unit=dz_mean_unit,
            h=dh, h_mean=dh_mean)
        row = dict(block=block, z_token_norm=z_token_norm,
                   dz_per_token=dz_per_token, dz_mean_per_token=dz_mean_per_token,
                   dz_over_z=dz_per_token / z_token_norm,
                   dz_mean_over_z=dz_mean_per_token / z_token_norm,
                   mean_fraction_of_per_token=dz_mean_per_token / dz_per_token,
                   dh_norm=float(dh.norm()), dh_mean_norm=float(dh_mean.norm()))
        summary.append(row)
        print(f"block={block:2d} |z_tok|={z_token_norm:.4f} "
              f"dz/tok={dz_per_token:.4f} ({row['dz_over_z']*100:.1f}% of z) "
              f"dz_mean/tok={dz_mean_per_token:.4f} ({row['dz_mean_over_z']*100:.1f}% of z) "
              f"mean/pertoken={row['mean_fraction_of_per_token']:.2f}", flush=True)
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pool.shutdown()

    saved = {**config, 'direction_roi': roi}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=3, config=saved, grid=grid, mask=mask,
                    direction_space='adapter_z', direction_roi=roi,
                    n_pairs=len(pair_list), summary=summary,
                    train_pair_ids=sorted(pairs), vectors=vectors), path)
    Path(str(path) + '.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('wrote', path, flush=True)
