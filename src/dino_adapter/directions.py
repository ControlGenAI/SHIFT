"""Paired train-only DINO direction and constant activation-shift controls."""
import math
from pathlib import Path
import torch
from .data import load_dataset, sample_tensors
from .runtime import require_compatible


def region_mask(grid, roi):
    """ROI is [left, top, right, bottom] in full-frame normalized coordinates."""
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('ROI must be a nonempty rectangle inside [0,1]')
    mask = torch.zeros(grid, dtype=torch.bool)
    mask[math.floor(y0 * grid[0]):math.ceil(y1 * grid[0]),
         math.floor(x0 * grid[1]):math.ceil(x1 * grid[1])] = True
    return mask.flatten()


def build_directions(config, dataset, output):
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    mask = region_mask(data['grid'], config['roi'])
    pairs = {}
    for sample in data['samples']:
        if sample['split'] == 'train':
            pairs.setdefault(sample['pair_id'], {})[sample['label']] = sample
    blocks = data['blocks'] if config['blocks'] == 'all' else config['blocks']
    if not blocks or any(b not in data['blocks'] for b in blocks):
        raise ValueError('Missing requested block')
    vectors = {}
    for block in blocks:
        delta_h, delta_z = None, None
        for pair in pairs.values():
            hp, yp = sample_tensors(dataset, pair[1], block)
            hn, yn = sample_tensors(dataset, pair[0], block)
            dh, dz = hp - hn, yp - yn
            delta_h = dh if delta_h is None else delta_h + dh
            delta_z = dz if delta_z is None else delta_z + dz
        delta_h, delta_z = delta_h / len(pairs), delta_z / len(pairs)
        delta_h[~mask], delta_z[~mask] = 0, 0
        if delta_h.norm() < 1e-8 or delta_z.norm() < 1e-8:
            raise ValueError('Degenerate paired direction; check prompts/data')
        # Keep mean-difference units: alpha=1 removes one average DINO displacement.
        vectors[str(block)] = dict(z=delta_z, h=delta_h)
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, config=config, grid=data['grid'], mask=mask,
                    train_pair_ids=sorted(pairs), vectors=vectors), path)
