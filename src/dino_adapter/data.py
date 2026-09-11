"""Artifact contract, pairing checks and optional GPU collection (explicit CLI only)."""
import json
from contextlib import ExitStack
from pathlib import Path
import torch
from .hooks import ImageBlockHook
from .runtime import generate, load_pipeline, save_json, selected_blocks, signature


def validate_pairs(rows):
    ids, splits = set(), set()
    for row in rows:
        if not row['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['id']):
            raise ValueError('Pair IDs must be safe filename components')
        if row['id'] in ids:
            raise ValueError('Duplicate pair ID')
        ids.add(row['id'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        splits.add(row['split'])
        if not row['positive'] or not row['negative'] or not isinstance(row['seed'], int):
            raise ValueError('Need positive/negative prompts and integer seed')
    if not {'train', 'val'}.issubset(splits):
        raise ValueError('Need disjoint train and val pairs')
    # Same seed can leak source identity even if prompt/ID changes.
    seed_splits = {}
    for row in rows:
        if row['seed'] in seed_splits and seed_splits[row['seed']] != row['split']:
            raise ValueError('A seed cannot occur in different splits')
        seed_splits[row['seed']] = row['split']


def collect(config, pairs_path, output, device):
    """Run only on explicit request on a GPU server; never called by training/steering."""
    from .features import DinoFeatures
    pairs = [json.loads(line) for line in Path(pairs_path).read_text().splitlines() if line.strip()]
    validate_pairs(pairs)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    pipe = load_pipeline(config, device)
    blocks = selected_blocks(config, len(pipe.transformer.transformer_blocks))
    grid = (config['height'] // (2 * pipe.vae_scale_factor), config['width'] // (2 * pipe.vae_scale_factor))
    # Sequential stages avoid keeping DINO on GPU alongside FLUX.
    samples = []
    for pair in pairs:
        for label, prompt_key in ((1, 'positive'), (0, 'negative')):
            sample_id = f"{pair['id']}_{label}"
            folder = root / sample_id
            folder.mkdir()
            with ExitStack() as stack:
                hooks = [stack.enter_context(ImageBlockHook(pipe.transformer, block, config['step'])) for block in blocks]
                class AllSteps:
                    def on_step_end(self, pipe, step, timestep, kwargs):
                        for hook in hooks:
                            hook.on_step_end(pipe, step, timestep, kwargs)
                        return kwargs
                image = generate(pipe, config, pair[prompt_key], pair['seed'], AllSteps())
            image.save(folder / 'image.png')
            paths = {}
            for block, hook in zip(blocks, hooks):
                h = hook.captured
                if h.shape[0] != 1 or h.shape[1] != grid[0] * grid[1]:
                    raise ValueError('Unexpected spatial image-token grid')
                path = folder / f'block_{block}.pt'
                torch.save(h[0].contiguous(), path)
                paths[str(block)] = str(path.relative_to(root))
            samples.append(dict(id=sample_id, pair_id=pair['id'], label=label,
                                split=pair['split'], seed=pair['seed'], prompt=pair[prompt_key],
                                image=str((folder / 'image.png').relative_to(root)), blocks=paths,
                                features=str((folder / 'features.pt').relative_to(root))))
            save_json(root / 'collection_progress.json', samples)
            del hooks, hook  # release module references before unloading FLUX
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from PIL import Image
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    for sample in samples:
        with Image.open(root / sample['image']) as image:
            patches, cls = dino(image, grid)
        torch.save(dict(patches=patches[0], cls=cls[0]), root / sample['features'])
    save_json(root / 'dataset.json', dict(version=1, config=config, signature=signature(config),
                                        grid=grid, blocks=blocks, samples=samples))


def load_dataset(root):
    root = Path(root)
    data = json.loads((root / 'dataset.json').read_text())
    seen, pairs = set(), {}
    for sample in data['samples']:
        if sample['id'] in seen or sample['label'] not in (0, 1):
            raise ValueError('Invalid/duplicate sample')
        seen.add(sample['id'])
        group = pairs.setdefault(sample['pair_id'], [])
        group.append(sample)
    for group in pairs.values():
        if len(group) != 2 or {s['label'] for s in group} != {0, 1} or len({s['split'] for s in group}) != 1:
            raise ValueError('Each pair must contain both labels in exactly one split')
        if len({s['seed'] for s in group}) != 1:
            raise ValueError('Paired prompts must share the generation seed')
    if not all(any(s['split'] == split for s in data['samples']) for split in ('train', 'val')):
        raise ValueError('Dataset needs train and val samples')
    seed_splits = {}
    for sample in data['samples']:
        if sample['seed'] in seed_splits and seed_splits[sample['seed']] != sample['split']:
            raise ValueError('Seed leakage between splits')
        seed_splits[sample['seed']] = sample['split']
    return data


def sample_tensors(root, sample, block):
    root = Path(root)
    h = torch.load(root / sample['blocks'][str(block)], map_location='cpu', weights_only=True).float()
    y = torch.load(root / sample['features'], map_location='cpu', weights_only=True)['patches'].float()
    if h.ndim != 2 or y.ndim != 2 or h.shape[0] != y.shape[0]:
        raise ValueError('Expected spatially aligned [N,C] activations and [N,D] patches')
    if not torch.isfinite(h).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite training tensor')
    return h, y
