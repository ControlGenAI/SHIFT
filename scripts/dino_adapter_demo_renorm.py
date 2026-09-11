"""All-block step-0 norm-preserving steering demo (reproduces figures/dino_adapter_steering).

Builds on adapter-z directions (z / z_mean / h / h_mean). Each generation edits every
double-stream block at diffusion step 0 via MultiImageBlockHook + Renorm*Edit.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.dino_adapter.data import load_dataset
from src.dino_adapter.hooks import MultiImageBlockHook
from src.dino_adapter.runtime import generate, load_adapter, load_pipeline, read_config
from src.dino_adapter.steering import RenormAdapterEdit, RenormImageEdit


def sheet(paths, labels, out_path, tile=224):
    img = Image.new('RGB', (tile * len(paths), tile + 20), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    for i, (p, label) in enumerate(zip(paths, labels)):
        with Image.open(p) as frame:
            img.paste(frame.convert('RGB').resize((tile, tile), Image.LANCZOS), (i * tile, 20))
        draw.text((i * tile + 4, 4), label, fill=(255, 220, 0))
    img.save(out_path)


VARIANTS = {
    'adapter_pertoken': ('adapter', 'z'),
    'adapter_mean': ('adapter', 'z_mean'),
    'image_pertoken': ('image', 'h'),
    'image_mean': ('image', 'h_mean'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dino_adapter.json')
    parser.add_argument('--dataset', default='out/dataset200')
    parser.add_argument('--adapters', default='out/adapters200')
    parser.add_argument('--directions', default='out/dirv3_full100.pt',
                        help='Payload from: python -m src.dino_adapter directions-adapter-z ...')
    parser.add_argument('--output', default='out/demo_renorm')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default='test')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    parser.add_argument('--blocks', type=int, nargs='*')
    parser.add_argument('--no-renorm', action='store_true')
    args = parser.parse_args()

    config = read_config(args.config)
    data = load_dataset(args.dataset)
    directions = torch.load(args.directions, map_location='cpu', weights_only=True)
    if directions.get('direction_space') != 'adapter_z':
        raise ValueError('Expected directions-adapter-z payload (direction_space=adapter_z)')
    blocks = args.blocks if args.blocks else (
        data['blocks'] if config['blocks'] == 'all' else config['blocks'])
    tokens = data['grid'][0] * data['grid'][1]
    samples = [s for s in data['samples'] if s['split'] == args.split and s['label'] == 1][:args.n]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    renorm = not args.no_renorm

    adapters = {b: load_adapter(Path(args.adapters) / f'block_{b}.pt', args.device)[0] for b in blocks}
    pipe = load_pipeline(config, args.device)
    entries = []

    for sample in samples:
        folder = root / sample['id']
        folder.mkdir(exist_ok=True)
        base = folder / 'baseline.png'
        if not base.exists():
            generate(pipe, config, sample['prompt'], sample['seed']).save(base)
        print(f'{sample["id"]}: baseline', flush=True)

        for variant in args.variants:
            kind, key = VARIANTS[variant]
            for alpha in args.alphas:
                path = folder / f'{variant}_a{alpha:g}.png'
                if path.exists():
                    continue
                edits = []
                for b in blocks:
                    vec = directions['vectors'][str(b)][key]
                    edits.append((b, RenormAdapterEdit(adapters[b], vec, alpha, renorm)
                                  if kind == 'adapter'
                                  else RenormImageEdit(vec, alpha, renorm)))
                try:
                    with MultiImageBlockHook(pipe.transformer, edits, config['step'],
                                             expected_tokens=tokens) as hook:
                        generate(pipe, config, sample['prompt'], sample['seed'], hook).save(path)
                    stats = edits[len(edits) // 2][1].stats
                    entries.append(dict(sample_id=sample['id'], variant=variant, alpha=alpha,
                                        renorm=renorm, image=str(path.relative_to(root)), **stats))
                    print(f'  {variant} a={alpha:g} edit_rms={stats.get("edit_rms", 0):.4g}', flush=True)
                except Exception as error:
                    entries.append(dict(sample_id=sample['id'], variant=variant, alpha=alpha,
                                        renorm=renorm, image=None, failure=str(error)))
                    print(f'  {variant} a={alpha:g} FAILED: {error}', flush=True)
            present = [(a, folder / f'{variant}_a{a:g}.png') for a in args.alphas
                       if (folder / f'{variant}_a{a:g}.png').exists()]
            if present:
                sheet([base] + [p for _, p in present],
                      ['baseline'] + [f'a={a:g}' for a, _ in present],
                      folder / f'strip_{variant}.png')
        for alpha in args.alphas:
            present = [(v, folder / f'{v}_a{alpha:g}.png') for v in args.variants
                       if (folder / f'{v}_a{alpha:g}.png').exists()]
            if present:
                sheet([base] + [p for _, p in present],
                      ['baseline'] + [v.replace('_', ' ') for v, _ in present],
                      folder / f'compare_a{alpha:g}.png')
        print(f'{sample["id"]} done', flush=True)

    (root / 'generations.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(f'wrote {len(entries)} steered images under {root}', flush=True)


if __name__ == '__main__':
    main()
