"""Train one invertible adapter per block using already saved activations."""
from pathlib import Path
import random
import torch
from .adapter import DinoAdapter
from .data import load_dataset, sample_tensors
from .runtime import save_json, signature, require_compatible, selected_blocks


@torch.no_grad()
def channel_statistics(root, samples, block):
    count, total, squares = 0, None, None
    for sample in samples:
        h, _ = sample_tensors(root, sample, block)
        h = h.double()
        total = h.sum(0) if total is None else total + h.sum(0)
        squares = h.square().sum(0) if squares is None else squares + h.square().sum(0)
        count += len(h)
    mean = total / count
    return mean.float(), (squares / count - mean.square()).clamp_min(1e-6).sqrt().float()


@torch.no_grad()
def validation(adapter, root, samples, block, chunk, device):
    total, count, inverse_max = 0., 0, 0.
    adapter.eval()
    for sample in samples:
        h, y = sample_tensors(root, sample, block)
        for start in range(0, len(h), chunk):
            x, target = h[start:start+chunk].to(device), y[start:start+chunk].to(device)
            z, r = adapter.encode(x)
            total += float((z - target).square().sum())
            count += target.numel()
            inverse_max = max(inverse_max, float((adapter.decode(z, r) - x).abs().max()))
    return dict(alignment_mse=total / count, inverse_max_abs=inverse_max)


def train(config, dataset, output, device):
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    requested = data['blocks'] if config['blocks'] == 'all' else config['blocks']
    if not requested or any(b not in data['blocks'] for b in requested):
        raise ValueError('Requested block is missing from saved data')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    settings = config['training']
    if settings['epochs'] < 1 or settings['token_batch'] < 1 or settings['learning_rate'] <= 0:
        raise ValueError('Invalid training settings')
    train_samples = [s for s in data['samples'] if s['split'] == 'train']
    val_samples = [s for s in data['samples'] if s['split'] == 'val']
    save_json(root / 'config.json', config)
    for block in requested:
        seed = settings['seed'] + block
        torch.manual_seed(seed)
        rng = random.Random(seed)
        token_rng = torch.Generator().manual_seed(seed)
        h, y = sample_tensors(dataset, train_samples[0], block)
        adapter = DinoAdapter(channels=h.shape[-1], z_dim=y.shape[-1],
                              hidden=settings['hidden'], layers=settings['layers']).to(device)
        center, spread = channel_statistics(dataset, train_samples, block)
        adapter.center.copy_(center.to(device))
        adapter.spread.copy_(spread.to(device))
        optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'])
        history, best = [], float('inf')
        for epoch in range(settings['epochs']):
            adapter.train()
            order = list(train_samples)
            rng.shuffle(order)
            train_sum, count = 0., 0
            for sample in order:
                h, y = sample_tensors(dataset, sample, block)
                indices = torch.randperm(len(h), generator=token_rng)
                for subset in indices.split(settings['token_batch']):
                    x, target = h[subset].to(device), y[subset].to(device)
                    loss = (adapter(x) - target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite alignment loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    train_sum += float(loss.detach()) * len(subset)
                    count += len(subset)
            metrics = validation(adapter, dataset, val_samples, block, settings['token_batch'], device)
            history.append(dict(epoch=epoch + 1, train_mse=train_sum / count, **metrics))
            print(f'block={block} epoch={epoch+1} train={train_sum/count:.6g} val={metrics}', flush=True)
            if metrics['alignment_mse'] < best:
                best = metrics['alignment_mse']
                torch.save(dict(version=1, block=block, step=config['step'], config=config,
                                grid=data['grid'], spec=adapter.spec, signature=signature(config),
                                state={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                                epoch=epoch+1, validation=metrics,
                                train_pair_ids=sorted({s['pair_id'] for s in train_samples})),
                           root / f'block_{block}.pt')
            save_json(root / f'block_{block}_history.json', history)
        del adapter, optimizer
