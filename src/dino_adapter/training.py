"""Train one invertible adapter per block using already saved activations."""
from pathlib import Path
import random
import time
import torch
from .adapter import DinoAdapter
from .data import load_dataset, sample_tensors
from .runtime import save_json, signature, require_compatible


def load_block(root, samples, block):
    """Read a block's activations and targets once and keep them in RAM.

    Re-reading per epoch means tens of gigabytes of NFS traffic per block and
    makes the run IO bound; one split of one block is only a few GB.
    """
    return [sample_tensors(root, sample, block) for sample in samples]


@torch.no_grad()
def channel_statistics(cached):
    count, total, squares = 0, None, None
    for h, _ in cached:
        h = h.double()
        total = h.sum(0) if total is None else total + h.sum(0)
        squares = h.square().sum(0) if squares is None else squares + h.square().sum(0)
        count += len(h)
    mean = total / count
    return mean.float(), (squares / count - mean.square()).clamp_min(1e-6).sqrt().float()


@torch.no_grad()
def target_statistics(cached):
    """Train-set target mean and variance, for a baseline the loss can be read against."""
    count, total = 0, None
    for _, y in cached:
        total = y.double().sum(0) if total is None else total + y.double().sum(0)
        count += len(y)
    mean = (total / count).float()
    squares, count = 0.0, 0
    for _, y in cached:
        squares += float((y - mean).square().sum())
        count += y.numel()
    return mean, squares / count


@torch.no_grad()
def validation(adapter, cached, chunk, device, target_var):
    total, count, inverse_max, inverse_sq = 0., 0, 0., 0.
    cos_sum, cos_count = 0., 0
    adapter.eval()
    for h, y in cached:
        for start in range(0, len(h), chunk):
            x, target = h[start:start + chunk].to(device), y[start:start + chunk].to(device)
            z, r = adapter.encode(x)
            total += float((z - target).square().sum())
            count += target.numel()
            back = adapter.decode(z, r)
            inverse_max = max(inverse_max, float((back - x).abs().max()))
            inverse_sq += float((back - x).square().sum())
            cos_sum += float(torch.nn.functional.cosine_similarity(z, target, dim=-1).sum())
            cos_count += len(x)
    mse = total / count
    return dict(alignment_mse=mse,
                # Fraction of target variance explained; <=0 means the adapter is
                # no better than predicting the train-set mean DINO patch.
                alignment_r2=1.0 - mse / target_var if target_var > 0 else 0.0,
                alignment_cosine=cos_sum / cos_count,
                inverse_max_abs=inverse_max,
                inverse_rms=(inverse_sq / count) ** 0.5)


def train(config, dataset, output, device):
    data = load_dataset(dataset)
    require_compatible(config, data['config'])
    requested = data['blocks'] if config['blocks'] == 'all' else config['blocks']
    if not requested or any(b not in data['blocks'] for b in requested):
        raise ValueError('Requested block is missing from saved data')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    settings = config['training']
    if settings['epochs'] < 1 or settings['token_batch'] < 1 or settings['learning_rate'] <= 0:
        raise ValueError('Invalid training settings')
    train_samples = [s for s in data['samples'] if s['split'] == 'train']
    val_samples = [s for s in data['samples'] if s['split'] == 'val']
    save_json(root / 'config.json', config)
    for block in requested:
        checkpoint_path = root / f'block_{block}.pt'
        if checkpoint_path.exists():
            print(f'block={block} already trained, skipping', flush=True)
            continue
        seed = settings['seed'] + block
        torch.manual_seed(seed)
        rng = random.Random(seed)
        token_rng = torch.Generator().manual_seed(seed)
        started = time.time()
        cached_train = load_block(dataset, train_samples, block)
        cached_val = load_block(dataset, val_samples, block)
        channels, z_dim = cached_train[0][0].shape[-1], cached_train[0][1].shape[-1]
        adapter = DinoAdapter(channels=channels, z_dim=z_dim,
                              hidden=settings['hidden'], layers=settings['layers'],
                              scale_bound=settings.get('scale_bound', 1.5),
                              permute=settings.get('permute', True),
                              perm_seed=seed).to(device)
        center, spread = channel_statistics(cached_train)
        adapter.center.copy_(center.to(device))
        adapter.spread.copy_(spread.to(device))
        # Statistics come from train only; val/test never touch normalisation.
        _, train_target_var = target_statistics(cached_train)
        _, val_target_var = target_statistics(cached_val)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'])
        steps_per_epoch = sum(-(-len(h) // settings['token_batch']) for h, _ in cached_train)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, settings['epochs'] * steps_per_epoch)
        history, best = [], float('inf')
        baseline = validation(adapter, cached_val, settings['token_batch'], device, val_target_var)
        print(f'block={block} init val={baseline} (train target var={train_target_var:.3e})', flush=True)
        for epoch in range(settings['epochs']):
            adapter.train()
            order = list(range(len(cached_train)))
            rng.shuffle(order)
            train_sum, count = 0., 0
            for index in order:
                h, y = cached_train[index]
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
                    schedule.step()
                    train_sum += float(loss.detach()) * len(subset)
                    count += len(subset)
            metrics = validation(adapter, cached_val, settings['token_batch'], device, val_target_var)
            train_mse = train_sum / count
            history.append(dict(epoch=epoch + 1, train_mse=train_mse,
                                train_r2=1.0 - train_mse / train_target_var, **metrics))
            print(f'block={block} epoch={epoch+1} train_mse={train_mse:.6g} '
                  f"train_r2={history[-1]['train_r2']:.4f} val_mse={metrics['alignment_mse']:.6g} "
                  f"val_r2={metrics['alignment_r2']:.4f} cos={metrics['alignment_cosine']:.4f} "
                  f"inv_max={metrics['inverse_max_abs']:.3g}", flush=True)
            if metrics['alignment_mse'] < best:
                best = metrics['alignment_mse']
                torch.save(dict(version=1, block=block, step=config['step'], config=config,
                                grid=data['grid'], spec=adapter.spec, signature=signature(config),
                                state={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                                epoch=epoch + 1, validation=metrics,
                                train_target_var=train_target_var, val_target_var=val_target_var,
                                train_pair_ids=sorted({s['pair_id'] for s in train_samples})),
                           checkpoint_path)
            save_json(root / f'block_{block}_history.json',
                      dict(block=block, init_validation=baseline, history=history,
                           train_target_var=train_target_var, val_target_var=val_target_var,
                           best_val_mse=best, seconds=time.time() - started,
                           spec=adapter.spec))
        print(f'block={block} done in {time.time() - started:.0f}s best_val_mse={best:.6g}', flush=True)
        del adapter, optimizer, cached_train, cached_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
