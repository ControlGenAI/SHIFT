"""CLS extraction, train-only mean differences, and real-feature guidance in SHIFT."""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from .features import DinoFeatures
from .runtime import load_pipeline, selected_blocks, save_json, digest, generate, pipeline_dtypes
from .cls_noise import direction_from_payload, extract_noised, noised_statistics, StepwiseCLSDirection
from .cls_guidance import CLSActivationGuidance


def read_cls_config(path):
    config = json.loads(Path(path).read_text())
    pipeline_dtypes(config)
    if any(type(config[k]) is not int for k in ('width', 'height', 'dino_size', 'inference_steps')):
        raise ValueError('Image sizes and inference_steps must be integers')
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
            len(set(blocks)) != len(blocks) or any(type(b) is not int or b < 0 for b in blocks)):
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


def paired_mean(rows, vectors, allow_zero=False):
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
    if not allow_zero and direction.norm() < 1e-8:
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
    if payload.get('kind') == 'noised_one_step':
        statistics = noised_statistics(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(statistics, path)
        return
    statistics = paired_mean(payload['rows'], payload['cls'])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, **statistics, cls_signature=payload['cls_signature']), path)


def guidance_options(config):
    settings = config['cls_optimization']
    if type(settings.get('save_step_predictions', False)) is not bool:
        raise ValueError('save_step_predictions must be boolean')
    return dict(steps=settings['steps'], iterations=settings['iterations'],
                learning_rate=settings['learning_rate'], preservation_weight=settings['preservation_weight'],
                max_relative_rms=settings['max_relative_rms'], selection=settings.get('selection', 'best'),
                resolution=(config['height'], config['width']),
                optimization_space=settings.get('space', 'activation'),
                block_mode=settings.get('block_mode', 'independent'),
                correction_scaling=settings.get('correction_scaling', 'rms'),
                match_rms_adam=settings.get('match_rms_adam', False),
                decode_mode=settings.get('decode_mode', 'pipeline'),
                first_update_probe=settings.get('first_update_probe', []))


def make_guidance(config, dino, direction, block, alpha, prediction_callback=None, iteration_callback=None,
                  reference_rgb=None):
    settings = config['cls_optimization']
    objective = settings.get('objective', 'one_step')
    if objective == 'one_step':
        return CLSActivationGuidance(dino, direction, block, alpha=alpha,
                                    prediction_callback=prediction_callback, **guidance_options(config))
    if objective != 'final':
        raise ValueError('objective must be one_step or final')
    if isinstance(direction, StepwiseCLSDirection):
        raise ValueError('Noise-level directions require the one_step objective')
    if settings.get('decode_mode', 'pipeline') != 'pipeline':
        raise ValueError('Final objective requires pipeline decoding')
    if settings.get('first_update_probe'):
        raise ValueError('first_update_probe is supported by the one_step objective')
    if (settings.get('space') != 'activation' or settings.get('block_mode') != 'joint' or
            settings['max_relative_rms'] is not None or settings['preservation_weight'] != 0 or
            settings.get('selection') != 'last'):
        raise ValueError('Final objective requires joint activation, null RMS cap, zero activation penalty and last selection')
    saved = settings.get('save_iteration_predictions', [])
    if (not isinstance(saved, list) or any(type(i) is not int or i < 0 or i > settings['iterations'] for i in saved)
            or len(set(saved)) != len(saved)):
        raise ValueError('save_iteration_predictions must contain distinct iteration indices in 0..iterations')
    from .cls_trajectory import CLSTrajectoryGuidance
    return CLSTrajectoryGuidance(dino, direction, block, steps=settings['steps'], alpha=alpha,
        iterations=settings['iterations'], learning_rate=settings['learning_rate'],
        resolution=(config['height'], config['width']),
        image_preservation_weight=settings['image_preservation_weight'], edit_roi=settings.get('edit_roi'),
        inside_weight=settings.get('inside_weight', .05), view_scales=settings.get('view_scales', [1., .5]),
        dino_checkpointing=settings.get('dino_checkpointing', True),
        correction_scaling=settings.get('correction_scaling', 'rms'),
        match_rms_adam=settings.get('match_rms_adam', False),
        prediction_callback=prediction_callback, iteration_callback=iteration_callback, reference_rgb=reference_rgb)


def optimize(config, rows, direction_path, output, device):
    payload = torch.load(direction_path, map_location='cpu', weights_only=True)
    if cls_signature(config) != payload['cls_signature']:
        raise ValueError('CLS direction and guidance DINO model/preprocessing differ')
    direction_spec = direction_from_payload(payload, config)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty prompts with unique IDs')
    conditioning = config['cls_optimization'].get('conditioning', 'source')
    if conditioning not in ('source', 'paired_target'):
        raise ValueError('conditioning must be source or paired_target')
    if conditioning == 'paired_target' and config['cls_optimization'].get('objective') != 'final':
        raise ValueError('paired_target conditioning requires the final-image objective')
    for row in rows:
        if row['pair_id'] in payload['train_pair_ids'] or row['seed'] in payload.get('train_seeds', []):
            raise ValueError('Use held-out pairs and seeds for comparisons')
        if not row['prompt'] or not isinstance(row['seed'], int):
            raise ValueError('Need a prompt and integer seed')
        if not row['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['id']):
            raise ValueError('Unsafe output ID')
        if conditioning == 'paired_target' and (not isinstance(row.get('target_prompt'), str) or not row['target_prompt'].strip()):
            raise ValueError('paired_target requires an explicit target_prompt or a matching negative dataset row')
    settings = config['cls_optimization']
    space = settings.get('space', 'activation')
    if not settings['steps'] or any(s < 0 or s >= config['inference_steps'] for s in settings['steps']):
        raise ValueError('Invalid optimization timestep')
    # Validate optimization settings without model loading.
    options = guidance_options(config)
    joint = options['block_mode'] == 'joint'
    preflight = make_guidance(config, None, direction_spec, [0] if joint else 0, config['alphas'][0])
    preflight.validate_run(config['height'], config['width'], config['inference_steps'])
    model_dtype, vae_dtype = pipeline_dtypes(config)
    print('CLS optimization: ' + json.dumps(dict(**options, model_dtype=str(model_dtype),
        vae_dtype=str(vae_dtype), dino_dtype='torch.float32', direction_kind=payload.get('kind', 'clean_cls'))), flush=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'config.json', config)
    import diffusers, transformers
    source_root = Path(__file__).resolve().parents[2]
    source_files = ['src/models/flux.py', 'src/dino_adapter/cls_guidance.py',
                    'src/dino_adapter/cls_images.py', 'src/dino_adapter/cls_trajectory.py',
                    'src/dino_adapter/cls_experiment.py', 'src/dino_adapter/features.py',
                    'src/dino_adapter/runtime.py', 'src/dino_adapter/cls_noise.py']
    direction_details = dict(kind=payload.get('kind', 'clean_cls'))
    if isinstance(direction_spec, StepwiseCLSDirection):
        direction_details.update(estimation=payload['estimation'],
            prediction_signature=payload['prediction_signature'], steps=list(direction_spec.steps),
            sigmas=direction_spec.sigmas.tolist(), timesteps=direction_spec.timesteps.tolist(),
            norms=direction_spec.values.norm(dim=-1).tolist())
    save_json(root / 'provenance.json', dict(direction_sha256=digest(direction_path),
              direction_details=direction_details,
              cls_signature=payload['cls_signature'], torch_version=str(torch.__version__),
              diffusers_version=diffusers.__version__, transformers_version=transformers.__version__,
              source_sha256={name: digest(source_root / name) for name in source_files},
              model_dtype=str(model_dtype), vae_dtype=str(vae_dtype), dino_dtype='torch.float32',
              decode_mode=options['decode_mode'], torch_cuda_version=torch.version.cuda,
              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
              cudnn_deterministic=torch.backends.cudnn.deterministic,
              cudnn_benchmark=torch.backends.cudnn.benchmark,
              float32_matmul_precision=torch.get_float32_matmul_precision()))
    pipe = load_pipeline(config, device)
    if settings.get('gradient_checkpointing', True):
        pipe.transformer.enable_gradient_checkpointing()
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    blocks = selected_blocks(config, len(pipe.transformer.transformer_blocks)) if space == 'activation' else [0]
    conditions = [blocks] if joint else blocks
    entries = []
    for row in rows:
        baseline = generate(pipe, config, row['prompt'], row['seed'])
        baseline.save(root / f"{row['id']}_baseline.png")
        with torch.no_grad():
            _, baseline_cls = dino(baseline, None)
        entries.append(dict(sample_id=row['id'], mode='baseline', image=f"{row['id']}_baseline.png"))
        source_rgb = None
        if conditioning == 'paired_target':
            from torchvision.transforms.functional import pil_to_tensor
            source_rgb = pil_to_tensor(baseline.convert('RGB')).unsqueeze(0).float() / 255
            prompt_only = generate(pipe, config, row['target_prompt'], row['seed'])
            prompt_only.save(root / f"{row['id']}_target_prompt_only.png")
            entries.append(dict(sample_id=row['id'], mode='target_prompt_only',
                prompt=row['target_prompt'], image=f"{row['id']}_target_prompt_only.png"))
        for block in conditions:
            for index, alpha in enumerate(config['alphas']):
                block_label = 'joint_blocks' + '-'.join(map(str, block)) if joint else f'block{block}'
                objective = settings.get('objective', 'one_step')
                objective_label = '_final' if objective == 'final' else ''
                name = f"{row['id']}_{space}_{block_label}{objective_label}_alpha{index}"
                intervention = dict(block=None if joint or space != 'activation' else block,
                    blocks=block if joint else ([block] if space == 'activation' else []),
                    block_mode=options['block_mode'], objective=objective, conditioning=conditioning,
                    correction_scaling=options['correction_scaling'], match_rms_adam=options['match_rms_adam'],
                    decode_mode=options['decode_mode'],
                    direction_kind=payload.get('kind', 'clean_cls'),
                    generation_prompt=row['target_prompt'] if conditioning == 'paired_target' and alpha != 0 else row['prompt'])
                def save_prediction(step, stage, decoded, stem=name):
                    preview = pipe.image_processor.postprocess(decoded.detach(), output_type='pil')[0]
                    preview.save(root / f'{stem}_step{step}_{stage}.png')
                def save_iteration(iteration, decoded, stem=name):
                    if iteration in settings.get('save_iteration_predictions', []):
                        preview = pipe.image_processor.postprocess(decoded.detach(), output_type='pil')[0]
                        preview.save(root / f'{stem}_iteration{iteration}_final.png')
                        print(f'{stem}: saved final image at iteration {iteration}', flush=True)
                guidance = make_guidance(config, dino, direction_spec, block, alpha,
                    prediction_callback=save_prediction if settings.get('save_step_predictions', False) else None,
                    iteration_callback=save_iteration if objective == 'final' else None, reference_rgb=source_rgb)
                image = pipe(intervention['generation_prompt'], width=config['width'], height=config['height'],
                    num_inference_steps=config['inference_steps'], guidance_scale=config['guidance_scale'],
                    max_sequence_length=256, generator=torch.Generator('cpu').manual_seed(row['seed']),
                    structure_strength=0., txt_steering={'vector': None}, activation_guidance=guidance).images[0]
                image.save(root / f'{name}.png')
                torch.save(guidance.references, root / f'{name}_cls_targets.pt')
                _, final_cls = dino(image, None)
                direction = (payload['clean_reference_direction'] if isinstance(direction_spec, StepwiseCLSDirection)
                             else payload['direction']).cpu()
                # Post-generation check: does optimizing x0 CLS affect the final image?
                projected_removal = (float(((baseline_cls - final_cls) * direction).sum() / direction.square().sum())
                                     if torch.count_nonzero(direction) else None)
                import numpy as np
                delta = np.asarray(image, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
                metrics = dict(final_cls_removal_proxy=projected_removal,
                               removal_proxy_basis='clean_cls_mean_diff',
                               baseline_pixel_mae=float(abs(delta).mean()), baseline_pixel_max_abs=float(abs(delta).max()))
                if objective == 'final' and guidance.references:
                    reference = guidance.references[-1]
                    metrics.update(final_png_target_loss=float(.5 * (final_cls - reference['target_cls'][0]).square().sum(-1).mean()),
                        evaluated_to_png_cls_l2=float((final_cls - reference['selected_cls'][0]).norm(dim=-1).mean()))
                elif guidance.references and guidance.references[-1]['step'] == config['inference_steps'] - 1:
                    reference = guidance.references[-1]
                    metrics.update(final_png_target_loss=float(.5 * (final_cls - reference['target_cls']).square().sum(-1).mean()),
                        evaluated_to_png_cls_l2=float((final_cls - reference['selected_cls']).norm(dim=-1).mean()))
                save_json(root / f'{name}.json', dict(sample=row, **intervention,
                          alpha=alpha, space=space, logs=guidance.logs, **metrics))
                entries.append(dict(sample_id=row['id'], mode=space, **intervention,
                                    alpha=alpha, image=f'{name}.png', **metrics))
                save_json(root / 'generations.json', entries)
    save_json(root / 'generations.json', entries)


def audit_direction(features_path, direction_path, output):
    """Check held-out separation using cached CLS only; no model loading/training."""
    features = torch.load(features_path, map_location='cpu', weights_only=True)
    payload = torch.load(direction_path, map_location='cpu', weights_only=True)
    if features['cls_signature'] != payload['cls_signature']:
        raise ValueError('Features and direction DINO signatures differ')
    if features.get('kind') == 'noised_one_step' or payload.get('kind') == 'noised_one_step':
        from .cls_noise import audit_noised_direction
        report = audit_noised_direction(features, payload)
        report.update(direction_sha256=digest(direction_path), features_sha256=digest(features_path))
        path = Path(output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, report)
        print(json.dumps(report, indent=2), flush=True)
        return
    rows = features['rows']
    statistics = paired_mean(rows, features['cls'])
    direction = payload['direction'].float()
    if (statistics['train_pair_ids'] != payload['train_pair_ids'] or
            direction.shape != statistics['direction'].shape or
            not torch.allclose(direction, statistics['direction'], rtol=1e-4, atol=1e-6)):
        raise ValueError('Direction does not match the train mean difference in these features')
    vectors = F.normalize(features['cls'].float(), dim=-1)
    center = (statistics['mean_positive'] + statistics['mean_negative']) / 2
    scores = ((vectors - center) * direction).sum(-1)
    splits = {}
    for split in ('train', 'val', 'test'):
        positive = [i for i, r in enumerate(rows) if r['split'] == split and r['label'] == 1]
        negative = [i for i, r in enumerate(rows) if r['split'] == split and r['label'] == 0]
        if not positive or not negative:
            splits[split] = dict(n_positive=len(positive), n_negative=len(negative), auc=None)
            continue
        p, n = scores[positive], scores[negative]
        comparison = p[:, None] - n[None, :]
        pairs = {}
        for i in positive + negative:
            pairs.setdefault(rows[i]['pair_id'], {})[rows[i]['label']] = scores[i]
        margins = torch.stack([pair[1] - pair[0] for pair in pairs.values() if len(pair) == 2]) if any(
            len(pair) == 2 for pair in pairs.values()) else torch.empty(0)
        splits[split] = dict(n_positive=len(positive), n_negative=len(negative),
            auc=float(((comparison > 0).float() + .5 * (comparison == 0).float()).mean()),
            balanced_accuracy=float(.5 * ((p > 0).float().mean() + (n <= 0).float().mean())),
            n_complete_pairs=len(margins), paired_order_accuracy=float((margins > 0).float().mean()) if len(margins) else None,
            paired_margin_mean=float(margins.mean()) if len(margins) else None)
    report = dict(direction_sha256=digest(direction_path), features_sha256=digest(features_path),
        direction_norm=float(direction.norm()), splits=splits,
        note='Held-out separation is a necessary diagnostic, not evidence of successful image editing.')
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, report)
    print(json.dumps(report, indent=2), flush=True)


def held_out_prompts(samples, split, conditioning='source'):
    rows = [r.copy() for r in samples if r['split'] == split and r['label'] == 1]
    if conditioning == 'paired_target':
        for row in rows:
            matches = [r for r in samples if r['pair_id'] == row['pair_id'] and r['label'] == 0]
            if len(matches) != 1 or matches[0]['split'] != split or matches[0]['seed'] != row['seed']:
                raise ValueError('Need one negative prompt with matching pair, split and seed')
            row['target_prompt'] = matches[0]['prompt']
    return rows


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
    noised = sub.add_parser('extract-noised', help='Paired mean diff at each scheduler noise level, through one-step prediction')
    source = noised.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--dataset')
    noised.add_argument('--output', required=True)
    noised.add_argument('--resume', action='store_true', help='Reuse per-image CLS with identical inputs/config/code')
    mean = sub.add_parser('mean', help='Recompute paired mean from saved cls_features.pt, no model loading')
    mean.add_argument('--features', required=True)
    mean.add_argument('--output', required=True)
    audit = sub.add_parser('audit', help='Held-out CLS direction separation from cached features, without models')
    audit.add_argument('--features', required=True)
    audit.add_argument('--direction', required=True)
    audit.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate', help='Independent eyewear classifier and optional face identity on saved PNGs')
    evaluate.add_argument('--results', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--with-identity', action='store_true', help='Use the existing optional facenet-pytorch scorer')
    opt = sub.add_parser('optimize')
    source = opt.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompts')
    source.add_argument('--dataset', help='Use positive held-out prompts from existing dataset.json')
    opt.add_argument('--split', choices=['val', 'test'], default='test')
    opt.add_argument('--num-samples', type=int, default=1)
    opt.add_argument('--learning-rate', type=float, help='Override the configured Adam LR; saved in the run config')
    opt.add_argument('--direction', required=True)
    opt.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'mean':
        mean_from_features(args.features, args.output)
        return
    if args.command == 'audit':
        audit_direction(args.features, args.direction, args.output)
        return
    if args.command == 'evaluate':
        from .cls_evaluation import evaluate_saved
        evaluate_saved(args.results, args.output, args.device, args.with_identity)
        return
    config = read_cls_config(args.config)
    if args.command == 'optimize' and args.learning_rate is not None:
        config['cls_optimization']['learning_rate'] = args.learning_rate
    if args.command == 'extract':
        extract(config, args.manifest, args.dataset, args.output, args.device, args.cached_cls)
    elif args.command == 'extract-noised':
        extract_noised(config, args.manifest, args.dataset, args.output, args.device, args.resume)
    else:
        if args.num_samples < 1:
            raise ValueError('--num-samples must be positive')
        if args.prompts:
            rows = read_rows(args.prompts)
        else:
            data = json.loads((Path(args.dataset) / 'dataset.json').read_text())
            rows = held_out_prompts(data['samples'], args.split, config['cls_optimization'].get('conditioning', 'source'))
        optimize(config, rows[:args.num_samples], args.direction, args.output, args.device)


if __name__ == '__main__':
    main()
