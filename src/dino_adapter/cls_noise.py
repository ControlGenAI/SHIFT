"""Paired DINO CLS directions from VAE -> flow noise -> one-step predictions.

No learned CLS predictor is used. A direction is matched to an actual scheduler
step, sigma, timestep, model precision and image decoding convention.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from PIL import Image

from .cls_images import decode_final, one_step_latents
from .features import DinoFeatures
from .runtime import digest, load_pipeline, pipeline_dtypes, save_json


def prediction_signature(config):
    model_dtype, vae_dtype = pipeline_dtypes(config)
    signature = {k: config.get(k) for k in ('model', 'model_revision', 'width', 'height',
                                          'inference_steps', 'guidance_scale')}
    return dict(**signature, model_dtype=str(model_dtype), vae_dtype=str(vae_dtype),
        decode_mode=config['cls_optimization'].get('decode_mode', 'pipeline'),
        max_sequence_length=256, vae_encoding='posterior_mode', noise_mixing='scheduler_scale_noise')


class StepwiseCLSDirection:
    def __init__(self, payload):
        self.values = torch.as_tensor(payload['directions']).detach().float().cpu()
        self.steps = tuple(payload['steps'])
        self.sigmas = torch.as_tensor(payload['sigmas']).float().cpu()
        self.timesteps = torch.as_tensor(payload['timesteps']).float().cpu()
        self.signature = payload['prediction_signature']
        if (not self.steps or any(type(s) is not int or s < 0 for s in self.steps) or
                len(set(self.steps)) != len(self.steps) or self.values.ndim != 2 or
                self.values.shape[0] != len(self.steps) or self.values.shape[1] < 1 or
                self.sigmas.shape != (len(self.steps),) or self.timesteps.shape != self.sigmas.shape or
                not all(torch.isfinite(x).all() for x in (self.values, self.sigmas, self.timesteps)) or
                not ((self.sigmas >= 0) & (self.sigmas <= 1)).all()):
            raise ValueError('Invalid stepwise CLS direction table')

    def at(self, pipe, step, sigma, timestep):
        if step not in self.steps:
            raise ValueError(f'No CLS direction for step {step}')
        index = self.steps.index(step)
        if (not math.isclose(float(sigma), float(self.sigmas[index]), rel_tol=0, abs_tol=1e-6) or
                not math.isclose(float(timestep), float(self.timesteps[index]), rel_tol=0, abs_tol=1e-4)):
            raise ValueError('CLS direction sigma/timestep does not match the running scheduler')
        if (str(pipe.transformer.dtype) != self.signature['model_dtype'] or
                str(pipe.vae.dtype) != self.signature['vae_dtype']):
            raise ValueError('CLS direction model/VAE dtype does not match the running pipeline')
        return self.values[index]


def direction_from_payload(payload, config):
    if payload.get('kind') != 'noised_one_step':
        return payload['direction']
    if config['cls_optimization'].get('objective', 'one_step') != 'one_step':
        raise ValueError('Noise-level directions require the one_step objective')
    if payload['prediction_signature'] != prediction_signature(config):
        raise ValueError('Noised CLS direction and optimization prediction settings differ')
    return StepwiseCLSDirection(payload)


def noise_seed(base_seed, pair_id, repeat):
    # Stable across row ordering, split selection and Python processes.
    key = json.dumps([base_seed, pair_id, repeat], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)


def encode_image(pipe, image, config, device):
    pixels = pipe.image_processor.preprocess(image, height=config['height'], width=config['width'])
    posterior = pipe.vae.encode(pixels.to(device=device, dtype=pipe.vae.dtype)).latent_dist
    clean = (posterior.mode() - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    b, c, h, w = clean.shape
    return pipe._pack_latents(clean, b, c, h, w).to(pipe.transformer.dtype)


def predict_noised_cls(pipe, dino, clean, noise, scheduler, step, conditioning, image_ids, config):
    if scheduler.begin_index is not None or scheduler.step_index is not None:
        raise ValueError('Extraction needs a fresh scheduler; scale_noise otherwise reuses its current index')
    t = scheduler.timesteps[step]
    noisy = scheduler.scale_noise(clean, t.expand(clean.shape[0]), noise=noise)
    embeds, pooled, text_ids = conditioning
    guidance = (torch.full((clean.shape[0],), config['guidance_scale'], device=clean.device, dtype=torch.float32)
                if pipe.transformer.config.guidance_embeds else None)
    velocity = pipe.transformer(hidden_states=noisy, timestep=t.expand(clean.shape[0]).to(noisy.dtype) / 1000,
        guidance=guidance, pooled_projections=pooled, encoder_hidden_states=embeds,
        txt_ids=text_ids, img_ids=image_ids, joint_attention_kwargs=None, return_dict=False)[0]
    if not torch.isfinite(velocity).all():
        raise RuntimeError('Nonfinite velocity in noised CLS extraction')
    sigma = scheduler.sigmas[step].to(device=clean.device, dtype=torch.float32)
    decoded, rgb = decode_final(pipe, one_step_latents(noisy, velocity, sigma),
                               (config['height'], config['width']))
    return dino.cls_from_rgb(rgb), decoded


def noised_statistics(features):
    from .cls_experiment import paired_mean
    rows, values = features['rows'], features['cls']
    if (values.ndim != 4 or values.shape[:2] != (len(rows), len(features['steps'])) or
            values.shape[2] < 1 or not torch.isfinite(values).all()):
        raise ValueError('Expected finite noised CLS [records, steps, repeats, features]')
    directions, means_positive, means_negative, delta_norms = [], [], [], []
    for index in range(len(features['steps'])):
        # Average normalized observations, not re-normalized averages of repeats.
        repeats = [paired_mean(rows, values[:, index, r], allow_zero=True) for r in range(values.shape[2])]
        directions.append(torch.stack([r['direction'] for r in repeats]).mean(0))
        means_positive.append(torch.stack([r['mean_positive'] for r in repeats]).mean(0))
        means_negative.append(torch.stack([r['mean_negative'] for r in repeats]).mean(0))
        delta_norms.append(sum(r['paired_delta_norm_mean'] for r in repeats) / len(repeats))
    clean = paired_mean(rows, features['clean_cls'], allow_zero=True)
    return dict(version=2, kind='noised_one_step', steps=features['steps'], sigmas=features['sigmas'],
        timesteps=features['timesteps'], cls_signature=features['cls_signature'],
        prediction_signature=features['prediction_signature'], estimation=features['estimation'],
        directions=torch.stack(directions), mean_positive=torch.stack(means_positive),
        mean_negative=torch.stack(means_negative), paired_delta_norm_mean=delta_norms,
        clean_reference_direction=clean['direction'], train_pair_ids=clean['train_pair_ids'],
        train_seeds=clean['train_seeds'], n_train_pairs=clean['n_train_pairs'])


def extraction_settings(config):
    settings = config['cls_direction_estimation']
    steps, repeats = settings['steps'], settings['noise_repeats']
    if (not isinstance(steps, list) or not steps or len(set(steps)) != len(steps) or
            any(type(s) is not int or not 0 <= s < config['inference_steps'] for s in steps) or
            type(repeats) is not int or repeats < 1 or type(settings['noise_seed']) is not int):
        raise ValueError('Need valid extraction steps, positive noise_repeats and an integer noise_seed')
    if settings['conditioning'] not in ('paired_prompts', 'shared_prompt'):
        raise ValueError('Extraction conditioning must be paired_prompts or shared_prompt')
    if settings['conditioning'] == 'shared_prompt' and not isinstance(settings.get('prompt'), str):
        raise ValueError('shared_prompt requires an explicit prompt string, possibly empty')
    if type(settings.get('preview_pairs', 0)) is not int or settings.get('preview_pairs', 0) < 0:
        raise ValueError('preview_pairs must be a nonnegative integer')
    if config['cls_optimization'].get('decode_mode', 'pipeline') != 'pipeline':
        raise ValueError('Noised extraction uses pipeline decoding')
    prediction_signature(config)
    return settings


def audit_noised_direction(features, payload):
    if features.get('kind') != 'noised_one_step' or payload.get('kind') != 'noised_one_step':
        raise ValueError('Both features and direction must use noised_one_step')
    statistics = noised_statistics(features)
    StepwiseCLSDirection(payload)
    for key in ('steps', 'prediction_signature', 'estimation', 'train_pair_ids', 'train_seeds'):
        if statistics[key] != payload[key]:
            raise ValueError(f'Noised CLS direction metadata differs: {key}')
    for key in ('directions', 'sigmas', 'timesteps', 'mean_positive', 'mean_negative', 'clean_reference_direction'):
        if statistics[key].shape != payload[key].shape or not torch.allclose(statistics[key], payload[key], rtol=1e-4, atol=1e-6):
            raise ValueError(f'Noised CLS direction does not match cached features: {key}')
    observations = F.normalize(features['cls'].float(), dim=-1).mean(2)
    levels = []
    for index, step in enumerate(payload['steps']):
        direction = statistics['directions'][index]
        center = (statistics['mean_positive'][index] + statistics['mean_negative'][index]) / 2
        scores = ((observations[:, index] - center) * direction).sum(-1)
        splits = {}
        for split in ('train', 'val', 'test'):
            p = scores[[i for i, r in enumerate(features['rows']) if r['split'] == split and r['label'] == 1]]
            n = scores[[i for i, r in enumerate(features['rows']) if r['split'] == split and r['label'] == 0]]
            comparison = p[:, None] - n[None, :]
            splits[split] = dict(n_positive=len(p), n_negative=len(n),
                auc=float(((comparison > 0).float() + .5 * (comparison == 0).float()).mean()) if p.numel() and n.numel() else None)
        levels.append(dict(step=step, sigma=float(payload['sigmas'][index]), direction_norm=float(direction.norm()), splits=splits))
    return dict(kind='noised_one_step', levels=levels,
        note='Scores average repeated normalized CLS per image. Separation is not evidence of successful editing.')


@torch.no_grad()
def extract_noised(config, manifest, dataset, output, device, resume=False):
    from src.models.flux import prepare_flow_schedule
    from .cls_experiment import cls_signature, read_rows, validate_records
    settings = extraction_settings(config)
    if dataset is not None:
        image_root = Path(dataset)
        rows = json.loads((image_root / 'dataset.json').read_text())['samples']
    else:
        image_root = Path(manifest).parent
        rows = read_rows(manifest)
    validate_records(rows)
    if (len({r['id'] for r in rows}) != len(rows) or
            any(not isinstance(r['id'], str) or re.fullmatch(r'[A-Za-z0-9_-]+', r['id']) is None for r in rows)):
        raise ValueError('Need distinct safe image IDs')
    inputs = []
    for row in rows:
        if settings['conditioning'] == 'paired_prompts' and not isinstance(row.get('prompt'), str):
            raise ValueError('paired_prompts needs a prompt for each image')
        path = image_root / row['image']
        with Image.open(path) as image:
            if image.size != (config['width'], config['height']):
                raise ValueError(f'Image resolution differs from prediction config: {path}')
        inputs.append(dict(row=row, image_sha256=digest(path)))
    source_root = Path(__file__).resolve().parents[2]
    sources = ['src/dino_adapter/cls_noise.py', 'src/dino_adapter/cls_images.py',
               'src/dino_adapter/features.py', 'src/dino_adapter/runtime.py',
               'src/dino_adapter/cls_experiment.py', 'src/models/flux.py']
    import diffusers, transformers
    environment = dict(torch_version=str(torch.__version__), diffusers_version=diffusers.__version__,
        transformers_version=transformers.__version__, device=str(device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)
    contract = dict(config=config, inputs=inputs, environment=environment,
                    source_sha256={p: digest(source_root / p) for p in sources})
    root = Path(output)
    if root.exists():
        if not resume or json.loads((root / 'extraction.json').read_text()) != contract:
            raise ValueError('Use a new output, or --resume with identical config, inputs and source code')
    else:
        root.mkdir(parents=True)
        save_json(root / 'extraction.json', contract)
    (root / 'samples').mkdir(exist_ok=True)
    (root / 'previews').mkdir(exist_ok=True)
    pipe = load_pipeline(config, device)
    dino = DinoFeatures(config['dino_model'], config['dino_size'], device, config.get('dino_revision'))
    scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    seq_len = (config['height'] // (2 * pipe.vae_scale_factor)) * (config['width'] // (2 * pipe.vae_scale_factor))
    prepare_flow_schedule(scheduler, config['inference_steps'], device, seq_len)
    steps = settings['steps']
    schedule = dict(steps=steps, sigmas=scheduler.sigmas[steps].cpu(), timesteps=scheduler.timesteps[steps].cpu())
    save_json(root / 'provenance.json', dict(**environment,
        prediction_signature=prediction_signature(config), source_sha256=contract['source_sha256'],
        steps=steps, sigmas=schedule['sigmas'].tolist(), timesteps=schedule['timesteps'].tolist(),
        transformer_dtype=str(pipe.transformer.dtype), vae_dtype=str(pipe.vae.dtype), dino_dtype=str(dino.model.dtype)))
    preview_pairs = list(dict.fromkeys(r['pair_id'] for r in rows))[:settings.get('preview_pairs', 0)]
    all_cls, clean_cls = [], []
    for row in rows:
        cached = root / 'samples' / f"{row['id']}.pt"
        if cached.exists():
            sample = torch.load(cached, map_location='cpu', weights_only=True)
        else:
            with Image.open(image_root / row['image']) as image:
                image = image.convert('RGB')
                clean = encode_image(pipe, image, config, device)
                _, raw_cls = dino(image, None)
            prompt = settings['prompt'] if settings['conditioning'] == 'shared_prompt' else row['prompt']
            conditioning = pipe.encode_prompt(prompt=prompt, device=device, max_sequence_length=256)
            if conditioning[0].dtype != clean.dtype:
                raise ValueError('Prompt embeddings and model latents must use the same dtype')
            samples = []
            seeds = [noise_seed(settings['noise_seed'], row['pair_id'], r) for r in range(settings['noise_repeats'])]
            for repeat, seed in enumerate(seeds):
                noise, image_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4,
                    config['height'], config['width'], clean.dtype, device,
                    torch.Generator('cpu').manual_seed(seed))
                by_step = []
                for step in steps:
                    cls, decoded = predict_noised_cls(pipe, dino, clean, noise, scheduler, step,
                                                     conditioning, image_ids, config)
                    if not torch.isfinite(cls).all():
                        raise RuntimeError(f"Nonfinite CLS for {row['id']} at step {step}")
                    by_step.append(cls[0].cpu())
                    if repeat == 0 and row['pair_id'] in preview_pairs:
                        pipe.image_processor.postprocess(decoded, output_type='pil')[0].save(
                            root / 'previews' / f"{row['id']}_step{step}.png")
                samples.append(torch.stack(by_step))
            sample = dict(cls=torch.stack(samples).transpose(0, 1).contiguous(), clean_cls=raw_cls[0], noise_seeds=seeds)
            temporary = cached.with_suffix('.tmp')
            torch.save(sample, temporary)
            temporary.replace(cached)
        all_cls.append(sample['cls'])
        clean_cls.append(sample['clean_cls'])
        print(f"Noised CLS {len(all_cls)}/{len(rows)}: {row['id']}", flush=True)
    features = dict(version=2, kind='noised_one_step', rows=rows, cls=torch.stack(all_cls),
        clean_cls=torch.stack(clean_cls), **schedule, estimation=settings,
        prediction_signature=prediction_signature(config), cls_signature=cls_signature(config))
    statistics = noised_statistics(features)
    torch.save(features, root / 'cls_features.pt')
    torch.save(statistics, root / 'cls_direction.pt')
    save_json(root / 'summary.json', dict(n_images=len(rows), n_train_pairs=statistics['n_train_pairs'],
        noise_repeats=settings['noise_repeats'], conditioning=settings['conditioning'],
        levels=[dict(step=s, sigma=float(schedule['sigmas'][i]), timestep=float(schedule['timesteps'][i]),
                     direction_norm=float(statistics['directions'][i].norm()),
                     exact_zero=not bool(torch.count_nonzero(statistics['directions'][i]))) for i, s in enumerate(steps)]))
