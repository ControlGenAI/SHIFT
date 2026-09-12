"""Noise-level CLS extraction and steering with small random CPU models only."""
import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.dino_adapter.cls_guidance import CLSActivationGuidance, correction_gradients
from src.dino_adapter.cls_noise import (StepwiseCLSDirection, audit_noised_direction,
    direction_from_payload, encode_image, extract_noised, noise_seed,
    noised_statistics, predict_noised_cls, prediction_signature)
from src.models.flux import prepare_flow_schedule
from test_cls_pipeline import tiny_pipeline


def config(dtype='float32', vae_dtype=None):
    return dict(model='tiny', model_revision='test', model_dtype=dtype, vae_dtype=vae_dtype or dtype,
        dino_model='tiny', dino_revision='test', dino_size=8, width=16, height=16,
        blocks='all', inference_steps=2, guidance_scale=0., alphas=[0., .5],
        cls_optimization=dict(objective='one_step', space='activation', block_mode='joint',
            correction_scaling='none', match_rms_adam=False, decode_mode='pipeline',
            steps=[0, 1], iterations=1, learning_rate=.003, preservation_weight=0.,
            max_relative_rms=None, selection='last', gradient_checkpointing=True),
        cls_direction_estimation=dict(steps=[0, 1], noise_repeats=2, noise_seed=13,
            conditioning='shared_prompt', prompt='portrait', preview_pairs=1))


@pytest.mark.parametrize('dtype,vae_dtype', [(torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize('shifted', [False, True])
def test_noised_cls_matches_native_flow_and_terminal_decode(dtype, vae_dtype, shifted, monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipe, dino, kwargs, _ = tiny_pipeline(dtype)
    pipe.vae.to(dtype=vae_dtype)
    settings = config(str(dtype).split('.')[-1], str(vae_dtype).split('.')[-1])
    settings.update(height=8, width=12, inference_steps=3)
    image = Image.fromarray(np.random.default_rng(31).integers(0, 256, (8, 12, 3), dtype=np.uint8))
    conditioning = (kwargs['prompt_embeds'], kwargs['pooled_prompt_embeds'], torch.zeros(2, 3).to(dtype))
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=shifted)
    prepare_flow_schedule(scheduler, 3, 'cpu', 24)
    # Independent schedule setup, including a non-power-of-two sigma and dynamic shift.
    reference = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=shifted)
    reference.set_timesteps(sigmas=[1., 2/3, 1/3], device='cpu', mu=.5 + (24 - 256) * .65 / (4096 - 256))
    torch.testing.assert_close(scheduler.sigmas, reference.sigmas, atol=0, rtol=0)
    recorded = []
    forward = pipe.transformer.forward
    def capture(*args, **kw):
        result = forward(*args, **kw)
        recorded.append((kw, result[0]))
        return result
    monkeypatch.setattr(pipe.transformer, 'forward', capture)
    with torch.no_grad():
        clean = encode_image(pipe, image, settings, 'cpu')
        pixels = pipe.image_processor.preprocess(image, height=8, width=12).to(vae_dtype)
        encoded = pipe.vae.encode(pixels).latent_dist.mode()
        encoded = (encoded - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        expected_clean = pipe._pack_latents(encoded, 1, 2, 8, 12).to(dtype)
        torch.testing.assert_close(clean, expected_clean, atol=0, rtol=0)
        noise, ids = pipe.prepare_latents(1, 2, 8, 12, dtype, 'cpu', torch.Generator().manual_seed(7))
        for step in range(3):
            actual_cls, _ = predict_noised_cls(pipe, dino, clean, noise, scheduler, step, conditioning, ids, settings)
            kw, velocity = recorded[-1]
            sigma = scheduler.sigmas[step].to(dtype).reshape(1, 1, 1)
            expected_noisy = sigma * noise + (1 - sigma) * expected_clean
            torch.testing.assert_close(kw['hidden_states'], expected_noisy, atol=0, rtol=0)
            torch.testing.assert_close(kw['timestep'], scheduler.timesteps[step:step+1].to(dtype) / 1000, atol=0, rtol=0)
            # Score an actual Euler step ending at zero, rather than calling the implementation helper.
            terminal = copy.deepcopy(reference)
            terminal.timesteps = reference.timesteps[step:step+1]
            terminal.sigmas = torch.cat([reference.sigmas[step:step+1], torch.zeros(1)])
            x0 = terminal.step(velocity, terminal.timesteps[0], expected_noisy, return_dict=False)[0]
            unpacked = pipe._unpack_latents(x0, 8, 12, 1).to(vae_dtype)
            decoded = pipe.vae.decode(unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor, return_dict=False)[0]
            expected_cls = dino.cls_from_rgb(pipe.image_processor.postprocess(decoded, output_type='pt').float())
            torch.testing.assert_close(actual_cls, expected_cls, atol=0, rtol=0)
    assert scheduler.step_index is None and scheduler.begin_index is None
    assert all(p.grad is None for m in (pipe.transformer, pipe.vae, dino.model) for p in m.parameters())


def test_pure_noise_erases_image_when_conditioning_and_noise_are_shared():
    pipe, dino, kw, _ = tiny_pipeline()
    settings = config()
    settings.update(height=8, width=12)
    prepare_flow_schedule(pipe.scheduler, 2, 'cpu', 24)
    noise, ids = pipe.prepare_latents(1, 2, 8, 12, torch.float32, 'cpu', torch.Generator().manual_seed(7))
    conditioning = (kw['prompt_embeds'], kw['pooled_prompt_embeds'], torch.zeros(2, 3))
    with torch.no_grad():
        first, _ = predict_noised_cls(pipe, dino, torch.randn_like(noise), noise, pipe.scheduler, 0, conditioning, ids, settings)
        second, _ = predict_noised_cls(pipe, dino, 10 * torch.randn_like(noise), noise, pipe.scheduler, 0, conditioning, ids, settings)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    pipe.scheduler.set_begin_index(0)
    with pytest.raises(ValueError, match='fresh scheduler'):
        predict_noised_cls(pipe, dino, noise, noise, pipe.scheduler, 1, conditioning, ids, settings)


def table(config, direction):
    return dict(kind='noised_one_step', directions=torch.stack([torch.zeros_like(direction), direction]),
                steps=[0, 1], sigmas=torch.tensor([1., .5]), timesteps=torch.tensor([1000., 500.]),
                prediction_signature=prediction_signature(config))


@pytest.mark.parametrize('space', ['activation', 'joint', 'velocity'])
def test_guidance_selects_sigma_direction_and_exact_zero_is_explicit_bypass(space):
    pipe, dino, kwargs, direction = tiny_pipeline(torch.bfloat16)
    pipe.transformer.enable_gradient_checkpointing()
    settings = config('bfloat16')
    payload = table(settings, direction)
    guide = CLSActivationGuidance(dino, StepwiseCLSDirection(payload), [0, 1] if space == 'joint' else 0,
        steps=[0, 1], block_mode='joint' if space == 'joint' else 'independent',
        optimization_space='velocity' if space == 'velocity' else 'activation',
        alpha=.5, iterations=1, resolution=(8, 12), correction_scaling='none',
        preservation_weight=0, max_relative_rms=None, selection='last')
    output = pipe(**kwargs, generator=torch.Generator().manual_seed(5), activation_guidance=guide).images
    assert torch.isfinite(output).all()
    assert guide.logs[0]['reason'] == 'zero_mean_diff' and guide.logs[0]['sigma'] == 1.
    assert len(guide.references) == 1 and guide.references[0]['step'] == 1
    r = guide.references[0]
    torch.testing.assert_close(r['target_cls'], F.normalize(r['source_cls'] - .5 * direction, dim=-1), atol=0, rtol=0)
    assert guide.logs[-1]['selected_iteration'] == 1
    assert all(p.grad is None for p in pipe.transformer.parameters())


def test_direction_rejects_mismatched_precision_and_schedule():
    pipe, _, _, direction = tiny_pipeline()
    settings = config()
    payload = table(settings, direction)
    direction_from_payload(payload, settings)
    changed = copy.deepcopy(settings)
    changed['vae_dtype'] = 'bfloat16'
    with pytest.raises(ValueError, match='prediction settings differ'):
        direction_from_payload(payload, changed)
    with pytest.raises(ValueError, match='sigma/timestep'):
        StepwiseCLSDirection(payload).at(pipe, 1, .75, 750.)
    with pytest.raises(ValueError, match='Missing noise-level'):
        CLSActivationGuidance(None, StepwiseCLSDirection(payload), 0, steps=[2])
    pipe.vae.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='dtype'):
        StepwiseCLSDirection(payload).at(pipe, 1, .5, 500.)


def test_dino_cls_and_rgb_gradients_stay_fp32_under_outer_autocast():
    _, dino, _, _ = tiny_pipeline()
    rgb = torch.rand(1, 3, 8, 12, requires_grad=True)
    reference = dino.cls_from_rgb(rgb)
    gradient = correction_gradients(reference[:, 0].sum(), rgb)[0]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = dino.cls_from_rgb(rgb)
        actual_gradient = correction_gradients(actual[:, 0].sum(), rgb)[0]
    assert actual.dtype == actual_gradient.dtype == torch.float32
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, gradient, atol=0, rtol=0)


@pytest.mark.parametrize('model_dtype,vae_dtype', [('bfloat16', 'bfloat16'),
    ('bfloat16', 'float32'), ('float32', 'float32')])
def test_loader_requests_original_vae_weights_in_the_selected_precision(model_dtype, vae_dtype, monkeypatch):
    from types import SimpleNamespace
    from diffusers import AutoencoderKL
    from src.models.flux import FluxPipeline
    from src.dino_adapter.runtime import load_pipeline
    requests = []
    def vae_factory(name, **kw):
        requests.append(('vae', kw))
        return torch.nn.Linear(2, 2).to(kw['torch_dtype'])
    def pipeline_factory(name, **kw):
        requests.append(('pipeline', kw))
        modules = {k: torch.nn.Linear(2, 2).to(kw['torch_dtype'])
                   for k in ('transformer', 'vae', 'text_encoder', 'text_encoder_2')}
        if 'vae' in kw:
            modules['vae'] = kw['vae']
        pipe = SimpleNamespace(**modules, set_progress_bar_config=lambda **kw: None)
        pipe.to = lambda device: pipe
        return pipe
    monkeypatch.setattr(AutoencoderKL, 'from_pretrained', vae_factory)
    monkeypatch.setattr(FluxPipeline, 'from_pretrained', pipeline_factory)
    pipe = load_pipeline(config(model_dtype, vae_dtype), 'cpu')
    assert next(pipe.transformer.parameters()).dtype == getattr(torch, model_dtype)
    assert next(pipe.vae.parameters()).dtype == getattr(torch, vae_dtype)
    assert [name for name, _ in requests] == (['vae', 'pipeline'] if model_dtype != vae_dtype else ['pipeline'])
    if model_dtype != vae_dtype:
        assert requests[0][1]['torch_dtype'] == torch.float32 and requests[0][1]['subfolder'] == 'vae'
    assert all(not p.requires_grad for m in (pipe.vae, pipe.transformer) for p in m.parameters())


def test_means_are_train_only_and_average_normalized_observations():
    from src.dino_adapter.cls_experiment import cls_signature
    settings = config()
    rows = [dict(id=f'{p}_{label}', pair_id=p, label=label, split='train' if p == 0 else 'test', seed=p)
            for p in range(2) for label in range(2)]
    values = torch.randn(4, 2, 2, 3)
    clean = torch.randn(4, 3)
    features = dict(kind='noised_one_step', rows=rows, cls=values, clean_cls=clean, steps=[0, 1],
        sigmas=torch.tensor([1., .5]), timesteps=torch.tensor([1000., 500.]), estimation=settings['cls_direction_estimation'],
        prediction_signature=prediction_signature(settings), cls_signature=cls_signature(settings))
    stats = noised_statistics(features)
    expected = (F.normalize(values[1], dim=-1) - F.normalize(values[0], dim=-1)).mean(1)
    torch.testing.assert_close(stats['directions'], expected)
    features['cls'][2:] = 100 * torch.randn_like(values[2:])
    torch.testing.assert_close(noised_statistics(features)['directions'], stats['directions'], atol=0, rtol=0)
    audit_noised_direction(features, stats)
    corrupted = copy.deepcopy(stats)
    corrupted['directions'][1].neg_()
    with pytest.raises(ValueError, match='does not match cached'):
        audit_noised_direction(features, corrupted)
    assert noise_seed(1, 'pair0', 0) == noise_seed(1, 'pair0', 0)
    assert len({noise_seed(1, p, r) for p in ['a', 'b'] for r in range(2)}) == 4


@pytest.mark.parametrize('conditioning', ['shared_prompt', 'paired_prompts'])
def test_extract_resume_mean_audit_and_optimize_end_to_end(tmp_path, monkeypatch, conditioning):
    from src.dino_adapter import cls_noise, cls_experiment
    pipe, dino, kwargs, _ = tiny_pipeline()
    settings = config()
    settings['cls_direction_estimation']['conditioning'] = conditioning
    # Only replace text encoding/model loading; keep actual FLUX, VAE and DINO computation.
    def encode_prompt(prompt, **kw):
        offset = .3 if prompt.endswith('1') else -.3 if prompt.endswith('0') else 0.
        return kwargs['prompt_embeds'] + offset, kwargs['pooled_prompt_embeds'] + offset, torch.zeros(2, 3)
    monkeypatch.setattr(pipe, 'encode_prompt', encode_prompt)
    for module in (cls_noise, cls_experiment):
        monkeypatch.setattr(module, 'load_pipeline', lambda *args: pipe)
        monkeypatch.setattr(module, 'DinoFeatures', lambda *args: dino)
    rows = []
    for pair in range(2):
        for label in range(2):
            name = f'{pair}_{label}'
            pixels = np.random.default_rng(pair * 2 + label).integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(tmp_path / f'{name}.png')
            rows.append(dict(id=name, image=f'{name}.png', label=label, pair_id=pair, seed=pair,
                             split='train' if pair == 0 else 'test', prompt=f'portrait {label}'))
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    output = tmp_path / 'features'
    extract_noised(settings, manifest, None, output, 'cpu')
    features = torch.load(output / 'cls_features.pt', weights_only=True)
    payload = torch.load(output / 'cls_direction.pt', weights_only=True)
    assert features['cls'].shape == (4, 2, 2, 12)
    assert bool(torch.count_nonzero(payload['directions'][0])) == (conditioning == 'paired_prompts')
    assert torch.count_nonzero(payload['directions'][1])
    a, b = [torch.load(output / 'samples' / f'0_{label}.pt', weights_only=True) for label in range(2)]
    assert a['noise_seeds'] == b['noise_seeds']
    assert len(list((output / 'previews').glob('*.png'))) == 4
    cls_experiment.mean_from_features(output / 'cls_features.pt', tmp_path / 'recomputed.pt')
    recomputed = torch.load(tmp_path / 'recomputed.pt', weights_only=True)
    torch.testing.assert_close(recomputed['directions'], payload['directions'], atol=0, rtol=0)
    cls_experiment.audit_direction(output / 'cls_features.pt', output / 'cls_direction.pt', tmp_path / 'audit.json')
    def forbidden(*args):
        raise AssertionError('Resume should use the saved per-image features')
    monkeypatch.setattr(cls_noise, 'encode_image', forbidden)
    extract_noised(settings, manifest, None, output, 'cpu', resume=True)
    cls_experiment.optimize(settings, [rows[-1]], output / 'cls_direction.pt', tmp_path / 'edited', 'cpu')
    generations = json.loads((tmp_path / 'edited' / 'generations.json').read_text())
    assert generations[-2]['baseline_pixel_max_abs'] == 0
    assert generations[-1]['direction_kind'] == 'noised_one_step'
    details = json.loads((tmp_path / 'edited' / generations[-1]['image']).with_suffix('.json').read_text())
    if conditioning == 'shared_prompt':
        assert details['logs'][0]['reason'] == 'zero_mean_diff'
    else:
        assert [r['step'] for r in details['logs'] if 'selected_iteration' in r] == [0, 1]
    assert details['logs'][-1]['selected_iteration'] == 1
    provenance = json.loads((tmp_path / 'edited' / 'provenance.json').read_text())
    assert provenance['direction_details']['estimation']['conditioning'] == conditioning
    with pytest.raises(ValueError, match='identical config'):
        changed = copy.deepcopy(settings)
        changed['cls_direction_estimation']['noise_seed'] += 1
        extract_noised(changed, manifest, None, output, 'cpu', resume=True)
