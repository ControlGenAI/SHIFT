"""The last one-step objective must score the image SHIFT actually returns."""
import pytest
import torch

from src.dino_adapter.cls_guidance import CLSActivationGuidance
from test_cls_pipeline import tiny_pipeline


@pytest.mark.parametrize('dtype,vae_dtype', [(torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize('saturated', [False, True])
@pytest.mark.parametrize('num_steps', [3, 4])
def test_last_step_cls_and_preview_match_actual_pipeline(dtype, vae_dtype, saturated, num_steps):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    pipe.vae.to(dtype=vae_dtype)
    pipe.transformer.enable_gradient_checkpointing()
    kwargs['num_inference_steps'] = num_steps
    if saturated:
        # Exercise the RGB clipping branch using only random, tiny CPU weights.
        with torch.no_grad():
            pipe.vae.decoder.conv_out.weight.mul_(4.)
    captured = {}
    def preview(step, stage, decoded):
        if stage == 'after':
            captured['decoded'] = decoded
            captured['rgb'] = pipe.image_processor.postprocess(decoded, output_type='pt').float()
    guide = CLSActivationGuidance(dino, direction, [0, 1], block_mode='joint',
        steps=[num_steps - 1], iterations=2, learning_rate=.003, resolution=(8, 12),
        preservation_weight=0., max_relative_rms=None, selection='last', prediction_callback=preview)
    actual = pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15),
                  activation_guidance=guide).images.float()
    with torch.no_grad():
        actual_cls = dino.cls_from_rgb(actual)
    if saturated:
        assert (captured['decoded'].abs() > 1).any()
    torch.testing.assert_close(captured['rgb'], actual, atol=0, rtol=0)
    reference = guide.references[-1]
    torch.testing.assert_close(reference['selected_cls'], actual_cls, atol=0, rtol=0)
    actual_loss = float(.5 * (actual_cls - reference['target_cls']).square().sum(-1).mean())
    assert guide.logs[-1]['final_semantic_loss'] == actual_loss
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)
    assert all(p.grad is None for module in (pipe.transformer, pipe.vae, dino.model) for p in module.parameters())


def test_legacy_unclipped_decode_is_explicit_and_reproducible():
    pipe, dino, _, direction = tiny_pipeline(torch.bfloat16)
    latents, velocity = [torch.randn(1, 24, 8).bfloat16() for _ in range(2)]
    sigma = torch.tensor(.37)
    guide = CLSActivationGuidance(dino, direction, 0, resolution=(8, 12),
                                  decode_mode='legacy_fp32_unclipped')
    with torch.no_grad():
        clean = latents.float() - sigma * velocity.float()
        unpacked = pipe._unpack_latents(clean, 8, 12, pipe.vae_scale_factor)
        unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        expected = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
        torch.testing.assert_close(guide.decode_velocity(pipe, latents, velocity, sigma), expected, atol=0, rtol=0)
        torch.testing.assert_close(guide.cls_of_velocity(pipe, latents, velocity, sigma),
                                   dino.cls_from_rgb(expected.float() / 2 + .5), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_first_update_probe_preserves_updates_and_repeats_gradients(dtype):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    pipe.transformer.enable_gradient_checkpointing()
    outputs = []
    for probes in [[], [-.1, 0., .1, 1.]]:
        guide = CLSActivationGuidance(dino, direction, [0, 1], block_mode='joint',
            steps=[0], iterations=2, learning_rate=.003, resolution=(8, 12),
            preservation_weight=0., max_relative_rms=None, selection='last', first_update_probe=probes)
        outputs.append(pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15),
                            activation_guidance=guide).images)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    rows = [r for r in guide.logs if r.get('phase') == 'first_update_probe']
    assert [r['multiplier'] for r in rows] == [-.1, 0., .1, 1.]
    zero = rows[1]
    assert zero['observed_loss_delta'] == 0 and zero['velocity_relative_rms'] == [0.]
    assert all(r['relative_l2'] == 0 and r['cosine'] == pytest.approx(1., abs=1e-6)
               for r in zero['repeat_gradients'])
    if dtype == torch.float32:
        numerical = (rows[2]['total_loss'] - rows[0]['total_loss']) / .2
        analytic = rows[2]['predicted_linear_loss_delta'] / .1
        assert analytic < 0
        assert numerical == pytest.approx(analytic, rel=.1, abs=1e-6)
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)


@pytest.mark.parametrize('settings', [dict(decode_mode='typo'), dict(first_update_probe=[.1]),
    dict(first_update_probe=[0., float('nan')]), dict(first_update_probe=[0., True])])
def test_invalid_rendering_diagnostics_fail_before_model_loading(settings):
    with pytest.raises(ValueError):
        CLSActivationGuidance(None, torch.tensor([1., -1.]), 0, **settings)


def test_probe_backward_failure_restores_corrections_and_removes_hooks(monkeypatch):
    pipe, dino, kwargs, direction = tiny_pipeline()
    pipe.transformer.enable_gradient_checkpointing()
    original = torch.autograd.grad
    calls = []
    def grad(outputs, inputs, **settings):
        calls.append(inputs)
        if len(calls) == 2:
            raise RuntimeError('repeated backward failure')
        return original(outputs, inputs, **settings)
    monkeypatch.setattr(torch.autograd, 'grad', grad)
    guide = CLSActivationGuidance(dino, direction, [0, 1], block_mode='joint',
        steps=[0], iterations=1, resolution=(8, 12), preservation_weight=0.,
        max_relative_rms=None, selection='last', first_update_probe=[.1, 0.])
    with pytest.raises(RuntimeError, match='repeated backward failure'):
        pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15), activation_guidance=guide)
    assert len(calls) == 2
    assert all(not torch.count_nonzero(u) for u in calls[0])
    assert pipe.scheduler.step_index is None
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)
