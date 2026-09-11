"""Correction units, physical budgets, and the Adam coordinate-change control."""
from types import SimpleNamespace

import pytest
import torch

from src.dino_adapter.cls_guidance import CLSActivationGuidance
from src.dino_adapter.cls_trajectory import CLSTrajectoryGuidance
from test_cls_guidance import Dino, Transformer, VAE
from test_cls_pipeline import tiny_pipeline


def test_matched_adam_reproduces_rms_coordinates_including_epsilon():
    # Small gradients make an incorrect epsilon transformation observable.
    references = {i: torch.tensor([[[scale]]], dtype=torch.float64) for i, scale in enumerate((.02, 7.3))}
    runs = []
    for scaling, matched in [('rms', False), ('none', True)]:
        guide = CLSActivationGuidance(None, torch.tensor([1., -1.]), [0, 1], block_mode='joint',
            correction_scaling=scaling, match_rms_adam=matched, learning_rate=.07)
        variables = {i: torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True) for i in references}
        scales = guide.correction_scales(references)
        optimizer, metadata = guide.correction_optimizer(variables, references, scales)
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            a, b = [scales[i] * variables[i] for i in references]
            target = a.new_tensor([.3, -.5, .7])
            loss = 1e-8 * ((a + .3 * b - target).square().sum() + .2 * (b - .1).square().sum())
            loss.backward()
            optimizer.step()
        runs.append({i: (scales[i] * variables[i]).detach() for i in references})
        for row in metadata:
            scale = float(references[row['block']])
            assert row['learning_rate'] == pytest.approx(.07 * scale if matched else .07)
            assert row['epsilon'] == pytest.approx(1e-8 / scale if matched else 1e-8, abs=1e-16)
    for i in references:
        torch.testing.assert_close(runs[0][i], runs[1][i], atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('space', ['activation', 'velocity'])
@pytest.mark.parametrize('cap', [None, .05])
def test_raw_corrections_keep_physical_rms_diagnostics_and_optional_budget(space, cap):
    transformer = Transformer().eval()
    latents = torch.full((1, 4, 3), 10.)
    vae = VAE()
    # Large activation units should not make the analytic decoder fully saturated.
    vae.decode = lambda x, **kwargs: (x / 16.,)
    pipe = SimpleNamespace(transformer=transformer, vae=vae, vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
        correction_scaling='none', optimization_space=space, iterations=3, learning_rate=1.,
        max_relative_rms=cap, preservation_weight=.1, selection='last', resolution=(2, 2))
    with torch.no_grad():
        baseline = transformer(hidden_states=latents)[0]
        actual = guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    original = baseline if space == 'velocity' else latents + .1
    modified = actual if space == 'velocity' else actual / transformer.weight
    absolute = (modified - original).square().mean().sqrt().item()
    reference = original.square().mean().sqrt().item()
    summary, last = guide.logs[-1], guide.logs[-2]
    assert summary['correction_scaling'] == 'none' and not summary['match_rms_adam']
    assert summary['optimizer_groups'][0]['correction_scale'] == [1.]
    assert summary['optimizer_groups'][0]['reference_rms'] == pytest.approx([reference])
    assert last['actual_absolute_rms'] == pytest.approx([absolute], abs=1e-6)
    assert last['actual_relative_rms'] == pytest.approx([absolute / reference], abs=1e-6)
    assert last['preservation_loss'] == pytest.approx((absolute / reference) ** 2, abs=1e-6)
    assert absolute > 0
    if cap is not None:
        assert absolute / reference <= cap
        assert any(r.get('projected') for r in guide.logs)


@pytest.mark.parametrize('space', ['activation', 'velocity', 'joint', 'final'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_matched_control_agrees_through_tiny_shift_and_keeps_zero_baseline(space, dtype):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    pipe.transformer.enable_gradient_checkpointing()
    def run(guide=None):
        return pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15), activation_guidance=guide).images
    baseline = run()
    outputs, guides = [], []
    for scaling, matched in [('rms', False), ('none', True), ('none', False)]:
        options = dict(steps=[0, 1], iterations=2, learning_rate=.004, resolution=(8, 12),
                       correction_scaling=scaling, match_rms_adam=matched)
        if space == 'final':
            make = lambda alpha: CLSTrajectoryGuidance(dino, direction, [0, 1], alpha=alpha,
                image_preservation_weight=.2, **options)
        else:
            make = lambda alpha: CLSActivationGuidance(dino, direction, [0, 1] if space == 'joint' else 1,
                alpha=alpha, optimization_space='activation' if space == 'joint' else space,
                block_mode='joint' if space == 'joint' else 'independent',
                preservation_weight=0., max_relative_rms=None, selection='last', **options)
        torch.testing.assert_close(run(make(0)), baseline, atol=0, rtol=0)
        guide = make(1)
        outputs.append(run(guide))
        guides.append(guide)
        assert all(not b._forward_hooks for b in pipe.transformer.transformer_blocks)
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-4)
    assert not torch.allclose(outputs[0], outputs[2], atol=2e-5, rtol=2e-4)
    assert all(p.grad is None for m in (pipe.transformer, pipe.vae, dino.model) for p in m.parameters())
    for guide in guides:
        for row in guide.logs:
            if 'selected_iteration' in row:
                assert row['selected_iteration'] == 2
                assert all(group['reference_rms'][0] > 0 for group in row['optimizer_groups'])


@pytest.mark.parametrize('settings', [dict(correction_scaling='typo'),
    dict(correction_scaling='rms', match_rms_adam=True), dict(match_rms_adam='false')])
def test_invalid_scaling_settings_fail_before_model_loading(settings):
    with pytest.raises(ValueError):
        CLSActivationGuidance(None, torch.tensor([1., -1.]), 0, **settings)


def test_matched_control_rejects_multiple_batch_scales():
    guide = CLSActivationGuidance(None, torch.tensor([1., -1.]), 0,
                                  correction_scaling='none', match_rms_adam=True)
    reference = {0: torch.tensor([.1, 2.]).reshape(2, 1, 1)}
    variables = {0: torch.zeros(2, 4, 3, requires_grad=True)}
    with pytest.raises(ValueError, match='batch size 1'):
        guide.correction_optimizer(variables, reference, guide.correction_scales(reference))
