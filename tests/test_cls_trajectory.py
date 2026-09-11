"""Final-image optimization must agree with the real SHIFT sampler, also in BF16."""
import copy
import json

import pytest
import torch

from src.dino_adapter.cls_experiment import audit_direction, paired_mean
from src.dino_adapter.cls_trajectory import CLSTrajectoryGuidance, channel_outputs, decode_final, rollout
from test_cls_pipeline import tiny_pipeline


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_optimized_final_pixels_equal_pipeline_output_and_zero_equals_baseline(dtype, checkpointing, monkeypatch):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    kwargs['num_inference_steps'] = 4
    if checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    calls = []
    scheduler_class = type(pipe.scheduler)
    original_step = scheduler_class.step
    def step(scheduler, *args, **kw):
        if scheduler is pipe.scheduler:
            calls.append(scheduler.step_index)
        return original_step(scheduler, *args, **kw)
    monkeypatch.setattr(scheduler_class, 'step', step)
    def run(guide=None):
        calls.clear()
        result = pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15), activation_guidance=guide).images
        assert calls == [None, 1, 2, 3]  # Inner rollouts never advance the outer scheduler.
        return result
    baseline = run()
    options = dict(steps=[0, 1, 2, 3], iterations=2, resolution=(8, 12),
                   image_preservation_weight=.2, edit_roi=[.2, .35, .8, .65],
                   dino_checkpointing=checkpointing)
    zero = CLSTrajectoryGuidance(dino, direction, [0, 1], alpha=0, **options)
    torch.testing.assert_close(run(zero), baseline, atol=0, rtol=0)
    snapshots = {}
    def capture(iteration, decoded):
        assert not decoded.requires_grad and not torch.is_grad_enabled()
        snapshots[iteration] = pipe.image_processor.postprocess(decoded, output_type='pt').float()
    guide = CLSTrajectoryGuidance(dino, direction, [0, 1], iteration_callback=capture, **options)
    actual = run(guide)
    # The optimized image and the returned image are the SAME complete rollout.
    torch.testing.assert_close(guide.selected_rgb, actual.float(), atol=0, rtol=0)
    torch.testing.assert_close(snapshots[0], baseline.float(), atol=0, rtol=0)
    torch.testing.assert_close(snapshots[2], actual.float(), atol=0, rtol=0)
    torch.testing.assert_close(guide.baseline_rgb, baseline.float(), atol=0, rtol=0)
    assert not torch.equal(actual, baseline)
    updates = [r for r in guide.logs if 'iteration' in r]
    assert [r['iteration'] for r in updates] == [0, 1, 2]
    assert all(b['gradient_nonzero'] for r in updates[:-1] for b in r['per_block'])
    summary = next(r for r in guide.logs if 'selected_iteration' in r)
    assert summary['selected_iteration'] == 2 and summary['optimized_parameters'] == 32
    assert summary['max_relative_rms'] is None and summary['selection'] == 'last'
    assert len(guide.references) == 1  # One fixed target for the entire generation.
    ref = guide.references[0]
    with torch.no_grad():
        actual_cls = guide.view_features(actual.float())
    torch.testing.assert_close(ref['selected_cls'], actual_cls, atol=0, rtol=0)
    assert all(not b._forward_hooks for b in pipe.transformer.transformer_blocks)
    assert all(p.grad is None for m in (pipe.transformer, pipe.vae, dino.model) for p in m.parameters())


@pytest.mark.parametrize('checkpointing', [False, True])
def test_full_sampler_channel_gradients_match_explicit_forward_and_finite_difference(checkpointing, monkeypatch):
    pipe, _, kwargs, _ = tiny_pipeline()
    kwargs['num_inference_steps'] = 4
    # Capture actual pipeline inputs and its initial scheduler; do not recreate its packing/schedule.
    captured = {}
    def inputs(module, args, kw):
        if not captured:
            captured['kwargs'] = kw.copy()
            captured['scheduler'] = copy.deepcopy(pipe.scheduler)
    handle = pipe.transformer.register_forward_pre_hook(inputs, with_kwargs=True)
    baseline = pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15)).images
    handle.remove()
    pipe.scheduler = captured['scheduler']
    inputs = captured['kwargs']
    initial = inputs['hidden_states']
    if checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    reference_model = copy.deepcopy(pipe.transformer)
    reference_model.disable_gradient_checkpointing()
    blocks = dict(enumerate(pipe.transformer.transformer_blocks))
    references = {}
    with torch.no_grad(), channel_outputs(blocks, references):
        terminal, _, _ = rollout(pipe, initial, inputs)
        torch.testing.assert_close(decode_final(pipe, terminal, (8, 12))[1], baseline, atol=0, rtol=0)
    u = {i: (torch.randn(r['shape']) * .01).requires_grad_() for i, r in references.items()}
    with channel_outputs(blocks, references, u):
        actual, _, _ = rollout(pipe, initial, inputs)
        gradients = torch.autograd.grad(actual.square().mean(), tuple(u.values()))
    def residual_forward(forward, correction, scale):
        def wrapped(*args, **kwargs):
            text, image = forward(*args, **kwargs)
            return text, image + scale * correction
        return wrapped
    for i, block in enumerate(reference_model.transformer_blocks):
        monkeypatch.setattr(block, 'forward', residual_forward(block.forward, u[i], references[i]['scale']))
    reference_pipe = copy.copy(pipe)
    reference_pipe.transformer = reference_model
    expected, _, _ = rollout(reference_pipe, initial, inputs)
    expected_gradients = torch.autograd.grad(expected.square().mean(), tuple(u.values()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(gradients, expected_gradients):
        assert a.norm() > 0
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    # Numerical derivative of the actual four-step sampler w.r.t. the earliest block.
    tangent = torch.randn_like(u[0])
    tangent /= tangent.norm()
    eps = .001
    losses = []
    with torch.no_grad():
        for sign in (-1, 1):
            candidates = dict(u)
            candidates[0] = u[0] + sign * eps * tangent
            with channel_outputs(blocks, references, candidates):
                final, _, _ = rollout(pipe, initial, inputs)
            losses.append(final.square().mean())
    numerical = (losses[1] - losses[0]) / (2 * eps)
    analytic = (gradients[0] * tangent).sum()
    torch.testing.assert_close(analytic, numerical, atol=2e-4, rtol=.02)
    assert pipe.scheduler.step_index is None
    assert all(not b._forward_hooks for b in blocks.values())


def test_trajectory_rejects_partial_schedule_and_conflicting_shift_steering():
    pipe, dino, kwargs, direction = tiny_pipeline()
    guide = CLSTrajectoryGuidance(dino, direction, [0, 1], steps=[0], resolution=(8, 12))
    with pytest.raises(ValueError, match='complete schedule'):
        pipe(**kwargs, activation_guidance=guide)
    guide = CLSTrajectoryGuidance(dino, direction, [0, 1], steps=[0, 1], resolution=(8, 12))
    kwargs['structure_strength'] = 1.
    with pytest.raises(ValueError, match='static conditioning'):
        pipe(**kwargs, activation_guidance=guide)


def test_backward_failure_removes_all_hooks_and_does_not_advance_scheduler(monkeypatch):
    pipe, dino, kwargs, direction = tiny_pipeline()
    pipe.transformer.enable_gradient_checkpointing()
    original = dino.cls_from_rgb
    def fail_grad(rgb):
        if torch.is_grad_enabled():
            raise RuntimeError('intentional failure')
        return original(rgb)
    monkeypatch.setattr(dino, 'cls_from_rgb', fail_grad)
    guide = CLSTrajectoryGuidance(dino, direction, [0, 1], steps=[0, 1],
                                  iterations=1, resolution=(8, 12))
    with pytest.raises(RuntimeError, match='intentional failure'):
        pipe(**kwargs, activation_guidance=guide)
    assert pipe.scheduler.step_index is None
    assert guide.planned_velocities is None
    assert all(not b._forward_hooks for b in pipe.transformer.transformer_blocks)


def test_cached_audit_reports_held_out_failure_without_retraining(tmp_path):
    rows = [dict(pair_id=p, split=split, label=label, seed=seed)
            for p, split, seed in [('train', 'train', 1), ('test', 'test', 2)] for label in (1, 0)]
    vectors = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    signature = dict(model='tiny', size=4)
    torch.save(dict(rows=rows, cls=vectors, cls_signature=signature), tmp_path / 'features.pt')
    torch.save(dict(**paired_mean(rows, vectors), cls_signature=signature), tmp_path / 'direction.pt')
    audit_direction(tmp_path / 'features.pt', tmp_path / 'direction.pt', tmp_path / 'audit.json')
    report = json.loads((tmp_path / 'audit.json').read_text())
    assert report['splits']['train']['auc'] == 1.
    assert report['splits']['test']['auc'] == 0.
    assert report['splits']['test']['paired_order_accuracy'] == 0.
    assert report['splits']['val']['auc'] is None


def test_paired_prompt_selection_uses_text_only_and_checks_split_seed():
    from src.dino_adapter.cls_experiment import held_out_prompts
    samples = [dict(id='p1', pair_id='p', split='test', label=1, seed=3, prompt='glasses'),
               dict(id='p0', pair_id='p', split='test', label=0, seed=3, prompt='bare eyes')]
    selected = held_out_prompts(samples, 'test', 'paired_target')
    assert selected[0]['prompt'] == 'glasses' and selected[0]['target_prompt'] == 'bare eyes'
    assert 'target_prompt' not in samples[0]
    samples[1]['seed'] = 4
    with pytest.raises(ValueError, match='matching pair, split and seed'):
        held_out_prompts(samples, 'test', 'paired_target')
