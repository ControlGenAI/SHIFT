"""SHIFT loop with tiny random CPU models; artifact tests use temporary directories."""
import pytest
import torch
import torch.nn.functional as F

from src.dino_adapter.cls_guidance import CLSActivationGuidance
from src.dino_adapter.features import DinoFeatures


def tiny_pipeline(dtype=torch.float32):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from transformers import Dinov2Config, Dinov2Model
    from src.models.flux import FluxPipeline

    torch.manual_seed(42)
    transformer = FluxTransformer2DModel(in_channels=8, num_layers=2, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        pooled_projection_dim=4, axes_dims_rope=(2, 2, 4)).to(dtype).eval().requires_grad_(False)
    vae = AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',),
        block_out_channels=(8,), layers_per_block=1, latent_channels=2, norm_num_groups=4,
        scaling_factor=.7, shift_factor=.2).to(dtype).eval().requires_grad_(False)
    pipe = FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, transformer=transformer,
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None).to('cpu')
    pipe.set_progress_bar_config(disable=True)
    dino = DinoFeatures.__new__(DinoFeatures)
    dino.model = Dinov2Model(Dinov2Config(hidden_size=12, num_hidden_layers=2,
        num_attention_heads=3, image_size=8, patch_size=2)).eval().requires_grad_(False)
    dino.size, dino.device = 8, 'cpu'
    kwargs = dict(prompt_embeds=torch.randn(1, 2, 12).to(dtype),
        pooled_prompt_embeds=torch.randn(1, 4).to(dtype), width=12, height=8,
        num_inference_steps=2, guidance_scale=0., max_sequence_length=256,
        structure_strength=0., txt_steering={'vector': None}, output_type='pt')
    direction = F.normalize(torch.randn(12), dim=0) * .2
    return pipe, dino, kwargs, direction


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('space', ['activation', 'velocity'])
def test_full_shift_loop_with_real_vae_zero_control_and_bf16(dtype, space, monkeypatch):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    pipe.transformer.enable_gradient_checkpointing()
    original_step = pipe.scheduler.step
    scheduler_calls = []
    def step(*args, **kw):
        scheduler_calls.append(pipe.scheduler.step_index)
        return original_step(*args, **kw)
    monkeypatch.setattr(pipe.scheduler, 'step', step)
    def run(guide=None):
        scheduler_calls.clear()
        result = pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15),
                      activation_guidance=guide).images
        assert scheduler_calls == [None, 1]
        assert pipe.scheduler.step_index == 2
        assert result.shape == (1, 3, 8, 12) and torch.isfinite(result).all()
        return result
    baseline = run()
    options = dict(steps=[0, 1], iterations=3, preservation_weight=.001,
                   resolution=(8, 12), optimization_space=space)
    zero = CLSActivationGuidance(dino, direction, 1, alpha=0, **options)
    torch.testing.assert_close(run(zero), baseline, rtol=0, atol=0)
    assert len(zero.logs) == 2 and all(log['bypass'] for log in zero.logs)
    guide = CLSActivationGuidance(dino, direction, 1, **options)
    run(guide)
    selected = [log for log in guide.logs if 'selected_iteration' in log]
    assert len(selected) == 2 and len(guide.references) == 2
    assert any(log['selected_iteration'] > 0 for log in selected)
    for log in selected:
        assert log['final_semantic_loss'] <= log['baseline_semantic_loss']
        assert max(log['selected_relative_rms']) <= guide.cap
    for module in (pipe.transformer, pipe.vae, dino.model):
        assert all(p.grad is None for p in module.parameters())
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)


def test_pipeline_rejects_wrong_spatial_grid_and_unused_steps():
    pipe, dino, kwargs, direction = tiny_pipeline()
    # Equal area, different aspect ratio: unpacking would silently mix spatial tokens.
    guide = CLSActivationGuidance(dino, direction, 0, resolution=(12, 8))
    with pytest.raises(ValueError, match='resolution must match'):
        pipe(**kwargs, activation_guidance=guide)
    guide = CLSActivationGuidance(dino, direction, 0, resolution=(8, 12), steps=[2])
    with pytest.raises(ValueError, match='outside the actual'):
        pipe(**kwargs, activation_guidance=guide)


@pytest.mark.parametrize('settings', [dict(steps=[.5]), dict(steps=[False]),
    dict(iterations=2.5), dict(block=1.5)])
def test_noninteger_guidance_settings_fail_before_model_loading(settings):
    kwargs = dict(block=0)
    kwargs.update(settings)
    with pytest.raises(ValueError, match='integer'):
        CLSActivationGuidance(None, torch.tensor([.1, -.1]), **kwargs)


@pytest.mark.parametrize('space', ['activation', 'velocity'])
def test_all_four_steps_use_last_adam_state_and_record_predictions(space, monkeypatch):
    from pathlib import Path
    from src.dino_adapter.cls_experiment import read_cls_config, guidance_options
    filename = 'cls_guidance_all_steps.json' if space == 'activation' else 'cls_velocity_guidance_all_steps.json'
    config = read_cls_config(Path(__file__).resolve().parents[1] / 'configs' / filename)
    options = guidance_options(config)
    assert options['steps'] == [0, 1, 2, 3]
    assert options['max_relative_rms'] is None and options['preservation_weight'] == 0.
    assert options['selection'] == 'last' and config['cls_optimization']['save_step_predictions']
    pipe, dino, kwargs, direction = tiny_pipeline(torch.bfloat16)
    pipe.transformer.enable_gradient_checkpointing()
    kwargs['num_inference_steps'] = 4
    options.update(iterations=2, resolution=(8, 12))
    scheduler_calls, snapshots = [], []
    original_step = pipe.scheduler.step
    def step(*args, **kw):
        scheduler_calls.append(pipe.scheduler.step_index)
        return original_step(*args, **kw)
    monkeypatch.setattr(pipe.scheduler, 'step', step)
    def capture(step, stage, decoded):
        assert not torch.is_grad_enabled() and not decoded.requires_grad
        assert decoded.shape == (1, 3, 8, 12) and torch.isfinite(decoded).all()
        snapshots.append((step, stage))
    guide = CLSActivationGuidance(dino, direction, 1, **options, prediction_callback=capture)
    output = pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15), activation_guidance=guide).images
    assert torch.isfinite(output).all()
    assert scheduler_calls == [None, 1, 2, 3]
    selected = [log for log in guide.logs if 'selected_iteration' in log]
    assert [log['step'] for log in selected] == [0, 1, 2, 3]
    assert all(log['selected_iteration'] == 2 and log['selection'] == 'last' for log in selected)
    assert all(log['correction_dtype'] == 'torch.float32' for log in selected)
    updates = [log for log in guide.logs if 'gradient_nonzero' in log]
    assert len(updates) == 8 and all(log['gradient_nonzero'] and not log['projected'] for log in updates)
    assert snapshots == [(step, stage) for step in range(4) for stage in ('before', 'after')]
    assert len(guide.references) == 4
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)
    assert all(p.grad is None for module in (pipe.transformer, pipe.vae, dino.model) for p in module.parameters())


def test_experiment_wires_unbounded_options_and_saves_step_images(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from src.dino_adapter import cls_experiment
    config = cls_experiment.read_cls_config(Path(__file__).resolve().parents[1] /
                                          'configs/cls_velocity_guidance_all_steps.json')
    config.update(width=16, height=16, dino_size=8, alphas=[0., 1.])
    config['cls_optimization']['iterations'] = 1
    pipe, dino, inputs, direction = tiny_pipeline()
    class PreparedPrompts:
        transformer = pipe.transformer
        image_processor = pipe.image_processor
        def __call__(self, prompt, **kwargs):
            return pipe(prompt_embeds=inputs['prompt_embeds'],
                        pooled_prompt_embeds=inputs['pooled_prompt_embeds'], **kwargs)
    monkeypatch.setattr(cls_experiment, 'load_pipeline', lambda *args: PreparedPrompts())
    monkeypatch.setattr(cls_experiment, 'DinoFeatures', lambda *args: dino)
    direction_path = tmp_path / 'direction.pt'
    torch.save(dict(direction=direction, train_pair_ids=['train'], train_seeds=[1],
                    cls_signature=cls_experiment.cls_signature(config)), direction_path)
    rows = [dict(id='test1', pair_id='heldout', seed=200, prompt='test prompt')]
    output = tmp_path / 'output'
    cls_experiment.optimize(config, rows, direction_path, output, 'cpu')
    entries = json.loads((output / 'generations.json').read_text())
    zero, edited = entries[1:]
    assert zero['alpha'] == 0 and zero['baseline_pixel_max_abs'] == 0
    stem = Path(edited['image']).stem
    for step in range(4):
        for stage in ('before', 'after'):
            assert (output / f'{stem}_step{step}_{stage}.png').is_file()
    details = json.loads((output / f'{stem}.json').read_text())
    selected = [log for log in details['logs'] if 'selected_iteration' in log]
    assert len(selected) == 4
    assert all(log['selection'] == 'last' and log['selected_iteration'] == 1 and
               log['max_relative_rms'] is None and log['preservation_weight'] == 0 for log in selected)
    assert len(torch.load(output / f'{stem}_cls_targets.pt', weights_only=True)) == 4
