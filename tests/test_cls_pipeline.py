"""Full SHIFT denoising loop with tiny random CPU models; no downloads or images saved."""
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
