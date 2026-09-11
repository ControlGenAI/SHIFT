"""Joint residuals must preserve early-block derivatives, including recomputation."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.dino_adapter.cls_guidance import CLSActivationGuidance, joint_image_outputs


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('penalty', [0., .03])
@pytest.mark.parametrize('num_layers', [3, 19])
def test_joint_flux_hooks_match_explicit_residual_forward_and_every_gradient(dtype, checkpoint, penalty, num_layers, monkeypatch):
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(32)
    model = FluxTransformer2DModel(in_channels=8, num_layers=num_layers, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        pooled_projection_dim=4, axes_dims_rope=(2, 2, 4)).to(dtype).eval().requires_grad_(False)
    reference = copy.deepcopy(model)
    if checkpoint:
        model.enable_gradient_checkpointing()
    kwargs = dict(hidden_states=torch.randn(2, 4, 8).to(dtype),
        encoder_hidden_states=torch.randn(2, 2, 12).to(dtype),
        pooled_projections=torch.randn(2, 4).to(dtype), timestep=torch.tensor([.75, .75]).to(dtype),
        img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
        txt_ids=torch.zeros(2, 3), return_dict=False)
    blocks = dict(enumerate(model.transformer_blocks))
    corrections = {i: (torch.randn(2, 4, 16) * .03).requires_grad_() for i in blocks}
    scales = {i: torch.tensor([.8, 1.3]).reshape(2, 1, 1) * (i + 1) for i in blocks}
    with torch.no_grad():
        baseline = model(**kwargs)[0]
    with joint_image_outputs(blocks, corrections, scales, track_preservation=bool(penalty)) as stats:
        actual = model(**kwargs)[0]
        loss = actual.float().square().mean()
        if penalty:
            loss = loss + penalty * torch.stack([s['preservation'] for s in stats.values()]).mean()
        gradients = torch.autograd.grad(loss, tuple(corrections.values()))
    assert all(not b._forward_hooks for b in blocks.values())
    assert model.is_gradient_checkpointing == checkpoint
    # Oracle: explicit additions in block.forward, without hooks or checkpointing.
    penalties = []
    def residual_forward(forward, u, scale):
        def wrapped(*args, **kwargs):
            text, h = forward(*args, **kwargs)
            modified = (h.float() + scale * u).to(h.dtype)
            penalties.append(((modified.float() - h.float()) / scale).square().mean())
            return text, modified
        return wrapped
    for i, block in enumerate(reference.transformer_blocks):
        monkeypatch.setattr(block, 'forward', residual_forward(block.forward, corrections[i], scales[i]))
    expected = reference(**kwargs)[0]
    expected_loss = expected.float().square().mean() + penalty * torch.stack(penalties).mean()
    expected_gradients = torch.autograd.grad(expected_loss, tuple(corrections.values()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(gradients, expected_gradients):
        assert torch.isfinite(actual_grad).all() and actual_grad.norm() > 0
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(model(**kwargs)[0], baseline, atol=0, rtol=0)


class CoupledBlock(torch.nn.Module):
    def forward(self, text, image):
        return (text + image.mean(1, keepdim=True) * .1,
                torch.tanh(image + text.mean(1, keepdim=True)) + image.roll(1, dims=1) * .2)


class CoupledTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([CoupledBlock() for _ in range(3)])
    def forward(self, hidden_states):
        text, image = torch.zeros(1, 2, 3), hidden_states
        for block in self.transformer_blocks:
            text, image = block(text, image)
        return (image * .4,)


class Dino:
    def cls_from_rgb(self, rgb):
        return F.normalize(rgb.mean((-1, -2)), dim=-1)


@pytest.mark.parametrize('scaling', ['rms', 'none'])
def test_joint_guidance_matches_plain_adam_on_coupled_tokens_without_caps(scaling):
    torch.manual_seed(3)
    transformer = CoupledTransformer().eval()
    latents = torch.randn(1, 4, 3)
    pipe = SimpleNamespace(transformer=transformer,
        vae=SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=1., shift_factor=0.),
                            decode=lambda x, **kw: (x,)), vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), [0, 1, 2], block_mode='joint',
        iterations=3, learning_rate=.2, preservation_weight=0., max_relative_rms=None,
        selection='last', resolution=(2, 2), correction_scaling=scaling)
    scales = []
    with torch.no_grad():
        text, h = torch.zeros(1, 2, 3), latents
        for block in transformer.transformer_blocks:
            text, h = block(text, h)
            scales.append(h.square().mean((1, 2), keepdim=True).sqrt())
        source = guide.cls_of_velocity(pipe, latents, h * .4, torch.tensor(.75))
        target = F.normalize(source - guide.direction, dim=-1)
    corrections = [torch.zeros_like(latents, requires_grad=True) for _ in scales]
    optimizer = torch.optim.Adam(corrections, lr=.2)
    def forward():
        text, h = torch.zeros(1, 2, 3), latents
        for block, scale, u in zip(transformer.transformer_blocks, scales, corrections):
            text, h = block(text, h)
            h = h + (scale if scaling == 'rms' else 1.) * u
        return h * .4
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        cls = guide.cls_of_velocity(pipe, latents, forward(), torch.tensor(.75))
        (.5 * (cls - target).square().sum(-1).mean()).backward()
        optimizer.step()
    with torch.no_grad():
        expected = forward()
        actual = guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not actual.requires_grad
    assert guide.logs[-1]['selected_iteration'] == 3
    assert any(max(b['selected_relative_rms']) > .05 for b in guide.logs[-1]['selected_per_block'])
    updates = [log for log in guide.logs if 'gradient_rms' in log]
    assert len(updates) == 3
    assert all(b['gradient_nonzero'] and not b['projected'] for log in updates for b in log['per_block'])


@pytest.mark.parametrize('block,space,mode', [([], 'activation', 'joint'),
    ([0, 0], 'activation', 'joint'), ([False], 'activation', 'joint'),
    ([-1], 'activation', 'joint'), ([0, 1.5], 'activation', 'joint'),
    (0, 'activation', 'joint'), ([0], 'velocity', 'joint'),
    ([0], 'activation', 'independent'), (0, 'activation', 'typo')])
def test_invalid_joint_settings_fail_before_loading_models(block, space, mode):
    with pytest.raises(ValueError):
        CLSActivationGuidance(None, torch.tensor([.1, -.1]), block,
                              optimization_space=space, block_mode=mode)


def test_joint_hooks_leave_text_untouched_and_are_removed_on_error():
    blocks = {0: CoupledBlock(), 1: CoupledBlock()}
    image, text = torch.ones(1, 4, 3), torch.ones(1, 2, 3)
    corrections = {i: torch.ones_like(image, requires_grad=True) for i in blocks}
    scales = {i: torch.ones(1, 1, 1) for i in blocks}
    expected_text, original = blocks[0](text, image)
    with pytest.raises(RuntimeError, match='test failure'):
        with joint_image_outputs(blocks, corrections, scales):
            actual_text, edited = blocks[0](text, image)
            torch.testing.assert_close(actual_text, expected_text, atol=0, rtol=0)
            torch.testing.assert_close(edited, original + 1, atol=0, rtol=0)
            raise RuntimeError('test failure')
    assert all(not b._forward_hooks for b in blocks.values())
