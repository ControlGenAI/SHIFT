from types import SimpleNamespace
import torch
import torch.nn.functional as F
from src.dino_adapter.cls_guidance import CLSActivationGuidance
from src.dino_adapter.cls_experiment import paired_mean


def test_mean_diff_uses_only_train_and_keeps_magnitude():
    rows = [dict(pair_id='p', label=1, split='train'), dict(pair_id='p', label=0, split='train'),
            dict(pair_id='q', label=1, split='test')]
    vectors = [torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([99., 99.])]
    result = paired_mean(rows, vectors)
    torch.testing.assert_close(result['direction'], torch.tensor([1., -1.]))


class Block(torch.nn.Module):
    def forward(self, text, image):
        return text, image + .1


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block()])
        self.weight = torch.nn.Parameter(torch.tensor(.4), requires_grad=False)
    def forward(self, hidden_states, **kwargs):
        _, h = self.transformer_blocks[0](torch.zeros(1, 2, 3), hidden_states)
        return (h * self.weight,)


class VAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1., shift_factor=0.)
    def decode(self, x, **kwargs):
        return (x,)


class Dino:
    def cls_from_rgb(self, x):
        return F.normalize(x.mean((-1, -2)), dim=-1)


def test_guidance_through_downstream_vae_and_cls_under_no_grad():
    torch.manual_seed(5)
    transformer = Transformer().eval()
    pipe = SimpleNamespace(transformer=transformer, vae=VAE(), vae_scale_factor=1,
                           scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
                           _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    latents = torch.randn(1, 4, 3)
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
                                  iterations=10, learning_rate=.01, preservation_weight=.01)
    guide.resolution = (2, 2)
    with torch.no_grad():
        output = guide.predict(pipe, 0, torch.tensor(750.), latents, dict(hidden_states=latents))
    assert not output.requires_grad
    assert max(guide.logs[-1]['selected_relative_rms']) <= .05
    assert guide.logs[-1]['final_semantic_loss'] < guide.logs[0]['semantic_loss']
    assert transformer.weight.grad is None and not transformer.transformer_blocks[0]._forward_hooks
    torch.testing.assert_close(pipe.scheduler.sigmas, torch.tensor([.75, 0.]))
    zero = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0, alpha=0)
    with torch.no_grad():
        original = transformer(hidden_states=latents)[0]
        actual = zero.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    torch.testing.assert_close(original, actual, rtol=0, atol=0)


def test_cached_cls_extraction_and_mean_never_load_models(tmp_path, monkeypatch):
    import json
    from src.dino_adapter import cls_experiment
    config = dict(dino_model='fake', dino_revision='pinned', dino_size=4)
    rows = [dict(id='p1', pair_id='p', label=1, split='train', seed=7, features='one.pt'),
            dict(id='p0', pair_id='p', label=0, split='train', seed=7, features='zero.pt'),
            dict(id='t1', pair_id='t', label=1, split='test', seed=8, features='test.pt')]
    (tmp_path / 'dataset.json').write_text(json.dumps(dict(samples=rows, config=config)))
    for row, vector in zip(rows, [torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([3., 4.])]):
        torch.save({'cls': vector}, tmp_path / row['features'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Cached extraction must not load DINO')
    monkeypatch.setattr(cls_experiment, 'DinoFeatures', forbidden)
    cls_experiment.extract(config, None, tmp_path, tmp_path / 'output', 'cpu', cached=True)
    payload = torch.load(tmp_path / 'output/cls_direction.pt', weights_only=True)
    torch.testing.assert_close(payload['direction'], torch.tensor([1., -1.]))
    assert payload['train_seeds'] == [7]
    cls_experiment.mean_from_features(tmp_path / 'output/cls_features.pt', tmp_path / 'again.pt')
    torch.testing.assert_close(torch.load(tmp_path / 'again.pt', weights_only=True)['direction'], payload['direction'])


def test_mean_rejects_split_leaks_and_vector_count_mismatch():
    import pytest
    rows = [dict(pair_id='p', label=1, split='train'), dict(pair_id='p', label=0, split='val')]
    with pytest.raises(ValueError, match='crossing'):
        paired_mean(rows, torch.randn(2, 3))
    rows[1]['split'] = 'train'
    with pytest.raises(ValueError, match='number_of_records'):
        paired_mean(rows, torch.randn(3, 3))


def test_velocity_space_optimizes_without_image_hook():
    transformer = Transformer().eval()
    pipe = SimpleNamespace(transformer=transformer, vae=VAE(), vae_scale_factor=1,
                           scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
                           _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    latents = torch.ones(1, 4, 3)
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
                                  iterations=3, preservation_weight=.01,
                                  resolution=(2, 2), optimization_space='velocity')
    with torch.no_grad():
        guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    assert guide.logs[-1]['final_semantic_loss'] <= guide.logs[-1]['baseline_semantic_loss']
    assert max(guide.logs[-1]['selected_relative_rms']) <= .05
    assert not transformer.transformer_blocks[0]._forward_hooks


def test_gradient_through_actual_tiny_flux_and_dino_with_checkpointing():
    # Real library implementations, random tiny weights on CPU; no from_pretrained.
    import copy
    from diffusers import FluxTransformer2DModel
    from transformers import Dinov2Config, Dinov2Model
    from src.dino_adapter.features import DinoFeatures
    torch.manual_seed(42)
    model = FluxTransformer2DModel(in_channels=8, num_layers=2, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        pooled_projection_dim=4, axes_dims_rope=(2, 2, 4)).eval().requires_grad_(False)
    dino = DinoFeatures.__new__(DinoFeatures)
    dino.model = Dinov2Model(Dinov2Config(hidden_size=12, num_hidden_layers=2,
        num_attention_heads=3, image_size=4, patch_size=2)).eval().requires_grad_(False)
    dino.size, dino.device = 4, 'cpu'
    latents = torch.randn(1, 4, 8)
    kwargs = dict(hidden_states=latents, encoder_hidden_states=torch.randn(1, 2, 12),
        pooled_projections=torch.randn(1, 4), timestep=torch.tensor([.75]),
        img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
        txt_ids=torch.zeros(2, 3), return_dict=False)
    direction = F.normalize(torch.randn(12), dim=0) * .2
    outputs = []
    for checkpoint in (False, True):
        transformer = copy.deepcopy(model)
        if checkpoint:
            transformer.enable_gradient_checkpointing()
        pipe = SimpleNamespace(transformer=transformer, vae=VAE(), vae_scale_factor=1,
            scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
            _unpack_latents=lambda x, *args: x[..., :3].transpose(1, 2).reshape(1, 3, 2, 2))
        guide = CLSActivationGuidance(dino, direction, 0, iterations=3,
                                      preservation_weight=.001, resolution=(2, 2))
        with torch.no_grad():
            outputs.append(guide.predict(pipe, 0, None, latents, kwargs))
        assert guide.logs[-1]['selected_iteration'] > 0
        assert guide.logs[-1]['final_semantic_loss'] < guide.logs[-1]['baseline_semantic_loss']
        assert all(p.grad is None for p in transformer.parameters())
        assert not transformer.transformer_blocks[0]._forward_hooks
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-5)
    assert all(p.grad is None for p in dino.model.parameters())


def test_real_scheduler_advances_only_in_outer_loop_and_can_keep_baseline():
    from diffusers import FlowMatchEulerDiscreteScheduler
    transformer = Transformer().eval()
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    pipe = SimpleNamespace(transformer=transformer, vae=VAE(), vae_scale_factor=1,
                           scheduler=scheduler,
                           _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    latents = torch.ones(1, 4, 3)
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
        steps=[0, 1], iterations=2, preservation_weight=1e10, resolution=(2, 2))
    for step, t in enumerate(scheduler.timesteps):
        before = scheduler.step_index
        with torch.no_grad():
            baseline = transformer(hidden_states=latents)[0]
            velocity = guide.predict(pipe, step, t, latents, dict(hidden_states=latents))
        assert scheduler.step_index == before
        assert guide.logs[-1]['selected_iteration'] == 0
        torch.testing.assert_close(velocity, baseline, rtol=0, atol=0)
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        assert scheduler.step_index == step + 1


def test_bf16_rounding_cannot_exceed_the_actual_change_budget():
    class Identity(torch.nn.Module):
        def forward(self, hidden_states):
            return (hidden_states,)
    # Around BF16 value 1, a small negative FP32 change can round to a larger
    # representable change. An FP32-only projection must not accept that state.
    latents = torch.ones(1, 4, 3, dtype=torch.bfloat16)
    pipe = SimpleNamespace(transformer=Identity().eval(), vae=VAE(), vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
        iterations=3, preservation_weight=0., max_relative_rms=.002,
        resolution=(2, 2), optimization_space='velocity')
    with torch.no_grad():
        actual = guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    candidates = [log for log in guide.logs if 'feasible' in log]
    assert any(not log['feasible'] for log in candidates)
    assert max(guide.logs[-1]['selected_relative_rms']) <= guide.cap
    actual_rms = (actual.float() - latents.float()).square().mean().sqrt()
    assert actual_rms <= guide.cap


def test_guidance_rejects_transformer_cache_before_forward():
    import pytest
    transformer = Transformer().eval()
    transformer.is_cache_enabled = True
    pipe = SimpleNamespace(transformer=transformer, vae=VAE())
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0)
    with pytest.raises(ValueError, match='caching'):
        guide.predict(pipe, 0, None, torch.ones(1, 4, 3), {})


def test_unbounded_last_matches_adam_even_when_the_last_update_is_worse():
    transformer = Transformer().eval()
    latents = torch.ones(1, 4, 3)
    pipe = SimpleNamespace(transformer=transformer, vae=VAE(), vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    direction = torch.tensor([.1, -.1, 0.])
    # Deliberately overshoot: this distinguishes "last" from silently keeping baseline.
    guide = CLSActivationGuidance(Dino(), direction, 0, iterations=1, learning_rate=2.,
        preservation_weight=0., max_relative_rms=None, selection='last',
        resolution=(2, 2), optimization_space='velocity')
    with torch.no_grad():
        baseline = transformer(hidden_states=latents)[0]
        scale = baseline.square().mean().sqrt()
        source = guide.cls_of_velocity(pipe, latents, baseline, torch.tensor(.75))
        target = F.normalize(source - direction, dim=-1)
    # Ordinary unconstrained Adam, without projection, penalty or best-candidate selection.
    u = torch.zeros_like(baseline, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=2.)
    loss = .5 * (guide.cls_of_velocity(pipe, latents, baseline + scale * u,
                                     torch.tensor(.75)) - target).square().sum(-1).mean()
    loss.backward()
    optimizer.step()
    expected = (baseline + scale * u).detach()
    with torch.no_grad():
        actual = guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    selected = guide.logs[-1]
    assert selected['selected_iteration'] == 1 and selected['selection'] == 'last'
    assert selected['selected_relative_rms'][0] > .05
    assert selected['final_semantic_loss'] > selected['baseline_semantic_loss']
    assert selected['selected_total_loss'] == selected['final_semantic_loss']
    assert guide.logs[0]['gradient_nonzero'] and guide.logs[0]['gradient_rms'] > 0
    assert not guide.logs[0]['projected']


def test_last_selection_still_respects_an_explicit_cap_after_rounding():
    import pytest
    class Identity(torch.nn.Module):
        def forward(self, hidden_states):
            return (hidden_states,)
    latents = torch.ones(1, 4, 3, dtype=torch.bfloat16)
    pipe = SimpleNamespace(transformer=Identity().eval(), vae=VAE(), vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
        iterations=1, preservation_weight=0., max_relative_rms=.002, selection='last',
        resolution=(2, 2), optimization_space='velocity')
    with pytest.raises(RuntimeError, match='Last iteration exceeds'):
        guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))


def test_fp32_correction_accumulates_updates_smaller_than_bf16_spacing():
    class Identity(torch.nn.Module):
        def forward(self, hidden_states):
            return (hidden_states,)
    latents = torch.ones(1, 4, 3, dtype=torch.bfloat16)
    pipe = SimpleNamespace(transformer=Identity().eval(), vae=VAE(), vae_scale_factor=1,
        scheduler=SimpleNamespace(sigmas=torch.tensor([.75, 0.])),
        _unpack_latents=lambda x, *args: x.transpose(1, 2).reshape(1, 3, 2, 2))
    guide = CLSActivationGuidance(Dino(), torch.tensor([.1, -.1, 0.]), 0,
        iterations=40, learning_rate=.0001, preservation_weight=0., max_relative_rms=None,
        selection='last', resolution=(2, 2), optimization_space='velocity')
    guide.predict(pipe, 0, None, latents, dict(hidden_states=latents))
    assert guide.logs[1]['fp32_relative_rms'][0] > 0
    assert guide.logs[1]['actual_relative_rms'][0] == 0
    assert guide.logs[-1]['selected_relative_rms'][0] > 0
