"""Regression tests for failure modes found by probing the real FLUX pipeline.

Everything here targets something that actually went wrong, not generic tensor
behaviour: the adapter's inability to express the DINO target, invertibility
after training rather than at initialisation, the bf16 cast, stream ordering,
grid mismatches, resumable collection and the zero-alpha identity control.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.dino_adapter.adapter import DinoAdapter
from src.dino_adapter.hooks import ImageBlockHook
from src.dino_adapter.steering import AdapterEdit, ConstantEdit, DirectImageEdit

# Measured on real FLUX/DINO data in out/probe/probe.json: DINO patch features
# are L2 normalised, so each of the 1024 coordinates has std about 0.031.
DINO_COORD_STD = 0.031


def _excite(adapter, std=0.5):
    with torch.no_grad():
        for block in adapter.blocks:
            block.net[-1].weight.normal_(std=std)
            block.net[-1].bias.normal_(std=std)
    return adapter


def test_permutation_and_scale_bound_options_stay_exactly_invertible():
    """Both knobs change the function but must not break the inverse.

    Which setting aligns best is decided by measurement on real activations
    (out/arch_ablation.json), not here; this only guards the invariant.
    """
    torch.manual_seed(0)
    h = torch.randn(128, 32) * 3 + 1
    scale = float(h.square().mean().sqrt())
    outputs, errors = [], []
    for layers, permute, bound in [(4, False, 1.5), (8, True, 1.5), (8, True, 4.0)]:
        adapter = _excite(DinoAdapter(channels=32, z_dim=12, hidden=64, layers=layers,
                                      permute=permute, scale_bound=bound), std=0.1)
        z, r = adapter.encode(h)
        errors.append(float((adapter.decode(z, r) - h).abs().max()) / scale)
        outputs.append(z)
    assert max(errors) < 5e-3, f'inverse is not usable: relative errors {errors}'
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-4), 'permutation had no effect'
    assert not torch.allclose(outputs[1], outputs[2], atol=1e-4), 'scale_bound had no effect'
    # A loose scale bound buys capacity but costs fp32 round-trip precision,
    # which is why the inverse error is reported per block during training.
    assert errors[2] > errors[1]


def test_r2_is_zero_for_the_mean_predictor_and_negative_for_a_worse_one():
    """R2 is the metric that makes a tiny MSE interpretable; check its scale."""
    from src.dino_adapter.training import target_statistics
    torch.manual_seed(0)
    y = torch.nn.functional.normalize(torch.randn(256, 24), dim=-1)
    mean, variance = target_statistics([(torch.zeros(256, 8), y)])
    assert abs(1 - float((y - mean).square().mean()) / variance - 0.0) < 1e-6
    assert 1 - float(y.square().mean()) / variance < 0.0   # zero predictor is worse


def test_self_channel_leak_floor_versus_dino_target_scale():
    """The factor multiplying input channel i into z_i is bounded away from zero.

    Only depth shrinks it. At four couplings the floor is above the DINO target
    scale, so the leak swamps the signal; at eight it is well below.
    """
    def floor(layers, bound=1.5):
        return math.exp(-bound * (layers // 2))

    assert floor(4) > DINO_COORD_STD
    assert floor(8) < DINO_COORD_STD / 10


def test_roundtrip_is_exact_after_training_not_just_at_init():
    """Invertibility must survive nonzero weights, which zero-init hides."""
    torch.manual_seed(0)
    adapter = DinoAdapter(channels=16, z_dim=6, hidden=32, layers=8, permute=True)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    target = torch.nn.functional.normalize(torch.randn(64, 6), dim=-1)
    h = torch.randn(64, 16) * 5 + 2
    for _ in range(60):
        loss = (adapter(h) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert any(p.abs().max() > 1e-3 for p in adapter.parameters()), 'weights never moved'
    z, r = adapter.encode(h)
    assert float((adapter.decode(z, r) - h).abs().max()) < 1e-3
    # Editing z must leave r untouched, otherwise r is not preserved.
    edited = adapter.decode(z - 0.7 * torch.randn(6), r)
    torch.testing.assert_close(adapter.encode(edited)[1], r, atol=1e-4, rtol=1e-4)


def test_bf16_cast_does_not_erase_the_intervention():
    """A small fp32 edit can vanish entirely when cast back to the FLUX dtype."""
    torch.manual_seed(0)
    adapter = _excite(DinoAdapter(channels=16, z_dim=6, hidden=32, layers=8, permute=True), std=0.05)
    with torch.no_grad():
        adapter.spread.fill_(40.0)         # FLUX-scale activations
    h = (torch.randn(1, 32, 16) * 40).bfloat16()
    edit = AdapterEdit(adapter, torch.randn(32, 6) * 0.3, 1.0)
    edit(h)
    assert edit.stats['edit_rms'] > 0, 'intervention disappeared after the bf16 cast'
    assert edit.stats['roundtrip_rms'] / edit.stats['activation_rms'] < 1e-2
    zero = AdapterEdit(adapter, torch.randn(32, 6), 0.0)
    torch.testing.assert_close(zero(h).to(h.dtype), h, rtol=0, atol=0)


def test_hook_rejects_unexpected_token_count():
    """Guards against a swapped (img, txt) output or a wrong spatial grid."""
    class Swapped(torch.nn.Module):
        def forward(self, txt, img):
            return img, txt                # wrong order on purpose

    block = Swapped()
    model = SimpleNamespace(transformer_blocks=[block])
    with pytest.raises(ValueError, match='image tokens'):
        with ImageBlockHook(model, 0, 0, lambda h: h, expected_tokens=4):
            block(torch.zeros(1, 2, 8), torch.zeros(1, 4, 8))
    assert not block._forward_hooks


def test_edits_reject_grid_and_dimension_mismatch():
    adapter = DinoAdapter(channels=8, z_dim=3, hidden=8, layers=4)
    h = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match='token count'):
        AdapterEdit(adapter, torch.randn(9, 3), 1.0)(h)
    with pytest.raises(ValueError, match='dimension'):
        AdapterEdit(adapter, torch.randn(4, 5), 1.0)(h)
    with pytest.raises(ValueError, match='match'):
        DirectImageEdit(torch.randn(9, 8), 1.0)(h)
    with pytest.raises(ValueError, match='match'):
        ConstantEdit(torch.randn(9, 8), 0.1, 1.0)(h)


def test_decode_rejects_wrong_split():
    adapter = DinoAdapter(channels=8, z_dim=3, hidden=8, layers=4)
    with pytest.raises(ValueError, match='z/r split'):
        adapter.decode(torch.randn(2, 4), torch.randn(2, 4))


def test_validate_pairs_catches_real_data_mistakes():
    from src.dino_adapter.data import validate_pairs
    validate_pairs([dict(id='a', split='train', seed=1, positive='p', negative='n'),
                    dict(id='b', split='val', seed=2, positive='p', negative='n')])
    with pytest.raises(ValueError, match='glasses clause'):
        validate_pairs([dict(id='a', split='train', seed=1, positive='same', negative='same'),
                        dict(id='b', split='val', seed=2, positive='p', negative='n')])
    with pytest.raises(ValueError, match='missing'):
        validate_pairs([dict(id='a', split='train', positive='p', negative='n')])
    with pytest.raises(ValueError, match='seed cannot occur'):
        validate_pairs([dict(id='a', split='train', seed=7, positive='p', negative='n'),
                        dict(id='b', split='val', seed=7, positive='p', negative='n')])


def test_region_mask_matches_the_probed_token_rows_and_columns():
    from src.dino_adapter.directions import region_mask
    mask = region_mask((32, 32), [0.25, 0.3125, 0.75, 0.5625]).reshape(32, 32)
    rows = mask.any(1).nonzero().flatten()
    cols = mask.any(0).nonzero().flatten()
    assert (int(rows.min()), int(rows.max()) + 1) == (10, 18)
    assert (int(cols.min()), int(cols.max()) + 1) == (8, 24)


def test_read_config_rejects_unusable_roi(tmp_path):
    from src.dino_adapter.runtime import read_config
    base = json.loads(Path('configs/dino_adapter.json').read_text())
    read_config('configs/dino_adapter.json')
    for bad in ([0., 0., 1., 1.], [0.5, 0., 0.5, 1.], [0., 0., 2., 1.]):
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps({**base, 'roi': bad}))
        with pytest.raises(ValueError):
            read_config(path)


def test_matched_effect_skips_the_zero_alpha_identity_control():
    """alpha=0 constant control is the identity, so matching to it proves nothing."""
    source = Path('src/dino_adapter/steering.py').read_text()
    assert "r['alpha'] != 0" in source.split('matched = []')[1]


def _fake_pairs(path):
    path.write_text('\n'.join(json.dumps(r) for r in [
        dict(id='a', split='train', seed=1, positive='with glasses', negative='no glasses'),
        dict(id='b', split='val', seed=2, positive='with glasses', negative='no glasses')]))
    return path


def test_collect_resumes_instead_of_regenerating(tmp_path, monkeypatch):
    from PIL import Image
    from src.dino_adapter import data as data_module
    config = json.loads(Path('configs/dino_adapter.json').read_text())
    config.update(width=32, height=32, blocks=[0], inference_steps=2)
    pairs = _fake_pairs(tmp_path / 'pairs.jsonl')
    calls = []

    class Block(torch.nn.Module):
        def forward(self, txt, img):
            return txt, img + 1

    blocks = [Block()]
    pipe = SimpleNamespace(transformer=SimpleNamespace(transformer_blocks=blocks),
                           vae_scale_factor=8)
    monkeypatch.setattr(data_module, 'load_pipeline', lambda *a: pipe)

    def fake_generate(pipe_, cfg, prompt, seed, hook=None):
        calls.append(prompt)
        txt, img = torch.zeros(1, 3, 8), torch.zeros(1, 4, 8)
        for step in range(cfg['inference_steps']):
            for block in blocks:
                block(txt, img)
            if hook:
                hook.on_step_end(pipe_, step, None, {})
        return Image.new('RGB', (32, 32), (7, 7, 7))

    monkeypatch.setattr(data_module, 'generate', fake_generate)

    class FakeDino:
        def __init__(self, *a):
            pass

        def __call__(self, image, grid):
            n = grid[0] * grid[1]
            return torch.nn.functional.normalize(torch.randn(1, n, 3), dim=-1), torch.randn(1, 3)

    monkeypatch.setattr('src.dino_adapter.features.DinoFeatures', FakeDino)
    out = tmp_path / 'ds'
    data_module.collect(config, pairs, out, 'cpu')
    assert len(calls) == 4
    data_module.collect(config, pairs, out, 'cpu', resume=True)
    assert len(calls) == 4, 'resume regenerated already-collected samples'
    assert len(data_module.load_dataset(out)['samples']) == 4
