import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.dino_adapter.adapter import DinoAdapter
from src.dino_adapter.hooks import ImageBlockHook
from src.dino_adapter.runtime import selected_blocks
from src.dino_adapter.steering import AdapterEdit, ConstantEdit


def nonlinear_adapter():
    torch.manual_seed(12)
    adapter = DinoAdapter(channels=8, z_dim=3, hidden=16, layers=4)
    with torch.no_grad():
        adapter.center.copy_(torch.randn(8))
        adapter.spread.copy_(torch.rand(8) + 0.5)
        for block in adapter.blocks:
            block.net[-1].weight.normal_(std=0.08)
            block.net[-1].bias.normal_(std=0.02)
    return adapter


def test_nonlinear_roundtrip_all_tokens_and_r_preserved():
    adapter = nonlinear_adapter()
    h = torch.randn(2, 7, 8)
    z, r = adapter.encode(h)
    assert z.shape == (2, 7, 3) and r.shape == (2, 7, 5)
    torch.testing.assert_close(adapter.decode(z, r), h, atol=2e-6, rtol=2e-6)
    v = torch.randn(7, 3) * .1
    edited = adapter.edit(h, v, .5)
    new_z, new_r = adapter.encode(edited)
    torch.testing.assert_close(new_z, z - .5 * v, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(new_r, r, atol=3e-6, rtol=3e-6)
    # Nonlinear coordinates produce different h displacements at different h.
    assert not torch.allclose(edited[0] - h[0], edited[1] - h[1])


def test_zero_alpha_really_inverts_and_bf16_roundtrip():
    adapter = nonlinear_adapter()
    h = torch.randn(1, 9, 8).bfloat16()
    edit = AdapterEdit(adapter, torch.randn(9, 3), 0.)
    result = edit(h)
    torch.testing.assert_close(result.to(h.dtype), h, rtol=0, atol=0)
    assert edit.stats['roundtrip_max_abs'] < 1e-5


def test_constant_control_matches_rms():
    h = torch.randn(1, 5, 8)
    edit = ConstantEdit(torch.randn(5, 8), .02, 1.)
    result = edit(h)
    assert abs(edit.stats['edit_rms'] - .02) < 1e-7
    assert torch.isfinite(result).all()


class Block(torch.nn.Module):
    def forward(self, txt, img):
        return txt, img + 1


def test_hook_step_gate_text_identity_and_removal():
    block = Block()
    model = SimpleNamespace(transformer_blocks=[block])
    txt, img = torch.randn(1, 2, 8), torch.randn(1, 4, 8)
    with ImageBlockHook(model, 0, 0, lambda h: h + 5) as hook:
        out_txt, out_img = block(txt, img)
        assert out_txt is txt
        torch.testing.assert_close(out_img, img + 6)
        hook.on_step_end(None, 0, None, {})
        _, out_img = block(txt, img)
        torch.testing.assert_close(out_img, img + 1)
        assert hook.captured.shape == (1, 4, 8) and hook.hits == 1
    assert not block._forward_hooks


def test_hook_rejects_repeated_call_and_cleans_up_on_exception():
    block = Block()
    with pytest.raises(RuntimeError, match='Multiple'):
        with ImageBlockHook(SimpleNamespace(transformer_blocks=[block]), 0, 0):
            block(torch.zeros(1, 1, 8), torch.zeros(1, 4, 8))
            block(torch.zeros(1, 1, 8), torch.zeros(1, 4, 8))
    assert not block._forward_hooks


def test_all_blocks_dynamic_and_no_single_blocks():
    assert selected_blocks({'blocks': 'all'}, 19) == list(range(19))
    with pytest.raises(ValueError):
        selected_blocks({'blocks': [19]}, 19)
    with pytest.raises(ValueError):
        selected_blocks({'blocks': [1, 1]}, 19)


def test_spatial_alignment_full_grid():
    from src.dino_adapter.features import align_patches
    patches = torch.eye(4).unsqueeze(0)
    aligned = align_patches(patches, (2, 2), (4, 4))
    assert aligned.shape == (1, 16, 4)
    torch.testing.assert_close(aligned[0, 0], patches[0, 0])
    torch.testing.assert_close(aligned[0, -1], patches[0, -1])
    torch.testing.assert_close(aligned.norm(dim=-1), torch.ones(1, 16))


def fake_dataset(tmp_path):
    config = json.loads(Path('configs/dino_adapter.json').read_text())
    config.update(width=32, height=32, dino_size=28, blocks='all')
    config['training'].update(epochs=2, token_batch=4, hidden=12, layers=2)
    root = tmp_path / 'data'
    root.mkdir()
    samples = []
    for i, split in enumerate(('train', 'train', 'val', 'test')):
        for label in (0, 1):
            name = f'p{i}_{label}'
            h = torch.randn(4, 8) + label
            y = torch.nn.functional.normalize(h[:, :3], dim=-1)
            torch.save({'patches': y, 'cls': y.mean(0)}, root / f'{name}_features.pt')
            blocks = {}
            for block in (0, 1):
                torch.save(h + block * .1, root / f'{name}_b{block}.pt')
                blocks[str(block)] = f'{name}_b{block}.pt'
            samples.append(dict(id=name, pair_id=f'p{i}', split=split, label=label,
                                seed=i, prompt='test', image=f'{name}.png',
                                features=f'{name}_features.pt', blocks=blocks))
    (root / 'dataset.json').write_text(json.dumps(dict(version=1, config=config, grid=[2, 2],
                                                      blocks=[0, 1], samples=samples)))
    return root, config


def test_training_and_directions_on_synthetic_tensors(tmp_path):
    from src.dino_adapter.training import train
    from src.dino_adapter.directions import build_directions
    from src.dino_adapter.runtime import load_adapter
    root, config = fake_dataset(tmp_path)
    train(config, root, tmp_path / 'adapters', 'cpu')
    build_directions(config, root, tmp_path / 'directions.pt')
    directions = torch.load(tmp_path / 'directions.pt', weights_only=True)
    assert directions['train_pair_ids'] == ['p0', 'p1']
    assert set(directions['vectors']) == {'0', '1'}
    for block in (0, 1):
        adapter, payload = load_adapter(tmp_path / 'adapters' / f'block_{block}.pt', 'cpu')
        assert payload['step'] == 0 and payload['block'] == block
        assert payload['train_pair_ids'] == ['p0', 'p1']
        assert payload['validation']['inverse_max_abs'] < 1e-5
        h = torch.randn(1, 4, 8)
        torch.testing.assert_close(adapter.edit(h, torch.ones(4, 3), 0), h, rtol=1e-5, atol=1e-5)


def test_data_split_leakage_rejected(tmp_path):
    from src.dino_adapter.data import load_dataset
    root, _ = fake_dataset(tmp_path)
    path = root / 'dataset.json'
    data = json.loads(path.read_text())
    data['samples'][-1]['split'] = 'train'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='exactly one split'):
        load_dataset(root)


def test_steer_uses_shift_style_hook_and_writes_all_block_controls(tmp_path, monkeypatch):
    from PIL import Image
    from src.dino_adapter.training import train
    from src.dino_adapter.directions import build_directions
    from src.dino_adapter import steering
    root, config = fake_dataset(tmp_path)
    config['alphas'] = [0., 1.]
    train(config, root, tmp_path / 'adapters', 'cpu')
    build_directions(config, root, tmp_path / 'directions.pt')
    model = SimpleNamespace(transformer_blocks=[Block(), Block()])
    calls = []
    monkeypatch.setattr(steering, 'load_pipeline', lambda *args: SimpleNamespace(transformer=model))
    def fake_generate(pipe, cfg, prompt, seed, hook=None):
        g = torch.Generator().manual_seed(seed)
        h, txt = torch.randn(1, 4, 8, generator=g), torch.zeros(1, 2, 8)
        for step in range(cfg['inference_steps']):
            for block in pipe.transformer.transformer_blocks:
                txt, h = block(txt, h)
            if hook:
                hook.on_step_end(pipe, step, None, {})
        calls.append(h.clone())
        return Image.new('RGB', (32, 32), (100, 100, 100))
    monkeypatch.setattr(steering, 'generate', fake_generate)
    steering.steer(config, root, tmp_path / 'adapters', tmp_path / 'directions.pt',
                   tmp_path / 'results', 'cpu')
    entries = json.loads((tmp_path / 'results/generations.json').read_text())
    assert len(entries) == 1 + 2 * 2 * 2  # baseline + blocks * alphas * modes
    assert {e['block'] for e in entries if e['mode'] == 'adapter'} == {0, 1}
    assert all(not b._forward_hooks for b in model.transformer_blocks)
    torch.testing.assert_close(calls[0], calls[1], rtol=1e-5, atol=1e-5)
    assert (tmp_path / 'results/visual_review.csv').exists()


def test_direct_image_edit_is_raw_post_block_shift():
    from src.dino_adapter.steering import DirectImageEdit
    h, direction = torch.randn(1, 4, 8), torch.randn(4, 8)
    torch.testing.assert_close(DirectImageEdit(direction, .5)(h), h - .5 * direction)
    torch.testing.assert_close(DirectImageEdit(direction, 0)(h), h, rtol=0, atol=0)
    torch.testing.assert_close(DirectImageEdit(direction, -1)(h), h + direction)


def test_image_only_mode_never_loads_adapter(tmp_path, monkeypatch):
    from PIL import Image
    from src.dino_adapter.directions import build_directions
    from src.dino_adapter import steering
    root, config = fake_dataset(tmp_path)
    config.update(steering_mode='image_tokens', alphas=[0., 1.])
    build_directions(config, root, tmp_path / 'directions.pt')
    model = SimpleNamespace(transformer_blocks=[Block(), Block()])
    monkeypatch.setattr(steering, 'load_pipeline', lambda *args: SimpleNamespace(transformer=model))
    def forbidden(*args):
        pytest.fail('Direct image steering must not load an adapter')
    monkeypatch.setattr(steering, 'load_adapter', forbidden)
    def fake_generate(pipe, cfg, prompt, seed, hook=None):
        for step in range(cfg['inference_steps']):
            for block in model.transformer_blocks:
                block(torch.zeros(1, 2, 8), torch.zeros(1, 4, 8))
            if hook:
                hook.on_step_end(pipe, step, None, {})
        return Image.new('RGB', (32, 32))
    monkeypatch.setattr(steering, 'generate', fake_generate)
    steering.steer(config, root, None, tmp_path / 'directions.pt', tmp_path / 'results', 'cpu')
    rows = json.loads((tmp_path / 'results/generations.json').read_text())
    assert len(rows) == 5
    assert {r['mode'] for r in rows} == {'baseline', 'image_tokens'}
