import json
from pathlib import Path

import pytest
import torch

from src.dino_adapter.cls_objective import cls_target, cls_loss, projection_diagnostics, class_projection
from src.dino_adapter.cls_guidance import CLSActivationGuidance
from src.dino_adapter.cls_audit import audit_saved_run, reference_scores
from test_cls_pipeline import tiny_pipeline


@pytest.mark.parametrize('alpha', [1., -1.])
def test_projection_accepts_concept_removal_despite_other_feature_changes(alpha):
    source = torch.tensor([[.6 * alpha, .8, 0.]])
    direction = torch.tensor([1.2, 0., 0.])
    target = cls_target(source, direction, alpha, 'projection')
    # Cross the concept threshold while changing the unrelated coordinates.
    edited = torch.tensor([[-.7 * alpha, 0., .51 ** .5]], requires_grad=True)
    loss = cls_loss(edited, target, direction, alpha, 'projection')
    assert loss == 0
    assert cls_loss(edited, cls_target(source, direction, alpha), direction, alpha) > .1
    before = source.clone().requires_grad_()
    gradient, = torch.autograd.grad(cls_loss(before, target, direction, alpha, 'projection'), before)
    assert gradient[0, 0] * alpha > 0
    assert not torch.count_nonzero(gradient[0, 1:])
    assert projection_diagnostics(edited, source, target, direction, alpha)['projection_target_reached'] == [True]


def test_projection_reports_unreachable_targets_without_clipping_or_rejecting_them():
    source, direction = torch.tensor([[1., 0.]]), torch.tensor([1., 0.])
    assert torch.equal(cls_target(source, direction, 1., 'projection'), torch.zeros_like(source))
    with pytest.raises(ValueError, match='Degenerate'):
        cls_target(source, direction, 1., 'shifted_cls')
    target = cls_target(source, direction, 5., 'projection')
    assert target[0, 0] == -4
    stats = projection_diagnostics(source, source, target, direction, 5.)
    assert stats['projection_target_reachable'] == [False]
    assert stats['remaining_mean_diffs'] == [5.]


def test_class_scores_use_train_centroids_and_are_not_probabilities():
    negative, direction = torch.tensor([-.6, .8]), torch.tensor([1.2, 0.])
    assert class_projection(torch.stack([negative, negative + direction]), negative, direction) == [0., 1.]
    assert class_projection((negative + 2 * direction)[None], negative, direction)[0] == pytest.approx(2.)
    assert class_projection(negative[None], negative, torch.zeros_like(direction)) is None


def test_cached_cls_extraction_excludes_screened_pairs_before_accessing_features(tmp_path, monkeypatch):
    from src.dino_adapter import cls_experiment
    from src.dino_adapter.runtime import load_image_dataset
    config = dict(dino_model='fake', dino_revision='pinned', dino_size=4)
    rows = [dict(id=f'{pair}_{label}', pair_id=pair, split='train', label=label,
                 features=f'{pair}_{label}.pt') for pair in ['keep', 'drop'] for label in [0, 1]]
    (tmp_path / 'dataset.json').write_text(json.dumps(dict(config=config, samples=rows)))
    (tmp_path / 'excluded_pairs.json').write_text(json.dumps(dict(pair_ids=['drop'])))
    for label in [0, 1]:
        torch.save(dict(cls=torch.tensor([float(label), float(1 - label)])), tmp_path / f'keep_{label}.pt')
    def forbidden(*args):
        raise AssertionError('Cached extraction must not load a model')
    monkeypatch.setattr(cls_experiment, 'DinoFeatures', forbidden)
    cls_experiment.extract(config, None, tmp_path, tmp_path / 'features', 'cpu', cached=True)
    payload = torch.load(tmp_path / 'features/cls_direction.pt', weights_only=True)
    assert payload['train_pair_ids'] == ['keep']
    assert torch.equal(payload['direction'], torch.tensor([1., -1.]))
    (tmp_path / 'excluded_pairs.json').write_text(json.dumps(dict(pair_ids=['drop', 'unknown'])))
    with pytest.raises(ValueError, match='not in the dataset'):
        load_image_dataset(tmp_path)


def test_cli_filters_heldout_pairs_and_records_explicit_sweep_overrides(tmp_path, monkeypatch):
    from src.dino_adapter import cls_experiment
    rows = [dict(id=f'{pair}_{label}', pair_id=pair, split='test', label=label, seed=i,
                 prompt=f'portrait {label}') for i, pair in enumerate(['drop', 'keep']) for label in [1, 0]]
    (tmp_path / 'dataset.json').write_text(json.dumps(dict(samples=rows)))
    (tmp_path / 'excluded_pairs.json').write_text(json.dumps(dict(pair_ids=['drop'])))
    seen = []
    monkeypatch.setattr(cls_experiment, 'optimize', lambda config, rows, *args: seen.append((config, rows)))
    config = Path(__file__).resolve().parents[1] / 'configs/cls_remove_projection_fp32.json'
    monkeypatch.setattr('sys.argv', ['cls_experiment', '--config', str(config), 'optimize',
        '--dataset', str(tmp_path), '--direction', 'unused.pt', '--output', str(tmp_path / 'output'),
        '--iterations', '40', '--learning-rate', '.00003', '--alphas', '0', '2'])
    cls_experiment.main()
    resolved, selected = seen[0]
    assert resolved['alphas'] == [0., 2.]
    assert resolved['cls_optimization']['iterations'] == 40
    assert resolved['cls_optimization']['learning_rate'] == .00003
    assert [r['id'] for r in selected] == ['keep_1']
    assert selected[0]['target_prompt'] == 'portrait 0'


@pytest.mark.parametrize('space', ['joint', 'activation', 'velocity'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_projection_guidance_scores_the_actual_terminal_image_and_keeps_zero_control(space, dtype):
    pipe, dino, kwargs, direction = tiny_pipeline(dtype)
    pipe.transformer.enable_gradient_checkpointing()
    options = dict(block=[0, 1] if space == 'joint' else 0,
        block_mode='joint' if space == 'joint' else 'independent',
        optimization_space='velocity' if space == 'velocity' else 'activation',
        steps=[kwargs['num_inference_steps'] - 1], iterations=3, learning_rate=.001,
        resolution=(8, 12), preservation_weight=0., max_relative_rms=None,
        selection='best', correction_scaling='none', cls_loss='projection')
    def run(guide=None):
        return pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(15), activation_guidance=guide).images
    baseline = run()
    assert torch.equal(baseline, run(CLSActivationGuidance(dino, direction, alpha=0., **options)))
    guide = CLSActivationGuidance(dino, direction, alpha=1., **options)
    image = run(guide)
    with torch.no_grad():
        actual_cls = dino.cls_from_rgb(image.float())
    ref = guide.references[-1]
    torch.testing.assert_close(actual_cls.cpu(), ref['selected_cls'], rtol=0, atol=0)
    assert guide.logs[-1]['final_semantic_loss'] == float(cls_loss(actual_cls, ref['target_cls'], direction, 1., 'projection'))
    assert guide.logs[-1]['selected_total_loss'] <= guide.logs[-1]['baseline_semantic_loss']
    assert any(row.get('gradient_nonzero') for row in guide.logs)
    assert all(not block._forward_hooks for block in pipe.transformer.transformer_blocks)
    assert all(p.grad is None for model in (pipe.transformer, pipe.vae, dino.model) for p in model.parameters())


def test_saved_reference_audit_handles_old_format_and_noised_levels(tmp_path):
    from src.dino_adapter.runtime import digest
    negative, direction = torch.tensor([-.6, .8]), torch.tensor([1.2, 0.])
    source = (negative + direction)[None]
    old = dict(step=3, sigma=.25, direction=direction, source_cls=source,
               target_cls=negative[None], selected_cls=negative[None])
    payload = dict(mean_negative=negative, direction=direction)
    report = reference_scores([old], payload, 1.)[0]
    assert report['source_class_score'] == [1.] and report['selected_class_score'] == [0.]
    assert report['selected_semantic_loss'] == 0
    noised = dict(kind='noised_one_step', steps=[0, 3], directions=torch.stack([torch.zeros_like(direction), direction]),
                  mean_negative=torch.stack([torch.ones_like(negative), negative]))
    assert reference_scores([old], noised, 1.) == [report]
    direction_path = tmp_path / 'direction.pt'
    torch.save(payload, direction_path)
    (tmp_path / 'provenance.json').write_text(json.dumps(dict(direction_sha256=digest(direction_path))))
    (tmp_path / 'generations.json').write_text(json.dumps([dict(image='baseline.png', mode='baseline'),
        dict(image='edited.png', alpha=1., objective='one_step')]))
    torch.save([old], tmp_path / 'edited_cls_targets.pt')
    saved = audit_saved_run(tmp_path, direction_path, tmp_path / 'audit.json')
    assert saved['runs'][0]['steps'] == [report]
    torch.save(dict(payload, direction=2 * direction), direction_path)
    with pytest.raises(ValueError, match='exact direction'):
        audit_saved_run(tmp_path, direction_path, tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


@pytest.mark.parametrize('filename', ['cls_remove_projection_fp32.json', 'cls_remove_shifted_fp32.json',
                                     'cls_remove_alpha_sweep.json'])
def test_removal_config_control_class_metrics_and_png_loss(tmp_path, monkeypatch, filename):
    from src.dino_adapter import cls_experiment
    config = cls_experiment.read_cls_config(Path(__file__).resolve().parents[1] / 'configs' / filename)
    assert config['alphas'] == [0., .5, 1., 2.]
    assert config['cls_optimization']['preservation_weight'] == 0 and config['cls_optimization']['max_relative_rms'] is None
    config.update(width=16, height=16, dino_size=8, alphas=[0., 2.])
    config['cls_optimization']['iterations'] = 1
    pipe, dino, inputs, direction = tiny_pipeline()
    prompts = []
    class PreparedPrompts:
        transformer = pipe.transformer
        image_processor = pipe.image_processor
        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            return pipe(prompt_embeds=inputs['prompt_embeds'], pooled_prompt_embeds=inputs['pooled_prompt_embeds'], **kwargs)
    monkeypatch.setattr(cls_experiment, 'load_pipeline', lambda *args: PreparedPrompts())
    monkeypatch.setattr(cls_experiment, 'DinoFeatures', lambda *args: dino)
    path = tmp_path / 'direction.pt'
    torch.save(dict(direction=direction, mean_negative=-direction / 2, train_pair_ids=['train'],
                    cls_signature=cls_experiment.cls_signature(config)), path)
    samples = [dict(id='p', pair_id='heldout', seed=200, prompt='with glasses', label=1, split='test'),
               dict(id='n', pair_id='heldout', seed=200, prompt='without glasses', label=0, split='test')]
    rows = cls_experiment.held_out_prompts(samples, 'test', include_target=True)
    out = tmp_path / 'out'
    cls_experiment.optimize(config, rows, path, out, 'cpu')
    assert prompts == ['with glasses', 'without glasses', 'with glasses', 'with glasses']
    entries = json.loads((out / 'generations.json').read_text())
    assert entries[1]['mode'] == 'target_prompt_only'
    assert entries[2]['baseline_pixel_max_abs'] == 0
    last = entries[-1]
    details = json.loads((out / Path(last['image']).with_suffix('.json')).read_text())
    assert len(details['cls_class_scores']) == 4
    assert len(last['final_cls_class_score']) == 1
    mode = config['cls_optimization']['cls_loss']
    assert details['cls_loss'] == mode
    from PIL import Image
    _, feature = dino(Image.open(out / last['image']), None)
    refs = torch.load(out / f"{Path(last['image']).stem}_cls_targets.pt", weights_only=True)
    assert last['final_png_target_loss'] == float(cls_loss(feature, refs[-1]['target_cls'], direction, 2., mode))
    audit = audit_saved_run(out, path, tmp_path / 'audit.json')
    assert audit['runs'][-1]['steps'] == details['cls_class_scores']
