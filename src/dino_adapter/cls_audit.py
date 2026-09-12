"""Audit saved one-step CLS targets without loading models or using a GPU."""
import json
from pathlib import Path

import torch

from .cls_objective import class_projection, cls_loss, projection_diagnostics
from .runtime import digest, save_json


def reference_scores(references, payload, alpha):
    if 'mean_negative' not in payload:
        return []  # Older/minimal direction artifacts may omit class centroids.
    scores = []
    for ref in references:
        if 'step' not in ref:
            continue  # The trajectory objective stores a different reference layout.
        step = ref['step']
        if payload.get('kind') == 'noised_one_step':
            index = payload['steps'].index(step)
            negative, direction = payload['mean_negative'][index], payload['directions'][index]
        else:
            negative, direction = payload['mean_negative'], payload['direction']
        if not torch.allclose(ref['direction'].float(), direction.float(), rtol=1e-4, atol=1e-6):
            raise ValueError('Saved CLS reference uses a different direction')
        if not torch.count_nonzero(direction):
            continue
        source, target, selected = [ref[k].float() for k in ('source_cls', 'target_cls', 'selected_cls')]
        scores.append(dict(step=step, sigma=ref['sigma'],
            source_class_score=class_projection(source, negative, direction),
            target_class_score=class_projection(target, negative, direction),
            selected_class_score=class_projection(selected, negative, direction),
            selected_semantic_loss=float(cls_loss(selected, target, direction, alpha, ref.get('cls_loss', 'shifted_cls'))),
            **projection_diagnostics(selected, source, target, direction, alpha)))
    return scores


def audit_saved_run(results, direction_path, output):
    root, path = Path(results), Path(output)
    if path.exists():
        raise FileExistsError(path)
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance['direction_sha256'] != digest(direction_path):
        raise ValueError('Use the exact direction artifact from this run')
    payload = torch.load(direction_path, map_location='cpu', weights_only=True)
    if 'mean_negative' not in payload:
        raise ValueError('Direction artifact lacks the negative train centroid')
    entries = json.loads((root / 'generations.json').read_text())
    reports = []
    for entry in entries:
        if 'alpha' not in entry or entry.get('objective', 'one_step') != 'one_step':
            continue
        image = Path(entry['image'])
        if image.name != str(image):
            raise ValueError('Expected an image filename inside the run directory')
        refs_path = root / f'{image.stem}_cls_targets.pt'
        references = torch.load(refs_path, map_location='cpu', weights_only=True)
        reports.append(dict(image=str(image), alpha=entry['alpha'],
            reference_sha256=digest(refs_path), steps=reference_scores(references, payload, entry['alpha'])))
    report = dict(direction_sha256=digest(direction_path), runs=reports,
        note='Scores use the saved float CLS: negative train centroid=0, positive=1. '
             'They are not probabilities or an independent check that glasses disappeared. '
             'Missing reference PT files in a figures selection require the original cluster output directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, report)
    print(json.dumps(report, indent=2), flush=True)
    return report
