"""Score saved CLS experiments with the existing independent eyewear/face models."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import GlassesScorer, IdentityScorer
from .runtime import save_json


def evaluate_saved(results, output, device, with_identity=False):
    root, destination = Path(results), Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads((root / 'config.json').read_text())
    entries = json.loads((root / 'generations.json').read_text())
    if not entries:
        raise ValueError('No saved generations')
    for row in entries:
        if not (root / row['image']).is_file():
            raise FileNotFoundError(root / row['image'])
    baseline_rows = {r['sample_id']: r for r in entries if r['mode'] == 'baseline'}
    if any(r['sample_id'] not in baseline_rows for r in entries):
        raise ValueError('Every edited sample needs its source baseline')
    glasses = GlassesScorer(device=device)
    identity = IdentityScorer(device=device) if with_identity else None
    roi = config.get('cls_optimization', {}).get('edit_roi')
    def score(row):
        with Image.open(root / row['image']) as opened:
            image = opened.convert('RGB')
            return np.asarray(image, dtype=np.float32) / 255, glasses(image), identity.embed(image) if identity else None
    baselines = {sample: score(row) for sample, row in baseline_rows.items()}
    scored = []
    for row in entries:
        baseline_pixels, baseline_prob, baseline_face = baselines[row['sample_id']]
        pixels, probability, face = baselines[row['sample_id']] if row['mode'] == 'baseline' else score(row)
        if pixels.shape != baseline_pixels.shape:
            raise ValueError('Edited image and source baseline dimensions differ')
        delta = pixels - baseline_pixels
        outside = np.ones(pixels.shape[:2], dtype=bool)
        if roi is not None:
            height, width = outside.shape
            left, top, right, bottom = roi
            outside[int(np.floor(top * height)):int(np.ceil(bottom * height)),
                    int(np.floor(left * width)):int(np.ceil(right * width))] = False
        scored.append(dict(**row, glasses_probability=float(probability), baseline_glasses_probability=float(baseline_prob),
            predicted_glasses_removed=bool(baseline_prob >= glasses.threshold > probability),
            pixel_mae=float(np.abs(delta).mean() * 255),
            outside_roi_mse=float(np.square(delta[outside]).mean()) if outside.any() else None,
            identity_similarity=identity.similarity(face, baseline_face) if identity else None,
            face_detected=face is not None if identity else None))
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, dict(rows=scored, detector=dict(glasses_model=glasses.model_name,
        glasses_threshold=glasses.threshold, identity_model=identity.model_name if identity else None),
        note='Independent model predictions require visual checking; DINO CLS loss is not an eyewear detector.'))
    review = destination.with_suffix('.csv')
    if not review.exists():
        with review.open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image', 'mode', 'alpha', 'glasses_present', 'same_person', 'artifacts', 'notes'])
            for row in entries:
                writer.writerow([row['image'], row['mode'], row.get('alpha'), '', '', '', ''])
    return scored
