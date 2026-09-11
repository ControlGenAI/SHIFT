"""SHIFT pipeline IO and artifact compatibility checks."""
import hashlib
import json
import math
from pathlib import Path
import torch


def read_config(path):
    config = json.loads(Path(path).read_text())
    if not 0 <= config['step'] < config['inference_steps']:
        raise ValueError('Invalid step/block')
    if config['width'] % 16 or config['height'] % 16:
        raise ValueError('Image size must be divisible by 16')
    alphas = config.get('alphas', [])
    if not alphas or any(not math.isfinite(a) for a in alphas) or 0.0 not in alphas:
        raise ValueError('alphas must be finite and include 0 for the inverse control')
    if len(set(alphas)) != len(alphas):
        raise ValueError('alphas must be distinct')
    roi = config.get('roi')
    if (not isinstance(roi, list) or len(roi) != 4
            or not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1):
        raise ValueError('roi must be [left, top, right, bottom] inside [0,1]')
    if roi == [0.0, 0.0, 1.0, 1.0]:
        raise ValueError('A full-frame ROI leaves no outside region to measure')
    if config['blocks'] != 'all' and not isinstance(config['blocks'], list):
        raise ValueError('blocks must be "all" or a list of indices')
    return config


def signature(config):
    keys = ('model', 'model_revision', 'dino_model', 'dino_revision', 'dino_size',
            'width', 'height', 'step', 'inference_steps', 'guidance_scale')
    return {k: config.get(k) for k in keys}


def require_compatible(left, right):
    if signature(left) != signature(right):
        raise ValueError('Model, spatial grid, block or timestep settings differ between artifacts')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_pipeline(config, device):
    from src.models.flux import FluxPipeline
    pipe = FluxPipeline.from_pretrained(config['model'], revision=config.get('model_revision'),
                                         torch_dtype=torch.bfloat16).to(device)
    for model in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        model.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, config, prompt, seed, hook=None):
    kwargs = {}
    if hook is not None:
        kwargs['callback_on_step_end'] = hook.on_step_end
    return pipe(prompt, width=config['width'], height=config['height'],
                num_inference_steps=config['inference_steps'],
                guidance_scale=config['guidance_scale'], max_sequence_length=256,
                generator=torch.Generator('cpu').manual_seed(seed),
                structure_strength=0.0, txt_steering={'vector': None},
                **kwargs).images[0]


def load_adapter(path, device):
    from .adapter import DinoAdapter
    payload = torch.load(path, map_location='cpu', weights_only=True)
    adapter = DinoAdapter(**payload['spec']).to(device)
    adapter.load_state_dict(payload['state'])
    adapter.eval().requires_grad_(False)
    return adapter, payload


def selected_blocks(config, count):
    selected = list(range(count)) if config['blocks'] == 'all' else config['blocks']
    if not selected or len(set(selected)) != len(selected) or any(not isinstance(b, int) or b < 0 or b >= count for b in selected):
        raise ValueError('blocks must be all or a nonempty list of distinct valid indices')
    return selected
