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
    model_dtype, vae_dtype = pipeline_dtypes(config)
    return dict(**{k: config.get(k) for k in keys}, model_dtype=str(model_dtype), vae_dtype=str(vae_dtype))


def require_compatible(left, right):
    if signature(left) != signature(right):
        raise ValueError('Model, spatial grid, block or timestep settings differ between artifacts')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_image_dataset(root):
    """Apply the same explicit screening exclusions to adapter and CLS inputs."""
    root = Path(root)
    data = json.loads((root / 'dataset.json').read_text())
    path = root / 'excluded_pairs.json'
    if path.exists():
        ids = json.loads(path.read_text())['pair_ids']
        if not isinstance(ids, list) or any(type(i) not in (str, int) for i in ids):
            raise ValueError('excluded_pairs.json needs a list of pair IDs')
        excluded = set(ids)
        if excluded - {s['pair_id'] for s in data['samples']}:
            raise ValueError('excluded_pairs.json lists pairs that are not in the dataset')
        data['samples'] = [s for s in data['samples'] if s['pair_id'] not in excluded]
        data['excluded_pair_ids'] = sorted(excluded, key=str)
    return data


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pipeline_dtypes(config):
    allowed = {'bfloat16': torch.bfloat16, 'float32': torch.float32}
    model = config.get('model_dtype', 'bfloat16')
    vae = config.get('vae_dtype', model)
    if model not in allowed or vae not in allowed:
        raise ValueError('model_dtype and vae_dtype must be bfloat16 or float32')
    return allowed[model], allowed[vae]


def load_pipeline(config, device):
    from diffusers import AutoencoderKL
    from src.models.flux import FluxPipeline
    model_dtype, vae_dtype = pipeline_dtypes(config)
    components = {}
    if vae_dtype != model_dtype:
        # Load in the requested precision, rather than upcasting weights already
        # rounded by the pipeline's global BF16 loading option.
        components['vae'] = AutoencoderKL.from_pretrained(config['model'], subfolder='vae',
            revision=config.get('model_revision'), torch_dtype=vae_dtype)
    pipe = FluxPipeline.from_pretrained(config['model'], revision=config.get('model_revision'),
                                         torch_dtype=model_dtype, **components).to(device)
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
