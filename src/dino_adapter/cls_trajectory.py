"""CLS editing of the final SHIFT image through the complete denoising trajectory.

One channel residual per double block is shared across image tokens and timesteps.
The optimizer differentiates through all sampler steps. Its evaluated velocities
are then replayed by the outer pipeline, which retains ownership of its scheduler.
"""
import copy
import math
from contextlib import contextmanager, ExitStack

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cls_guidance import CLSActivationGuidance, relative_rms
from .cls_images import decode_final, one_step_latents


def rollout(pipe, initial, kwargs):
    """Differentiable native scheduler on a private copy; no hand-coded Euler variant."""
    scheduler = copy.deepcopy(pipe.scheduler)
    states, velocities = [], []
    current = initial
    for timestep in scheduler.timesteps:
        states.append(current.detach().clone())
        inputs = dict(kwargs, hidden_states=current,
                      timestep=timestep.expand(current.shape[0]).to(current.dtype) / 1000)
        velocity = pipe.transformer(**inputs)[0]
        velocities.append(velocity.detach().clone())
        current = scheduler.step(velocity, timestep, current, return_dict=False)[0]
    return current, states, velocities


@contextmanager
def channel_outputs(blocks, references, corrections=None, scales=None):
    """Same intervention in every forward and checkpoint recomputation.

    The first baseline visit fixes each block's scale. Later interventions never
    recompute that scale from modified activations or detach the image stream.
    """
    with ExitStack() as stack:
        for index, block in blocks.items():
            def hook(module, inputs, output, index=index):
                if not isinstance(output, tuple) or len(output) != 2 or output[1].ndim != 3:
                    raise ValueError('Expected post-block (text, image[B,N,C])')
                text, image = output
                if corrections is None:
                    if index not in references:
                        references[index] = dict(shape=(image.shape[0], 1, image.shape[2]),
                            scale=image.detach().float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6))
                    return output
                u = corrections[index]
                scale = references[index]['scale'] if scales is None else scales[index]
                if (image.shape[0], 1, image.shape[2]) != tuple(u.shape):
                    raise ValueError('Channel residual must match batch and hidden channels')
                return text, (image.float() + scale * u).to(image.dtype)
            handle = block.register_forward_hook(hook)
            stack.callback(handle.remove)
        yield


def preservation_weights(rgb, roi, inside_weight):
    """ROI affects only the RGB preservation loss; all tokens remain editable."""
    height, width = rgb.shape[-2:]
    weights = rgb.new_ones((1, 1, height, width))
    if roi is not None:
        left, top, right, bottom = roi
        x0, x1 = math.floor(left * width), math.ceil(right * width)
        y0, y1 = math.floor(top * height), math.ceil(bottom * height)
        weights[..., y0:y1, x0:x1] = inside_weight
    return weights


class CLSTrajectoryGuidance(CLSActivationGuidance):
    """One fixed final-image target, one optimizer, one complete edited trajectory."""
    requires_static_conditioning = True
    def __init__(self, dino, direction, block, *, steps, alpha=1., iterations=20,
                 learning_rate=.001, resolution=(512, 512), image_preservation_weight=10.,
                 edit_roi=None, inside_weight=.05, view_scales=(1., .5),
                 dino_checkpointing=True, prediction_callback=None, iteration_callback=None,
                 reference_rgb=None, correction_scaling='rms', match_rms_adam=False):
        super().__init__(dino, direction, block, steps=steps, alpha=alpha,
            iterations=iterations, learning_rate=learning_rate, resolution=resolution,
            preservation_weight=0., max_relative_rms=None, selection='last',
            optimization_space='activation', block_mode='joint', prediction_callback=prediction_callback,
            correction_scaling=correction_scaling, match_rms_adam=match_rms_adam)
        if (not math.isfinite(image_preservation_weight) or image_preservation_weight < 0 or
                not math.isfinite(inside_weight) or not 0 <= inside_weight <= 1):
            raise ValueError('Invalid image preservation settings')
        if edit_roi is not None and (not isinstance(edit_roi, (list, tuple)) or len(edit_roi) != 4 or
                not all(math.isfinite(v) for v in edit_roi) or
                not 0 <= edit_roi[0] < edit_roi[2] <= 1 or not 0 <= edit_roi[1] < edit_roi[3] <= 1):
            raise ValueError('edit_roi must be null or normalized [left, top, right, bottom]')
        if edit_roi is not None and tuple(edit_roi) == (0., 0., 1., 1.) and inside_weight == 0:
            raise ValueError('Preservation weights cannot be zero everywhere; use image_preservation_weight=0 to disable')
        if (not isinstance(view_scales, (list, tuple)) or not view_scales or view_scales[0] != 1. or
                len(set(view_scales)) != len(view_scales) or
                any(not math.isfinite(v) or not 0 < v <= 1 for v in view_scales)):
            raise ValueError('view_scales must start with 1 and contain distinct finite scales in (0,1]')
        if type(dino_checkpointing) is not bool or (iteration_callback is not None and not callable(iteration_callback)):
            raise ValueError('Invalid checkpointing/callback settings')
        self.image_weight, self.edit_roi, self.inside_weight = image_preservation_weight, edit_roi, inside_weight
        self.view_scales, self.dino_checkpointing = tuple(view_scales), dino_checkpointing
        self.iteration_callback = iteration_callback
        if reference_rgb is not None and (reference_rgb.ndim != 4 or reference_rgb.shape[1] != 3 or
                tuple(reference_rgb.shape[-2:]) != tuple(resolution) or not torch.isfinite(reference_rgb).all() or
                (reference_rgb < 0).any() or (reference_rgb > 1).any()):
            raise ValueError('reference_rgb must be finite BCHW in [0,1] at the generation resolution')
        self.reference_rgb = None if reference_rgb is None else reference_rgb.detach().float().clone()
        self.planned_states, self.planned_velocities = None, None
        self.selected_rgb, self.baseline_rgb = None, None
        self._conditioning = None

    def validate_run(self, height, width, num_steps):
        super().validate_run(height, width, num_steps)
        if self.steps != set(range(num_steps)):
            raise ValueError('Final-image optimization must cover the complete schedule from step 0')
        if self.planned_states is not None or self.logs:
            raise ValueError('Create a fresh trajectory guidance instance for each generation')

    def view_features(self, rgb):
        features = []
        for scale in self.view_scales:
            size = tuple(max(1, round(s * scale)) for s in rgb.shape[-2:])
            view = rgb if scale == 1 else F.interpolate(rgb, size=size, mode='bicubic',
                                                       align_corners=False, antialias=True)
            if torch.is_grad_enabled() and self.dino_checkpointing:
                cls = checkpoint(self.dino.cls_from_rgb, view, use_reentrant=False)
            else:
                cls = self.dino.cls_from_rgb(view)
            features.append(cls)
        return torch.stack(features)

    def predict(self, pipe, step, timestep, latents, kwargs):
        from diffusers import FlowMatchEulerDiscreteScheduler
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Use no_grad, not inference_mode, for CLS trajectory optimization')
        for module in (pipe.transformer, pipe.vae, self.dino.model):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('FLUX, VAE and DINO must be frozen and in eval mode')
        if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler) or pipe.scheduler.config.get('stochastic_sampling', False):
            raise ValueError('Final-image optimization requires deterministic FlowMatchEulerDiscreteScheduler')
        if getattr(pipe.transformer, 'is_cache_enabled', False) or kwargs.get('joint_attention_kwargs'):
            raise ValueError('Final-image optimization requires uncached FLUX without attention extensions')
        if max(self.blocks) >= len(pipe.transformer.transformer_blocks):
            raise ValueError('Selected double block does not exist')
        if not torch.equal(torch.as_tensor(timestep).cpu(), pipe.scheduler.timesteps[step].cpu()):
            raise ValueError('Timestep does not match the scheduler index')
        if self.alpha == 0:
            with torch.no_grad():
                velocity = pipe.transformer(**kwargs)[0]
            self.logs.append(dict(step=step, alpha=0., bypass=True, objective='final',
                correction_scaling=self.correction_scaling, match_rms_adam=self.match_rms_adam))
            return velocity
        if step == 0:
            if pipe.scheduler.step_index not in (None, 0) or pipe.scheduler.begin_index not in (None, 0):
                raise ValueError('Final-image optimization must start at scheduler index 0')
            if self.planned_states is not None:
                raise ValueError('A trajectory was already optimized')
            self._conditioning = {k: v.detach().clone() for k, v in kwargs.items()
                                  if isinstance(v, torch.Tensor) and k not in ('hidden_states', 'timestep')}
            self._optimize_trajectory(pipe, latents, kwargs)
        if self.planned_states is None or not torch.equal(latents, self.planned_states[step]):
            raise RuntimeError('Outer latents differ from the evaluated trajectory; disable latent-changing callbacks')
        if any(k not in kwargs or not torch.equal(v, kwargs[k]) for k, v in self._conditioning.items()):
            raise ValueError('Conditioning changed during the evaluated trajectory')
        self.logs.append(dict(step=step, objective='final', replay=True,
                              sigma=float(pipe.scheduler.sigmas[step])))
        return self.planned_velocities[step]

    def _optimize_trajectory(self, pipe, initial, kwargs):
        index_before = pipe.scheduler.step_index
        blocks = {i: pipe.transformer.transformer_blocks[i] for i in self.blocks}
        block_references = {}
        with torch.no_grad(), channel_outputs(blocks, block_references):
            baseline_latents, baseline_states, baseline_velocities = rollout(pipe, initial, kwargs)
            _, initializer_rgb = decode_final(pipe, baseline_latents, self.resolution)
            reference_rgb = initializer_rgb if self.reference_rgb is None else self.reference_rgb.to(initial.device)
            if reference_rgb.shape != initializer_rgb.shape:
                raise ValueError('Reference image batch must match the generated batch')
            source = self.view_features(reference_rgb)
            if source.shape[-1:] != self.direction.shape:
                raise ValueError('CLS direction must match DINO hidden size')
            raw_target = source - self.alpha * self.direction.to(source.device)
            if not torch.isfinite(raw_target).all() or (raw_target.norm(dim=-1) < 1e-8).any():
                raise ValueError('Degenerate target CLS')
            target = F.normalize(raw_target, dim=-1).detach()
            initializer_cls = source if self.reference_rgb is None else self.view_features(initializer_rgb)
            baseline_loss = float(.5 * (initializer_cls - target).square().sum(-1).mean())
        if block_references.keys() != blocks.keys():
            raise RuntimeError('Not all selected double blocks were called')
        reference_scales = {i: r['scale'] for i, r in block_references.items()}
        scales = self.correction_scales(reference_scales)
        weights = preservation_weights(reference_rgb, self.edit_roi, self.inside_weight)
        self.baseline_rgb = reference_rgb.detach().cpu()
        with torch.enable_grad():
            corrections = {i: torch.zeros(r['shape'], device=initial.device, dtype=torch.float32, requires_grad=True)
                           for i, r in block_references.items()}
            optimizer, optimizer_groups = self.correction_optimizer(corrections, reference_scales, scales)
            for iteration in range(self.iterations + 1):
                # These hooks are deliberately active across ALL sampler steps and backward.
                with channel_outputs(blocks, block_references, corrections, scales=scales):
                    terminal, states, velocities = rollout(pipe, initial, kwargs)
                    decoded, rgb = decode_final(pipe, terminal, self.resolution)
                    cls = self.view_features(rgb)
                    semantic = .5 * (cls - target).square().sum(-1).mean()
                    pixel_loss = ((rgb - reference_rgb).square().mean(1, keepdim=True) * weights).sum() / (
                        weights.sum() * rgb.shape[0])
                    loss = semantic + self.image_weight * pixel_loss
                    if not torch.isfinite(loss) or not torch.isfinite(terminal).all():
                        raise RuntimeError('Nonfinite final-image objective')
                    entry = dict(iteration=iteration, objective='final', semantic_loss=float(semantic.detach()),
                        semantic_loss_by_view=(.5 * (cls.detach() - target).square().sum(-1).mean(-1)).cpu().tolist(),
                        image_preservation_loss=float(pixel_loss.detach()), total_loss=float(loss.detach()),
                        image_preservation_weight=self.image_weight,
                        pixel_mae=float((rgb.detach() - reference_rgb).abs().mean() * 255),
                        per_block=[dict(block=i,
                            fp32_relative_rms=(u.detach().square().mean((1, 2)).sqrt() *
                                              (scales[i] / reference_scales[i]).flatten()).cpu().tolist(),
                            fp32_absolute_rms=(u.detach().square().mean((1, 2)).sqrt() * scales[i].flatten()).cpu().tolist())
                                   for i, u in corrections.items()])
                    self.logs.append(entry)
                    if self.iteration_callback is not None:
                        with torch.no_grad():
                            self.iteration_callback(iteration, decoded.detach())
                    if iteration == self.iterations:
                        self.planned_states, self.planned_velocities = states, velocities
                        self.selected_rgb = rgb.detach().cpu()
                        self.references.append(dict(objective='final', view_scales=list(self.view_scales),
                            source_cls=source.detach().cpu(), target_cls=target.cpu(), selected_cls=cls.detach().cpu()))
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        gradients = torch.autograd.grad(loss, tuple(corrections.values()))
                        for row, u, gradient in zip(entry['per_block'], corrections.values(), gradients):
                            if not torch.isfinite(gradient).all():
                                raise RuntimeError('Nonfinite channel gradient')
                            row.update(gradient_rms=float(gradient.square().mean().sqrt()),
                                       gradient_nonzero=bool(torch.count_nonzero(gradient)))
                            u.grad = gradient
                        del gradients, gradient
                if iteration < self.iterations:
                    optimizer.step()
                    if any(not torch.isfinite(u).all() for u in corrections.values()):
                        raise RuntimeError('Nonfinite Adam channel correction')
                del terminal, states, velocities, decoded, rgb, cls, semantic, pixel_loss, loss
        if pipe.scheduler.step_index != index_before:
            raise RuntimeError('Outer scheduler advanced during trajectory optimization')
        self.logs.append(dict(objective='final', selected_iteration=self.iterations, selection='last',
            blocks=list(self.blocks), block_mode='joint', correction_layout='channel_shared_across_tokens_and_steps',
            optimized_parameters=sum(u.numel() for u in corrections.values()),
            baseline_semantic_loss=baseline_loss, final_semantic_loss=self.logs[-1]['semantic_loss'],
            image_preservation_weight=self.image_weight, edit_roi=self.edit_roi, inside_weight=self.inside_weight,
            view_scales=list(self.view_scales), max_relative_rms=None, activation_preservation_weight=0.,
            learning_rate=self.lr, iterations=self.iterations, correction_dtype='torch.float32',
            correction_scaling=self.correction_scaling, match_rms_adam=self.match_rms_adam,
            optimizer_groups=optimizer_groups,
            velocity_relative_rms_by_step=[relative_rms(v.float() - b.float(),
                b.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)).cpu().tolist()
                for v, b in zip(self.planned_velocities, baseline_velocities)]))
        if self.prediction_callback is not None:
            with torch.no_grad():
                for stage, states, velocities in (('before', baseline_states, baseline_velocities),
                                                 ('after', self.planned_states, self.planned_velocities)):
                    for step, (state, velocity) in enumerate(zip(states, velocities)):
                        clean = one_step_latents(state, velocity, pipe.scheduler.sigmas[step].to(state.device))
                        decoded, _ = decode_final(pipe, clean, self.resolution)
                        self.prediction_callback(step, stage, decoded)
