"""Real DINO CLS guidance through FLUX and its decoded one-step estimate.

Only an FP32 correction is optimized, either at a post-block image output or
at the velocity output. No learned CLS predictor, inverse map, scheduler step,
or image detachment is used within the differentiable objective.
"""
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F


def relative_rms(delta, reference_scale):
    return (delta.float().square().mean(dim=(1, 2), keepdim=True).sqrt() / reference_scale).flatten()


@contextmanager
def image_output(block, replacement=None, capture=None):
    """Replace only image output. Keep text and all spatial positions intact."""
    hits = []
    def hook(module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Expected post-block (text, image)')
        text, image = output
        if image.ndim != 3 or (replacement is not None and replacement.shape != image.shape):
            raise ValueError('Replacement must match all image tokens [B,N,C]')
        hits.append(1)
        if capture is not None:
            capture.append(image.detach().clone())
        return output if replacement is None else (text, replacement.to(image.dtype))
    handle = block.register_forward_hook(hook)
    try:
        yield
        if len(hits) != 1:
            raise RuntimeError('Expected exactly one selected block call')
    finally:
        handle.remove()


class CLSActivationGuidance:
    def __init__(self, dino, direction, block, steps=(0,), alpha=1., iterations=20,
                 learning_rate=.01, preservation_weight=1., max_relative_rms=.05,
                 resolution=(512, 512), optimization_space='activation', selection='best',
                 prediction_callback=None):
        if (type(block) is not int or type(iterations) is not int or
                not isinstance(steps, (list, tuple)) or any(type(s) is not int for s in steps)):
            raise ValueError('Block, steps and iterations must be integer indices/counts')
        if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or
                any(type(s) is not int or s <= 0 for s in resolution)):
            raise ValueError('Resolution must be positive integer (height, width)')
        if (iterations < 1 or learning_rate <= 0 or preservation_weight < 0 or
                block < 0 or not steps or min(steps) < 0 or
                len(set(steps)) != len(steps) or
                not all(math.isfinite(x) for x in (alpha, learning_rate, preservation_weight))):
            raise ValueError('Invalid CLS guidance settings')
        if max_relative_rms is not None and (not math.isfinite(max_relative_rms) or max_relative_rms <= 0):
            raise ValueError('max_relative_rms must be null or a finite positive number')
        if selection not in ('best', 'last'):
            raise ValueError('selection must be best or last')
        if prediction_callback is not None and not callable(prediction_callback):
            raise ValueError('prediction_callback must be callable')
        if optimization_space not in ('activation', 'velocity'):
            raise ValueError('optimization_space must be activation or velocity')
        if direction.ndim != 1 or not torch.isfinite(direction).all() or direction.norm() <= 1e-8:
            raise ValueError('Need a nonzero finite CLS mean-difference vector')
        self.dino, self.direction = dino, direction.detach().float()
        self.block, self.steps, self.alpha = block, set(steps), alpha
        self.iterations, self.lr = iterations, learning_rate
        self.preservation_weight, self.cap = preservation_weight, max_relative_rms
        self.resolution, self.space = resolution, optimization_space
        self.selection, self.prediction_callback = selection, prediction_callback
        self.logs, self.references = [], []

    def active(self, step):
        return step in self.steps

    def validate_run(self, height, width, num_steps):
        if (height, width) != tuple(self.resolution):
            raise ValueError('CLS resolution must match the pipeline height and width')
        if max(self.steps) >= num_steps:
            raise ValueError('CLS guidance step is outside the actual denoising schedule')

    def decode_velocity(self, pipe, latents, velocity, sigma):
        # Keep clean estimate subtraction in FP32 before casting for the VAE.
        clean = latents.float() - sigma.float() * velocity.float()
        height, width = self.resolution
        unpacked = pipe._unpack_latents(clean, height, width, pipe.vae_scale_factor)
        unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        return pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]

    def cls_of_velocity(self, pipe, latents, velocity, sigma):
        decoded = self.decode_velocity(pipe, latents, velocity, sigma)
        return self.dino.cls_from_rgb(decoded.float() / 2 + .5)

    def predict(self, pipe, step, timestep, latents, kwargs):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Use torch.no_grad, not inference_mode: CLS guidance needs autograd')
        for module in (pipe.transformer, pipe.vae, getattr(self.dino, 'model', None)):
            if isinstance(module, torch.nn.Module) and (module.training or any(p.requires_grad for p in module.parameters())):
                raise ValueError('FLUX, VAE and DINO must be frozen and in eval mode')
        if self.space == 'activation' and self.block >= len(pipe.transformer.transformer_blocks):
            raise ValueError('Selected double block does not exist')
        if getattr(pipe.transformer, 'is_cache_enabled', False):
            raise ValueError('Disable transformer caching for repeated differentiable CLS forwards')
        # This pipeline starts from scheduler index 0. Explicitly check that contract.
        if hasattr(pipe.scheduler, 'timesteps') and timestep is not None:
            if not torch.equal(torch.as_tensor(timestep).cpu(), pipe.scheduler.timesteps[step].cpu()):
                raise ValueError('Timestep does not match the scheduler index')
        index_before = getattr(pipe.scheduler, 'step_index', None)
        sigma = pipe.scheduler.sigmas[step].to(device=latents.device, dtype=torch.float32)
        if self.alpha == 0:
            with torch.no_grad():
                velocity = pipe.transformer(**kwargs)[0]
            self.logs.append(dict(step=step, alpha=0., bypass=True))
            return velocity
        captured = []
        if self.space == 'activation':
            block = pipe.transformer.transformer_blocks[self.block]
            with torch.no_grad(), image_output(block, capture=captured):
                baseline_velocity = pipe.transformer(**kwargs)[0].detach()
            original = captured.pop()
        else:
            with torch.no_grad():
                baseline_velocity = pipe.transformer(**kwargs)[0].detach()
            original = baseline_velocity
        base = original.float()
        scale = base.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        velocity_scale = baseline_velocity.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        with torch.no_grad():
            source_cls = self.cls_of_velocity(pipe, latents, baseline_velocity, sigma)
            direction = self.direction.to(source_cls.device)
            if direction.shape != source_cls.shape[-1:]:
                raise ValueError('CLS direction must match DINO hidden size')
            raw_target = source_cls - self.alpha * direction
            if not torch.isfinite(raw_target).all() or (raw_target.norm(dim=-1) < 1e-8).any():
                raise ValueError('Degenerate target CLS; reduce alpha')
            target = F.normalize(raw_target, dim=-1).detach()
            baseline_loss = float(.5 * (source_cls - target).square().sum(-1).mean())
            if self.prediction_callback is not None:
                self.prediction_callback(step, 'before', self.decode_velocity(pipe, latents, baseline_velocity, sigma))
        selected_loss, selected_index = baseline_loss, 0
        selected_velocity, selected_cls = baseline_velocity, source_cls
        selected_rms = [0.] * base.shape[0]

        def objective(u):
            modified = (base + scale * u).to(original.dtype)
            if self.space == 'activation':
                # Prefix has frozen inputs/weights. Gradients start at replacement H.
                with image_output(block, replacement=modified):
                    velocity = pipe.transformer(**kwargs)[0]
            else:
                velocity = modified
            cls = self.cls_of_velocity(pipe, latents, velocity, sigma)
            semantic = .5 * (cls - target).square().sum(-1).mean()
            # Penalize actual model-input changes, including BF16 rounding.
            applied = (modified.float() - base) / scale
            preserve = applied.square().mean()
            loss = semantic + self.preservation_weight * preserve
            actual_rms = relative_rms(modified.float() - base, scale)
            return loss, semantic, preserve, velocity, cls, actual_rms

        with torch.enable_grad():
            u = torch.zeros_like(base, requires_grad=True)
            optimizer = torch.optim.Adam([u], lr=self.lr)
            # Evaluate iterations 0..N; the final updated state is also assessed.
            for iteration in range(self.iterations + 1):
                loss, semantic, preserve, velocity, cls, actual_rms = objective(u)
                if not torch.isfinite(loss) or not torch.isfinite(velocity).all():
                    raise RuntimeError('Nonfinite CLS objective/velocity')
                value = float(loss.detach())
                feasible = self.cap is None or bool((actual_rms <= self.cap).all())
                # "last" takes every Adam update, even when the objective worsens.
                # A separately requested cap must never be silently violated.
                if self.selection == 'last' and iteration == self.iterations and not feasible:
                    raise RuntimeError('Last iteration exceeds max_relative_rms after model-dtype rounding')
                if ((self.selection == 'best' and value < selected_loss and feasible) or
                        (self.selection == 'last' and iteration == self.iterations)):
                    selected_loss, selected_index = value, iteration
                    selected_velocity, selected_cls = velocity.detach().clone(), cls.detach().clone()
                    selected_rms = actual_rms.detach().cpu().tolist()
                entry = dict(step=step, iteration=iteration,
                                      semantic_loss=float(semantic.detach()), total_loss=value,
                                      preservation_loss=float(preserve.detach()), feasible=feasible,
                                      actual_relative_rms=actual_rms.detach().cpu().tolist(),
                                      fp32_relative_rms=u.detach().square().mean((1, 2)).sqrt().cpu().tolist(),
                                      velocity_relative_rms=relative_rms(velocity.detach().float() - baseline_velocity.float(),
                                          velocity_scale).cpu().tolist())
                self.logs.append(entry)
                if iteration < self.iterations:
                    optimizer.zero_grad(set_to_none=True)
                    gradient = torch.autograd.grad(loss, u)[0]
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError('Nonfinite activation gradient')
                    entry.update(gradient_rms=float(gradient.square().mean().sqrt()),
                                 gradient_nonzero=bool(torch.count_nonzero(gradient)),
                                 projected=False)
                    u.grad = gradient
                    optimizer.step()
                    with torch.no_grad():
                        if not torch.isfinite(u).all():
                            raise RuntimeError('Nonfinite Adam correction')
                        if self.cap is not None:
                            norm = u.square().mean(dim=(1, 2), keepdim=True).sqrt()
                            entry['projected'] = bool((norm > self.cap * .99).any())
                            # Small margin for the following BF16 rounding.
                            u.mul_((self.cap * .99 / norm.clamp_min(1e-12)).clamp(max=1))
                    del gradient
                del loss, semantic, preserve, velocity, cls, actual_rms
        if getattr(pipe.scheduler, 'step_index', None) != index_before:
            raise RuntimeError('Scheduler advanced during inner optimization')
        self.logs.append(dict(step=step, selected_iteration=selected_index,
                              baseline_semantic_loss=baseline_loss, selected_total_loss=selected_loss,
                              selection=self.selection, max_relative_rms=self.cap,
                              preservation_weight=self.preservation_weight,
                              learning_rate=self.lr, iterations=self.iterations,
                              sigma=float(sigma), model_tensor_dtype=str(original.dtype),
                              correction_dtype=str(base.dtype),
                              selected_relative_rms=selected_rms,
                              final_semantic_loss=float(.5 * (selected_cls - target).square().sum(-1).mean()),
                              velocity_relative_rms=relative_rms(selected_velocity.float() - baseline_velocity.float(),
                                  velocity_scale).cpu().tolist()))
        if self.prediction_callback is not None:
            with torch.no_grad():
                self.prediction_callback(step, 'after', self.decode_velocity(pipe, latents, selected_velocity, sigma))
        self.references.append(dict(step=step, sigma=float(sigma), source_cls=source_cls.detach().cpu(),
                                    target_cls=target.cpu(), selected_cls=selected_cls.detach().cpu()))
        # This is the velocity actually evaluated for the selected intervention.
        # There is no update to latents here; the outer pipeline takes exactly one step.
        return selected_velocity
