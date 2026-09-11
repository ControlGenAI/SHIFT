"""Real DINO CLS guidance through FLUX and its decoded one-step estimate.

Only FP32 corrections are optimized, at one or several post-block image outputs
or at the velocity output. No learned CLS predictor, inverse map, scheduler step,
or image detachment is used within the differentiable objective.
"""
from contextlib import contextmanager, ExitStack
import math
import torch
import torch.nn.functional as F
from .cls_images import decode_latents, one_step_latents, pipeline_rgb


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


@contextmanager
def joint_image_outputs(blocks, corrections, scales, track_preservation=False, reference_scales=None):
    """Add residuals to LIVE outputs; keep hooks installed through backward.

    Replacing every output with a cached baseline would sever the gradient to
    earlier blocks. Checkpoint recomputation must see the same residual hooks
    as the forward, while diagnostics are recorded only on the initial pass.
    """
    stats = {}
    reference_scales = scales if reference_scales is None else reference_scales
    with ExitStack() as stack:
        for index, block in blocks.items():
            def hook(module, inputs, output, index=index):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError('Expected post-block (text, image)')
                text, image = output
                u, scale = corrections[index], scales[index]
                if image.ndim != 3 or image.shape != u.shape:
                    raise ValueError('Correction must match all image tokens [B,N,C]')
                modified = (image.float() + scale * u).to(image.dtype)
                if index not in stats or track_preservation:
                    # No extra autograd branch when the penalty is disabled.
                    with torch.set_grad_enabled(torch.is_grad_enabled() and track_preservation):
                        applied = (modified.float() - image.float()) / reference_scales[index]
                        squared_rms = applied.square().mean((1, 2))
                    if index not in stats:
                        stats[index] = dict(preservation=squared_rms.mean(),
                                            actual_rms=squared_rms.detach().sqrt())
                return text, modified
            handle = block.register_forward_hook(hook)
            stack.callback(handle.remove)
        yield stats
        if stats.keys() != blocks.keys():
            raise RuntimeError('Not all selected double blocks were called')


class CLSActivationGuidance:
    def __init__(self, dino, direction, block, steps=(0,), alpha=1., iterations=20,
                 learning_rate=.01, preservation_weight=1., max_relative_rms=.05,
                 resolution=(512, 512), optimization_space='activation', selection='best',
                 prediction_callback=None, block_mode='independent',
                 correction_scaling='rms', match_rms_adam=False, decode_mode='pipeline',
                 first_update_probe=()):
        if decode_mode not in ('pipeline', 'legacy_fp32_unclipped'):
            raise ValueError('decode_mode must be pipeline or legacy_fp32_unclipped')
        if (not isinstance(first_update_probe, (list, tuple)) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in first_update_probe) or
                len(set(first_update_probe)) != len(first_update_probe) or
                (first_update_probe and 0 not in first_update_probe)):
            raise ValueError('first_update_probe must contain distinct finite multipliers including zero, or be empty')
        if correction_scaling not in ('rms', 'none'):
            raise ValueError('correction_scaling must be rms or none')
        if type(match_rms_adam) is not bool or (match_rms_adam and correction_scaling != 'none'):
            raise ValueError('match_rms_adam must be boolean and requires correction_scaling=none')
        if block_mode not in ('independent', 'joint'):
            raise ValueError('block_mode must be independent or joint')
        if block_mode == 'joint':
            if optimization_space != 'activation':
                raise ValueError('Joint blocks require activation space')
            if (not isinstance(block, (list, tuple)) or not block or
                    any(type(b) is not int or b < 0 for b in block) or len(set(block)) != len(block)):
                raise ValueError('Joint blocks must be distinct nonnegative integer indices')
            blocks = tuple(block)
        else:
            if type(block) is not int or block < 0:
                raise ValueError('Block must be a nonnegative integer index')
            blocks = (block,)
        if (type(iterations) is not int or
                not isinstance(steps, (list, tuple)) or any(type(s) is not int for s in steps)):
            raise ValueError('Block, steps and iterations must be integer indices/counts')
        if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or
                any(type(s) is not int or s <= 0 for s in resolution)):
            raise ValueError('Resolution must be positive integer (height, width)')
        if (iterations < 1 or learning_rate <= 0 or preservation_weight < 0 or
                not steps or min(steps) < 0 or
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
        self.blocks, self.joint = blocks, block_mode == 'joint'
        self.iterations, self.lr = iterations, learning_rate
        self.preservation_weight, self.cap = preservation_weight, max_relative_rms
        self.resolution, self.space = resolution, optimization_space
        self.selection, self.prediction_callback = selection, prediction_callback
        self.correction_scaling, self.match_rms_adam = correction_scaling, match_rms_adam
        self.decode_mode = decode_mode
        self.first_update_probe = tuple(first_update_probe)
        self.logs, self.references = [], []

    def correction_scales(self, reference_scales):
        return {i: s if self.correction_scaling == 'rms' else torch.ones_like(s)
                for i, s in reference_scales.items()}

    def correction_optimizer(self, corrections, reference_scales, scales):
        """Optional coordinate-change control: delta=s*u, lr_delta=s*lr, eps_delta=eps/s.

        Plain `none` uses ordinary Adam in the tensor's original units. Matching
        is explicit and limited to batch 1 because Adam groups have scalar LR.
        """
        groups, metadata = [], []
        for i, u in corrections.items():
            reference = reference_scales[i]
            if self.match_rms_adam and reference.numel() != 1:
                raise ValueError('match_rms_adam requires batch size 1; run samples separately')
            factor = float(reference) if self.match_rms_adam else 1.
            lr, eps = self.lr * factor, 1e-8 / factor
            groups.append(dict(params=[u], lr=lr, eps=eps))
            metadata.append(dict(block=i if self.space == 'activation' else None,
                reference_rms=reference.flatten().cpu().tolist(),
                correction_scale=scales[i].flatten().cpu().tolist(), learning_rate=lr, epsilon=eps))
        return torch.optim.Adam(groups, lr=self.lr), metadata

    def active(self, step):
        return step in self.steps

    def _probe_first_update(self, objective, corrections, optimizer, initial_loss,
                            baseline_velocity, velocity_scale, step):
        """Evaluate signed first-Adam directions; restore zeros and preserve Adam state.

        This diagnostic adds forwards and one repeated backward. It never accepts,
        rejects, clips, or changes an optimizer update.
        """
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        with torch.no_grad():
            directions = {i: -groups[id(u)]['lr'] * u.grad / (u.grad.abs() + groups[id(u)]['eps'])
                          for i, u in corrections.items()}
            slope = sum(float((u.grad * directions[i]).sum()) for i, u in corrections.items())
        try:
            for multiplier in self.first_update_probe:
                with torch.no_grad():
                    for i, u in corrections.items():
                        u.copy_(directions[i] * multiplier)
                with objective(corrections) as result:
                    loss, semantic, preserve, velocity, cls, rms, block_rms, image_stats = result
                    if not torch.isfinite(loss) or not torch.isfinite(velocity).all():
                        raise RuntimeError('Nonfinite first-update diagnostic')
                    row = dict(step=step, phase='first_update_probe', multiplier=multiplier,
                        total_loss=float(loss.detach()), semantic_loss=float(semantic.detach()),
                        observed_loss_delta=float(loss.detach()) - initial_loss,
                        predicted_linear_loss_delta=multiplier * slope,
                        actual_relative_rms=rms.detach().cpu().tolist(), **image_stats,
                        velocity_relative_rms=relative_rms(velocity.detach().float() - baseline_velocity.float(),
                                                          velocity_scale).cpu().tolist())
                    if multiplier == 0:
                        repeated = torch.autograd.grad(loss, tuple(corrections.values()))
                        row['repeat_gradients'] = []
                        for (i, u), gradient in zip(corrections.items(), repeated):
                            if not torch.isfinite(gradient).all():
                                raise RuntimeError('Nonfinite repeated diagnostic gradient')
                            old_norm, new_norm = u.grad.norm(), gradient.norm()
                            denom = old_norm * new_norm
                            row['repeat_gradients'].append(dict(block=i,
                                relative_l2=float((gradient - u.grad).norm() / old_norm.clamp_min(1e-20)),
                                cosine=float((gradient * u.grad).sum() / denom) if denom > 0 else None))
                        del repeated, gradient
                self.logs.append(row)
                del result, loss, semantic, preserve, velocity, cls, rms, block_rms
        finally:
            with torch.no_grad():
                for u in corrections.values():
                    u.zero_()

    def validate_run(self, height, width, num_steps):
        if (height, width) != tuple(self.resolution):
            raise ValueError('CLS resolution must match the pipeline height and width')
        if max(self.steps) >= num_steps:
            raise ValueError('CLS guidance step is outside the actual denoising schedule')

    def decode_velocity(self, pipe, latents, velocity, sigma):
        if self.decode_mode == 'pipeline':
            return decode_latents(pipe, one_step_latents(latents, velocity, sigma), self.resolution)
        # Reproduce older experiments, including their different BF16 rounding.
        clean = latents.float() - sigma.float() * velocity.float()
        height, width = self.resolution
        unpacked = pipe._unpack_latents(clean, height, width, pipe.vae_scale_factor)
        unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        return pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]

    def cls_of_velocity(self, pipe, latents, velocity, sigma):
        decoded = self.decode_velocity(pipe, latents, velocity, sigma)
        return self.dino.cls_from_rgb(self.rgb_of_decoded(pipe, decoded))

    def rgb_of_decoded(self, pipe, decoded):
        return pipeline_rgb(pipe, decoded) if self.decode_mode == 'pipeline' else decoded.float() / 2 + .5

    def predict(self, pipe, step, timestep, latents, kwargs):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Use torch.no_grad, not inference_mode: CLS guidance needs autograd')
        for module in (pipe.transformer, pipe.vae, getattr(self.dino, 'model', None)):
            if isinstance(module, torch.nn.Module) and (module.training or any(p.requires_grad for p in module.parameters())):
                raise ValueError('FLUX, VAE and DINO must be frozen and in eval mode')
        if self.space == 'activation' and max(self.blocks) >= len(pipe.transformer.transformer_blocks):
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
            self.logs.append(dict(step=step, alpha=0., bypass=True,
                correction_scaling=self.correction_scaling, match_rms_adam=self.match_rms_adam,
                decode_mode=self.decode_mode))
            return velocity
        first = self.blocks[0]
        if self.space == 'activation':
            blocks = {i: pipe.transformer.transformer_blocks[i] for i in self.blocks}
            captured = {i: [] for i in self.blocks}
            with torch.no_grad(), ExitStack() as stack:
                for i, block in blocks.items():
                    stack.enter_context(image_output(block, capture=captured[i]))
                baseline_velocity = pipe.transformer(**kwargs)[0].detach()
            originals = {i: captured[i].pop() for i in self.blocks}
        else:
            with torch.no_grad():
                baseline_velocity = pipe.transformer(**kwargs)[0].detach()
            originals = {first: baseline_velocity}
        reference_scales = {i: h.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
                            for i, h in originals.items()}
        scales = self.correction_scales(reference_scales)
        layouts = {i: (h.shape, h.device) for i, h in originals.items()}
        model_dtype = originals[first].dtype
        if not self.joint:
            base, scale = originals[first].float(), scales[first]
        # Joint residuals need shapes and fixed RMS scales, not cached activations.
        del originals
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
        selected_rms = [0.] * latents.shape[0]
        selected_block_rms = {i: list(selected_rms) for i in self.blocks}

        @contextmanager
        def objective(corrections):
            # In joint mode this context remains open during autograd.grad, so
            # checkpoint backward recomputes the same intervened transformer.
            with ExitStack() as stack:
                if self.joint:
                    stats = stack.enter_context(joint_image_outputs(
                        blocks, corrections, scales, self.preservation_weight != 0,
                        reference_scales=reference_scales))
                    velocity = pipe.transformer(**kwargs)[0]
                    if stats.keys() != blocks.keys():
                        raise RuntimeError('Not all selected double blocks were called')
                    block_rms = {i: stats[i]['actual_rms'] for i in self.blocks}
                    preserve = torch.stack([stats[i]['preservation'] for i in self.blocks]).mean()
                    # Aggregate diagnostics; an optional cap is enforced PER block.
                    actual_rms = torch.stack(list(block_rms.values())).square().mean(0).sqrt()
                else:
                    modified = (base + scale * corrections[first]).to(model_dtype)
                    if self.space == 'activation':
                        # Frozen prefix; gradients start at the one replacement H.
                        with image_output(blocks[first], replacement=modified):
                            velocity = pipe.transformer(**kwargs)[0]
                    else:
                        velocity = modified
                    applied = (modified.float() - base) / reference_scales[first]
                    preserve = applied.square().mean()
                    actual_rms = relative_rms(modified.float() - base, reference_scales[first])
                    block_rms = {first: actual_rms}
                decoded = self.decode_velocity(pipe, latents, velocity, sigma)
                rgb = self.rgb_of_decoded(pipe, decoded)
                cls = self.dino.cls_from_rgb(rgb)
                with torch.no_grad():
                    image_stats = dict(decoded_out_of_range_fraction=float((decoded.abs() > 1).float().mean()),
                                       dino_rgb_min=float(rgb.min()), dino_rgb_max=float(rgb.max()))
                semantic = .5 * (cls - target).square().sum(-1).mean()
                loss = semantic if self.preservation_weight == 0 else semantic + self.preservation_weight * preserve
                yield loss, semantic, preserve, velocity, cls, actual_rms, block_rms, image_stats

        with torch.enable_grad():
            corrections = {i: torch.zeros(shape, device=device, dtype=torch.float32, requires_grad=True)
                           for i, (shape, device) in layouts.items()}
            optimizer, optimizer_groups = self.correction_optimizer(corrections, reference_scales, scales)
            # Evaluate iterations 0..N; the final updated state is also assessed.
            for iteration in range(self.iterations + 1):
                with objective(corrections) as result:
                    loss, semantic, preserve, velocity, cls, actual_rms, block_rms, image_stats = result
                    if not torch.isfinite(loss) or not torch.isfinite(velocity).all():
                        raise RuntimeError('Nonfinite CLS objective/velocity')
                    value = float(loss.detach())
                    feasible = self.cap is None or all(bool((rms <= self.cap).all()) for rms in block_rms.values())
                    if self.selection == 'last' and iteration == self.iterations and not feasible:
                        raise RuntimeError('Last iteration exceeds max_relative_rms after model-dtype rounding')
                    if ((self.selection == 'best' and value < selected_loss and feasible) or
                            (self.selection == 'last' and iteration == self.iterations)):
                        selected_loss, selected_index = value, iteration
                        selected_velocity, selected_cls = velocity.detach().clone(), cls.detach().clone()
                        selected_rms = actual_rms.detach().cpu().tolist()
                        selected_block_rms = {i: rms.detach().cpu().tolist() for i, rms in block_rms.items()}
                    fp32_rms = {i: u.detach().square().mean((1, 2)).sqrt() *
                                (scales[i] / reference_scales[i]).flatten() for i, u in corrections.items()}
                    entry = dict(step=step, iteration=iteration, **image_stats,
                                      semantic_loss=float(semantic.detach()), total_loss=value,
                                      preservation_loss=float(preserve.detach()), feasible=feasible,
                                      actual_relative_rms=actual_rms.detach().cpu().tolist(),
                                      fp32_relative_rms=torch.stack(list(fp32_rms.values())).square().mean(0).sqrt().cpu().tolist(),
                                      velocity_relative_rms=relative_rms(velocity.detach().float() - baseline_velocity.float(),
                                          velocity_scale).cpu().tolist())
                    if self.joint:
                        entry['per_block'] = [dict(block=i, actual_relative_rms=block_rms[i].cpu().tolist(),
                            fp32_relative_rms=fp32_rms[i].cpu().tolist(),
                            actual_absolute_rms=(block_rms[i] * reference_scales[i].flatten()).cpu().tolist(),
                            fp32_absolute_rms=(fp32_rms[i] * reference_scales[i].flatten()).cpu().tolist())
                            for i in self.blocks]
                    else:
                        entry['actual_absolute_rms'] = (actual_rms * reference_scales[first].flatten()).detach().cpu().tolist()
                        entry['fp32_absolute_rms'] = (fp32_rms[first] * reference_scales[first].flatten()).cpu().tolist()
                    self.logs.append(entry)
                    if iteration < self.iterations:
                        optimizer.zero_grad(set_to_none=True)
                        gradients = torch.autograd.grad(loss, tuple(corrections.values()))
                        if any(not torch.isfinite(g).all() for g in gradients):
                            raise RuntimeError('Nonfinite activation gradient')
                        entry.update(gradient_rms=float(torch.stack([g.square().mean() for g in gradients]).mean().sqrt()),
                                     gradient_nonzero=any(bool(torch.count_nonzero(g)) for g in gradients),
                                     projected=False)
                        for index, (u, gradient) in enumerate(zip(corrections.values(), gradients)):
                            u.grad = gradient
                            if self.joint:
                                entry['per_block'][index].update(gradient_rms=float(gradient.square().mean().sqrt()),
                                    gradient_nonzero=bool(torch.count_nonzero(gradient)), projected=False)
                        del gradients, gradient
                # All residual hooks are removed before changing their parameters.
                if iteration < self.iterations:
                    if iteration == 0 and self.first_update_probe:
                        self._probe_first_update(objective, corrections, optimizer, value,
                                                 baseline_velocity, velocity_scale, step)
                    optimizer.step()
                    with torch.no_grad():
                        for index, (i, u) in enumerate(corrections.items()):
                            if not torch.isfinite(u).all():
                                raise RuntimeError('Nonfinite Adam correction')
                            if self.cap is not None:
                                norm = u.square().mean(dim=(1, 2), keepdim=True).sqrt() * (scales[i] / reference_scales[i])
                                projected = bool((norm > self.cap * .99).any())
                                entry['projected'] |= projected
                                if self.joint:
                                    entry['per_block'][index]['projected'] = projected
                                u.mul_((self.cap * .99 / norm.clamp_min(1e-12)).clamp(max=1))
                del result, loss, semantic, preserve, velocity, cls, actual_rms, block_rms
        if getattr(pipe.scheduler, 'step_index', None) != index_before:
            raise RuntimeError('Scheduler advanced during inner optimization')
        self.logs.append(dict(step=step, selected_iteration=selected_index,
                              baseline_semantic_loss=baseline_loss, selected_total_loss=selected_loss,
                              selection=self.selection, max_relative_rms=self.cap,
                              preservation_weight=self.preservation_weight,
                              learning_rate=self.lr, iterations=self.iterations,
                              correction_scaling=self.correction_scaling, match_rms_adam=self.match_rms_adam,
                              decode_mode=self.decode_mode,
                              optimizer_groups=optimizer_groups,
                              sigma=float(sigma), model_tensor_dtype=str(model_dtype),
                              correction_dtype='torch.float32',
                              selected_relative_rms=selected_rms,
                              final_semantic_loss=float(.5 * (selected_cls - target).square().sum(-1).mean()),
                              velocity_relative_rms=relative_rms(selected_velocity.float() - baseline_velocity.float(),
                                  velocity_scale).cpu().tolist()))
        if self.joint:
            self.logs[-1].update(block_mode='joint', blocks=list(self.blocks),
                selected_per_block=[dict(block=i, selected_relative_rms=selected_block_rms[i]) for i in self.blocks])
        if self.prediction_callback is not None:
            with torch.no_grad():
                self.prediction_callback(step, 'after', self.decode_velocity(pipe, latents, selected_velocity, sigma))
        self.references.append(dict(step=step, sigma=float(sigma), source_cls=source_cls.detach().cpu(),
                                    target_cls=target.cpu(), selected_cls=selected_cls.detach().cpu()))
        # This is the velocity actually evaluated for the selected intervention.
        # There is no update to latents here; the outer pipeline takes exactly one step.
        return selected_velocity
