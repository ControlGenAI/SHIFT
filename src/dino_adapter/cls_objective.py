"""CLS objectives and diagnostics in units of a train mean difference."""
import torch
import torch.nn.functional as F


def validate_cls_loss(mode):
    if mode not in ('shifted_cls', 'projection'):
        raise ValueError('cls_loss must be shifted_cls or projection')


def cls_target(source, direction, alpha, mode='shifted_cls'):
    validate_cls_loss(mode)
    raw = source.detach().float() - alpha * direction.detach().float()
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite target CLS')
    if mode == 'projection':
        # Only its projection is a target. The full vector need not lie on the
        # unit sphere and is never used as a reconstruction/preservation loss.
        return raw
    if (raw.norm(dim=-1) < 1e-8).any():
        raise ValueError('Degenerate target CLS; reduce alpha')
    return F.normalize(raw, dim=-1)


def cls_loss(cls, target, direction, alpha, mode='shifted_cls'):
    validate_cls_loss(mode)
    error = cls.float() - target
    if mode == 'shifted_cls':
        return .5 * error.square().sum(-1).mean()
    unit = direction / direction.norm()
    # Positive alpha removes the positive concept; negative alpha adds it.
    # Once the requested projection is crossed, do not pull the image back.
    remaining = (error * unit).sum(-1) * (1 if alpha >= 0 else -1)
    return .5 * remaining.clamp_min(0).square().mean()


@torch.no_grad()
def projection_diagnostics(cls, source, target, direction, alpha):
    direction = direction.detach().float().to(cls.device)
    norm = direction.norm()
    unit = direction / norm
    source, target, cls = [x.detach().float().to(cls.device) for x in (source, target, cls)]
    source_score, target_score, score = [(x * unit).sum(-1) for x in (source, target, cls)]
    sign = 1 if alpha >= 0 else -1
    return dict(
        projected_removal=((source_score - score) / norm).cpu().tolist(),
        requested_projected_removal=((source_score - target_score) / norm).cpu().tolist(),
        remaining_mean_diffs=(sign * (score - target_score) / norm).clamp_min(0).cpu().tolist(),
        projection_target_reached=(sign * (score - target_score) <= 1e-6).cpu().tolist(),
        # This is a mathematical bound from unit-normalized CLS, not a cap on
        # the optimizer. Impossible targets are reported, never silently clipped.
        projection_target_reachable=(sign * target_score >= -1).cpu().tolist())


@torch.no_grad()
def class_projection(cls, mean_negative, direction):
    """Negative train centroid = 0, positive centroid = 1; NOT a probability."""
    cls = cls.detach().float().cpu()
    direction, mean_negative = [x.detach().float().cpu() for x in (direction, mean_negative)]
    if direction.shape != mean_negative.shape or cls.shape[-1:] != direction.shape:
        raise ValueError('Class means and CLS dimensions differ')
    if not torch.count_nonzero(direction):
        return None
    return (((cls - mean_negative) * direction).sum(-1) / direction.square().sum()).tolist()
