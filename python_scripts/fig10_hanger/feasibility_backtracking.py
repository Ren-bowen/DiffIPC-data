"""Fig.10 rest-shape trials: orientation and successful finite forward solve."""
import numpy as np


def signed_tet_quality(vertices, tets):
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(tets, dtype=np.int64)]
    edges = np.stack([points[:, i] - points[:, 0] for i in (1, 2, 3)], axis=-1)
    denominator = np.prod(np.linalg.norm(edges, axis=1), axis=1)
    return np.divide(np.linalg.det(edges), denominator,
                     out=np.zeros_like(denominator), where=denominator > 1e-30)


def feasible_backtracking(base, delta, tets, evaluate, *, max_backtracks=24):
    """Return (vertices, scale, loss); evaluate must raise on failed rollouts.

    No loss-decrease condition. On exhaustion, re-evaluate the original shape
    to restore both mesh and simulation state, then return a zero step.
    """
    base = np.asarray(base, dtype=np.float64).copy()
    orientation = np.sign(signed_tet_quality(base, tets))
    if not np.isfinite(base).all() or not np.all(orientation != 0):
        raise ValueError('Invalid base tetrahedral mesh')
    scale = 1.0
    for _ in range(max_backtracks + 1):
        candidate = base + scale * delta
        if (np.isfinite(candidate).all()
                and np.all(signed_tet_quality(candidate, tets) * orientation > 1e-10)):
            try:
                loss = float(evaluate(candidate))
                if np.isfinite(loss):
                    return candidate, scale, loss
            except RuntimeError:
                pass
        scale *= 0.5
    loss = float(evaluate(base))
    if not np.isfinite(loss):
        raise RuntimeError('Fig.10 base rollout failed after rejected trials')
    return base, 0.0, loss


def needs_remesh(quality, quality_threshold, step_scale, step_threshold=0.25):
    return not np.isfinite(quality) or quality < quality_threshold or step_scale <= step_threshold
