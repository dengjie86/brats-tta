"""A single foreground offset preserving the source mean of a WT margin proxy."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def logit_gap(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 2 or z.shape[0] != 4 or not np.isfinite(z).all():
        raise ValueError('Expected finite four-class logits.')
    return z[1:].max(0) - z[0]


def probability_gap(probability):
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim < 2 or p.shape[0] != 4 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected nonnegative finite four-class probabilities.')
    if not np.allclose(p.sum(0), 1, atol=1e-6, rtol=0):
        raise ValueError('Probabilities do not sum to one.')
    floor = np.finfo(np.float64).tiny
    return np.log(np.maximum(p[1:].max(0), floor)) - np.log(np.maximum(p[0], floor))


def fit(gap, target):
    g = np.asarray(gap, dtype=np.float64)
    if not g.size or not np.isfinite(g).all() or not 0 < target < 1:
        raise ValueError('Finite nonempty gap and interior target required.')
    calls = 0

    def objective(b):
        nonlocal calls
        calls += 1
        return float(expit(g + b).mean() - target)

    before = objective(0.) + target
    if before == target:
        b = 0.
    else:
        bound = 1.
        while objective(-bound) > 0 or objective(bound) < 0:
            bound *= 2
            if bound > 1024:
                raise RuntimeError('Failed to bracket the unique bias.')
        b = brentq(objective, -bound, bound, xtol=1e-12,
                   rtol=4*np.finfo(np.float64).eps, maxiter=200)
    after = float(expit(g + b).mean())
    if abs(after-target) > 1e-10:
        raise RuntimeError('Moment residual exceeds the fixed tolerance.')
    return dict(bias=float(b), target=float(target), before=float(before), after=after,
                residual=abs(after-target), function_calls=calls)


def decode(probability, bias):
    p = np.asarray(probability)
    foreground = p[1:].argmax(0).astype(np.uint8) + 1
    return np.where(probability_gap(p) + bias > 0, foreground, 0).astype(np.uint8)
