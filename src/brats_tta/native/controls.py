"""Direction/count-matched margin, random and geometric diagnostic controls."""

import numpy as np


def _matched(source, pred, prob, seed, random=False):
    s = source.ravel()
    t = pred.ravel()
    p = prob.reshape(4, -1)
    out = s.copy()
    rng = np.random.default_rng(seed)
    for a in range(4):
        available = np.flatnonzero(s == a)
        for b in range(4):
            if a == b:
                continue
            k = int(((s == a) & (t == b)).sum())
            if not k:
                continue
            score = rng.random(len(available)) if random else p[b, available] - p[a, available]
            order = np.argsort(-score, kind="stable")
            take = order[:k]
            assert len(take) == k
            out[available[take]] = b
            available = available[order[k:]]
    for a in range(4):
        for b in range(4):
            if a != b:
                assert ((s == a) & (out == b)).sum() == ((s == a) & (t == b)).sum()
    return out.reshape(source.shape)


def matched(source, pred, prob, seed, random=False, eligible=None):
    if eligible is None:
        return _matched(source, pred, prob, seed, random)
    assert not ((pred != source) & ~eligible).any()
    out = source.copy()
    out[eligible] = _matched(source[eligible], pred[eligible], prob[:, eligible], seed, random)
    return out


def geometry(source, target, prob, eligible):
    # Same fixed control used in the previous full99 and structure experiments.
    from scipy.ndimage import distance_transform_edt

    assert not ((target != source) & ~eligible).any()
    wt = source > 0
    signed = distance_transform_edt(wt) - distance_transform_edt(~wt)
    etd = distance_transform_edt(source != 3) if (source == 3).any() else None
    out = source.copy()
    for src in range(4):
        available = (source == src) & eligible
        for dest in range(4):
            if dest == src:
                continue
            n = int(((source == src) & (target == dest)).sum())
            if not n:
                continue
            ids = np.flatnonzero(available)
            margin = (prob[dest] - prob[src]).ravel()[ids]
            score = (
                (-signed if dest == 0 else signed if dest == 1 else -np.abs(signed)).ravel()[ids]
                if dest != 3
                else (-etd.ravel()[ids] if etd is not None else margin)
            )
            take = ids[np.lexsort((ids, -margin, -score))[:n]]
            assert len(take) == n
            out.ravel()[take] = dest
            available.ravel()[take] = False
    return out
