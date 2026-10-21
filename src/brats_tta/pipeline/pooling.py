import numpy as np


def pool(p, q):
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] != 4:
        raise ValueError("Expected (4,N) probabilities")
    for x in (p, q):
        if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(0), 1, atol=2e-6, rtol=0):
            raise ValueError("Expected normalized finite probabilities")
    tiny = np.finfo(float).tiny
    lp, lq = np.log(np.maximum(p, tiny)), np.log(np.maximum(q, tiny))
    gap = (lp[1:].max(0) - lp[0] + lq[1:].max(0) - lq[0]) * 0.5
    foreground = q[1:].argmax(0).astype(np.uint8) + 1
    return dict(
        spatial_extent=np.where(gap > 0, foreground, 0).astype(np.uint8),
        hard_extent=np.where(p.argmax(0) > 0, foreground, 0).astype(np.uint8),
    )


def union(s, a, b):
    out = s.copy()
    only_a = (a != s) & (b == s)
    only_b = (b != s) & (a == s)
    agree = (a == b) & (a != s)
    out[only_a | agree] = a[only_a | agree]
    out[only_b] = b[only_b]
    return out
