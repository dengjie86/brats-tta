"""Frozen-background observations and class-symmetric consensus (IN250 protocol).

Inputs are native-preprocessed four-channel arrays on the 1 mm grid.
The algorithm and constants match the October 8 development experiments.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

MODALITIES = ("t2f", "t1n", "t1c", "t2w")
VIEWS = ("blur_context", "neutral_context")


def teacher(source, probs, protected):
    accepted = protected.copy()
    labels = [p.argmax(0) for p in probs]
    for p, label in zip(probs, labels):
        top = np.sort(p, axis=0)[-2:]
        accepted &= (label == labels[0]) & (top[1] >= 0.8) & ((top[1] - top[0]) >= 0.2)
    out = source.copy()
    out[accepted] = labels[0][accepted]
    return out, accepted


def make_contexts(x, source):
    assert x.shape[0] == 4 and x.shape[1:] == source.shape
    if not (source > 0).any():
        return {m: x.copy() for m in VIEWS}, np.zeros(source.shape, bool), dict(reason="no_source_WT")
    distance = distance_transform_edt(source == 0)
    protected = distance <= 10
    far = distance >= 20
    z = np.clip((distance - 10) / 10, 0, 1).astype(np.float32)
    weight = z * z * (3 - 2 * z)
    masks = [x[c] != 0 for c in range(4)]
    if any(int((mask & far).sum()) < 1000 for mask in masks):
        return {m: x.copy() for m in VIEWS}, protected, dict(reason="insufficient_far_context")
    output = {m: np.empty_like(x) for m in VIEWS}
    stats = []
    for c, mask in enumerate(masks):
        roi = mask & far
        original = x[c][roi]
        den = gaussian_filter(mask.astype(np.float32), sigma=4, mode="constant", cval=0)
        blurred = gaussian_filter(x[c] * mask, sigma=4, mode="constant", cval=0) / np.maximum(den, 1e-6)
        a, b = float(blurred[roi].mean()), float(blurred[roi].std())
        mu, sd, med = float(original.mean()), float(original.std()), float(np.median(original))
        adjusted = (blurred - a) * (sd / max(b, 1e-6)) + mu
        for name, background in [("blur_context", adjusted), ("neutral_context", med)]:
            out = x[c] * (1 - weight) + background * weight
            out[~mask] = 0
            out[protected] = x[c][protected]
            output[name][c] = out
        stats.append(dict(n=int(roi.sum()), mean=mu, std=sd, median=med, blur_mean=a, blur_std=b))
    for out in output.values():
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out[:, protected], x[:, protected])
        np.testing.assert_array_equal(out[x == 0], 0)
    return (
        output,
        protected,
        dict(reason="ok", protected_voxels=int(protected.sum()), far_voxels=int(far.sum()), channels=stats),
    )
