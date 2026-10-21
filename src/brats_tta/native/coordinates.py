"""Native crop/original NIfTI array coordinate conversions."""

import numpy as np


def crop(a, bbox, fw):
    return a.transpose(2, 1, 0).transpose(fw)[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)].copy()


def restore(a, shape, bbox, fw):
    pre = np.zeros(tuple(np.array(shape[::-1])[fw]), a.dtype)
    pre[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)] = a
    return pre.transpose(tuple(np.argsort(fw))).transpose(2, 1, 0).copy()


def crop_patch(a, start, size=128):
    out = np.zeros(a.shape[:-3] + (size,) * 3, a.dtype)
    start, shape = np.asarray(start), np.asarray(a.shape[-3:])
    lo, hi = np.maximum(start, 0), np.minimum(start + size, shape)
    if np.all(hi > lo):
        out[
            (...,)
            + tuple(slice(int(low - offset), int(high - offset)) for low, high, offset in zip(lo, hi, start))
        ] = a[(...,) + tuple(slice(int(low), int(high)) for low, high in zip(lo, hi))]
    return out
