import numpy as np
from scipy.special import softmax
from scipy.ndimage import convolve, generate_binary_structure
from brats_tta.native.coordinates import crop, restore
from . import projection as frozen_projection, pooling as frozen_pool
from . import support as frozen_filter, presence as frozen_guard

mods = dict(projection=frozen_projection, pool=frozen_pool, filter=frozen_filter, guard=frozen_guard)


def spatial(data, student, bias, independent=False):
    tiny = np.finfo(np.float64).tiny
    d = data
    p = d["p"]
    s = p.argmax(0).astype(np.uint8)
    endpoint = mods["projection"].decode(student, bias)
    endpoint[d["accepted"]] = d["qteacher"].argmax(0)[d["accepted"]]
    idx = np.flatnonzero(endpoint != s)
    # Preserve the historical disagreement-only construction, including numerical order.
    qs = np.log(np.maximum(student.reshape(4, -1)[:, idx].astype(np.float64), tiny))
    qs[1:] += bias
    q = softmax(qs, axis=0)
    selected = d["accepted"].ravel()[idx]
    q[:, selected] = d["qteacher"].reshape(4, -1)[:, idx][:, selected]
    pp = p.reshape(4, -1)[:, idx].astype(np.float64)
    if independent:
        fg = q[1:].argmax(0).astype(np.uint8) + 1
        positive = np.maximum(pp[1:].max(0), tiny) * np.maximum(q[1:].max(0), tiny) > np.maximum(
            pp[0], tiny
        ) * np.maximum(q[0], tiny)
        label = np.where(positive, fg, 0).astype(np.uint8)
    else:
        label = mods["pool"].pool(pp, q)["spatial_extent"]
    native = np.array(s, order="C", copy=True)
    native.ravel()[idx] = label
    return restore(native, d["geo"]["shape"], d["geo"]["bbox"], d["geo"]["fw"])


def filter_margin(d, proposal, independent):
    s = d["source"]
    native = crop(proposal, d["geo"]["bbox"], d["geo"]["fw"])
    np.testing.assert_array_equal(
        restore(native, d["geo"]["shape"], d["geo"]["bbox"], d["geo"]["fw"]), proposal
    )
    margin = d["p"].max(0).astype(np.float64) - np.take_along_axis(
        d["p"], native[None].astype(np.intp), axis=0
    )[0].astype(np.float64)
    margin = restore(margin, d["geo"]["shape"], d["geo"]["bbox"], d["geo"]["fw"])
    if not independent:
        opened = mods["filter"].apply(s, proposal, 1)
        return mods["filter"].matched_margin(s, proposal, opened, margin)
    padded = np.pad(proposal, 2, mode="edge")
    kernel = generate_binary_structure(3, 1).astype(np.int16)
    keep = np.zeros(proposal.shape, bool)
    for label in range(4):
        core = convolve((padded == label).astype(np.int16), kernel, mode="constant", cval=0) == 7
        support = (convolve(core.astype(np.int16), kernel, mode="constant", cval=0) > 0)[2:-2, 2:-2, 2:-2]
        keep |= support
    opened = np.where(keep, proposal, s)
    result = np.array(s, order="C", copy=True)
    for old in range(4):
        for new in range(4):
            if old == new:
                continue
            idx = np.flatnonzero((s == old) & (proposal == new))
            count = int(((s == old) & (opened == new)).sum())
            order = np.lexsort((idx, margin.ravel()[idx]))
            result.ravel()[idx[order[:count]]] = new
    return result


def guard(filtered, proposal, independent):
    return (mods["guard"].independent_guard if independent else mods["guard"].guard)(filtered, proposal)[0]


def decode_no_come(data, student, bias, independent=False):
    spatial_labels = spatial(data, student, bias, independent)
    filtered = filter_margin(data, spatial_labels, independent)
    return guard(filtered, spatial_labels, independent)
