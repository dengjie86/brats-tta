"""IN affine selection, centered targets and source extent anchors.

Default loss is the native objective's extent anchor; hierarchical is retained
as an explicit diagnostic control. This module performs no I/O or training.
"""

import numpy as np

PROB_FLOOR = 1e-6
ARMS = ("self", "hierarchical", "extent")
STEPS = 8
LR = 1e-3
INITIAL_SCALE = 1024.0


def delta_logits(q, p):
    d = np.log(np.maximum(q, PROB_FLOOR)) - np.log(np.maximum(p, PROB_FLOOR))
    return (d - d.mean(0, keepdims=True)).astype(np.float32)


def project_transition_delta(delta, source, target):
    """Minimum-norm correction preserving only each proposed class contrast.

    For a source class a and proposed class b, retain delta[b]-delta[a]
    along (e_b-e_a), and discard the orthogonal components. Unedited voxels
    receive zero correction. This is a logit target, not an output mask lock.
    """
    delta = np.asarray(delta, dtype=np.float32)
    source, target = np.asarray(source), np.asarray(target)
    if delta.shape != (4, *source.shape) or target.shape != source.shape:
        raise ValueError("Expected four-channel delta and aligned source/target labels")
    if not np.isfinite(delta).all() or not np.isin(source, range(4)).all() or not np.isin(target, range(4)).all():
        raise ValueError("Expected finite delta and labels in 0..3")
    result = np.zeros_like(delta)
    flat = result.reshape(4, -1)
    indices = np.flatnonzero(source.ravel() != target.ravel())
    a = source.ravel()[indices].astype(np.intp)
    b = target.ravel()[indices].astype(np.intp)
    values = delta.reshape(4, -1)
    half = (values[b, indices] - values[a, indices]) * 0.5
    flat[a, indices] = -half
    flat[b, indices] = half
    return result


def select_affine(network):
    import torch

    modules = [m for m in network.modules() if isinstance(m, torch.nn.InstanceNorm3d)]
    assert len(modules) == 22 and all(m.affine and not m.track_running_stats for m in modules)
    network.requires_grad_(False)
    for m in modules:
        m.weight.requires_grad_(True)
        m.bias.requires_grad_(True)
    selected = {n: p for n, p in network.named_parameters() if p.requires_grad}
    ids = {id(p) for p in selected.values()}
    aliases = {n for n, p in network.named_parameters(remove_duplicate=False) if id(p) in ids}
    assert len(selected) == 44 and sum(p.numel() for p in selected.values()) == 7680
    return selected, aliases


def centered(logits):
    z = logits.float()
    return z - z.mean(1, keepdim=True)


def extent_gap(z):
    z = z.float()
    return z[:, 1:].max(dim=1).values - z[:, 0]


def anchor_parts(z, ref):
    z, ref = z.float(), ref.detach().float()
    extent = (3 / 16) * (extent_gap(z) - extent_gap(ref)).square()
    zfg = z[:, 1:] - z[:, 1:].mean(1, keepdim=True)
    rfg = ref[:, 1:] - ref[:, 1:].mean(1, keepdim=True)
    conditional = (1 / 4) * (zfg - rfg).square().sum(1)
    return extent, conditional


def loss_terms(logits, reference, delta, edits, valid, arm="extent"):
    import torch

    assert arm in ARMS
    z, ref = centered(logits), centered(reference.detach())
    anchors = valid & ~edits
    assert edits.any() and anchors.any() and not (edits & ~valid).any()
    edit_loss = ((z - ref - delta.detach()).square().mean(1))[edits].mean()
    extent, conditional = anchor_parts(logits, reference)
    foreground = reference.detach().argmax(1) > 0
    bg, fg = anchors & ~foreground, anchors & foreground
    groups = [m for m in [bg, fg] if m.any()]
    extent_loss = torch.stack([extent[m].mean() for m in groups]).mean()
    conditional_loss = torch.stack([conditional[m].mean() for m in groups]).mean()
    anchor_loss = extent_loss + conditional_loss if arm == "hierarchical" else extent_loss
    return (
        edit_loss,
        anchor_loss,
        dict(
            anchor_BG=int(bg.sum()),
            anchor_FG=int(fg.sum()),
            extent_anchor=float(extent_loss.detach()),
            conditional_anchor_diagnostic=float(conditional_loss.detach()),
        ),
    )
