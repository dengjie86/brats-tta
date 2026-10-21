"""Explicit one/eight-view learning primitives; the historical path is unchanged.

The soft-extent coordinates preserve the frozen extent-factor equations.
One-view learning uses its own Source reference, so Source self-loss stays zero.
This module performs no dataset, checkpoint or experiment-directory I/O.
"""
import torch

from .replay import AXES, native_forward, objective as native_objective


class LearningLayout:
    def __init__(self, base, views=8):
        if views not in (1, 8):
            raise ValueError('Use one or eight learning views')
        self.base = base
        self.axes = AXES if views == 8 else [()]
        self.slicers = base.slicers

    def padded(self, value):
        return self.base.padded(value)

    def weight(self, sl):
        weight = self.base.weight(sl)
        return weight if len(self.axes) == 8 else weight * 8

    def views(self):
        for j, sl in enumerate(self.slicers):
            x = self.base.data[sl][None]
            for i, axes in enumerate(self.axes):
                yield len(self.axes) * j + i, sl, axes, torch.flip(x, axes) if axes else x


def forward_views(predictor, x, views=8):
    if views not in (1, 8):
        raise ValueError('Use one or eight inference views')
    original = predictor.use_mirroring
    try:
        predictor.use_mirroring = views == 8
        return native_forward(predictor, x)[0]
    finally:
        predictor.use_mirroring = original


def factor_scores(logits, reference, gain=0.5):
    if logits.shape != reference.shape or logits.shape[0] != 4:
        raise ValueError('Expected aligned four-class fused logits')
    z, source = logits.float(), reference.detach().float()
    source_max = source[1:].max(0).values
    current_max = z[1:].max(0).values
    gap_change = (current_max - z[0]) - (source_max - source[0])
    maximum = source_max + gain * gap_change
    foreground = maximum[None] + (z[1:] - current_max[None])
    return torch.cat((source[0:1], foreground), 0)


def soft_extent_adjoint(z, reference, delta, edits, scale):
    leaf = z.detach().float().requires_grad_(True)
    source = factor_scores(reference, reference)
    wanted = factor_scores(reference.float() + delta.detach().float(), reference)
    target = wanted - source
    target = target - target.mean(0, keepdim=True)
    output = factor_scores(leaf, reference)
    loss, info = native_objective(output, source, target, edits)
    gradient = torch.autograd.grad(loss * scale, leaf)[0].detach()
    assert torch.isfinite(gradient).all()
    return gradient, dict(total=float(loss.detach()), **info)


def replay_learning(network, layout, gradient, amp=True):
    padded = layout.padded(gradient)
    calls = 0
    for _, sl, axes, x in layout.views():
        with torch.autocast(x.device.type, enabled=amp):
            y = network(x)
        grad = (padded[sl] * layout.weight(sl))[None]
        if axes:
            grad = torch.flip(grad, axes)
        torch.autograd.backward(y, grad.to(y.dtype))
        calls += 1
    return calls
