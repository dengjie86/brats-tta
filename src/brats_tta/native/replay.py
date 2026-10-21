"""Native FP16 fusion with FP32 adjoint and one network graph at a time.

The native layout requires nnunetv2; the loss and replay primitives do not.
"""

import itertools

import torch

from .adaptation import loss_terms

AXES = [()] + [c for i in range(1, 4) for c in itertools.combinations([2, 3, 4], i)]


class Layout:
    def __init__(self, data, slicers, revert, gaussian):
        self.data = data
        self.slicers = slicers
        self.revert = revert
        self.gaussian = gaussian
        self.denominator = torch.zeros(data.shape[1:], dtype=torch.float16, device=data.device)
        for sl in slicers:
            self.denominator[sl[1:]] += gaussian
        assert (self.denominator > 0).all()

    def padded(self, value):
        shape = value.shape[:-3] + self.data.shape[1:]
        result = torch.zeros(shape, device=value.device, dtype=value.dtype)
        result[(...,) + self.revert[1:]] = value
        return result

    def weight(self, sl):
        # These constants reproduce the derivative of the unrounded linear
        # operation with the actual native FP16 coverage denominator.
        return self.gaussian.float() / self.denominator[sl[1:]].float() / 8

    def views(self):
        for j, sl in enumerate(self.slicers):
            x = self.data[sl][None]
            for i, axes in enumerate(AXES):
                yield 8 * j + i, sl, axes, torch.flip(x, axes) if axes else x


def native_layout(predictor, x):
    from nnunetv2.inference.predict_from_raw_data import compute_gaussian, pad_nd_image

    data, revert = pad_nd_image(
        torch.from_numpy(x), predictor.configuration_manager.patch_size, "constant", {"value": 0}, True, None
    )
    slices = predictor._internal_get_sliding_window_slicers(data.shape[1:])
    gaussian = compute_gaussian(
        tuple(predictor.configuration_manager.patch_size),
        sigma_scale=1 / 8,
        value_scaling_factor=10,
        device=predictor.device,
    )
    return Layout(data.to(predictor.device), slices, revert, gaussian)


def native_forward(predictor, x, capture=False):
    refs = []

    def hook(module, args, output):
        refs.append(output.detach().cpu().clone())

    handle = predictor.network.register_forward_hook(hook) if capture else None
    try:
        with torch.no_grad():
            z = predictor.predict_sliding_window_return_logits(torch.from_numpy(x))
    finally:
        if handle is not None:
            handle.remove()
    assert z.dtype == torch.float16 and torch.isfinite(z).all()
    return z, refs


def objective(z, reference, delta, edits):
    valid = torch.ones_like(edits, dtype=torch.bool)
    edit, anchor, counts = loss_terms(z[None], reference[None], delta[None], edits[None], valid[None])
    return edit + anchor, dict(edit=float(edit.detach()), anchor=float(anchor.detach()), **counts)


def fused_adjoint(z, reference, delta, edits, scale):
    leaf = z.detach().float().requires_grad_(True)
    loss, info = objective(leaf, reference, delta, edits)
    gradient = torch.autograd.grad(loss * scale, leaf)[0].detach()
    assert torch.isfinite(gradient).all()
    return gradient, info


def replay_fused(network, layout, gradient, amp=True, source_references=None):
    padded = layout.padded(gradient)
    checked = 0
    for i, sl, axes, x in layout.views():
        with torch.autocast("cuda", enabled=amp):
            y = network(x)
        if source_references is not None:
            assert torch.equal(y.detach().cpu(), source_references[i])
            checked += 1
        grad = (padded[sl] * layout.weight(sl))[None]
        if axes:
            grad = torch.flip(grad, axes)
        torch.autograd.backward(y, grad.to(y.dtype))
        del y, grad
    return dict(views=8 * len(layout.slicers), exact_source_reference_checks=checked)


def viewwise_loss(y, reference, delta, edits, background, foreground, weight, denoms):
    y, reference = y.float(), reference.float()
    yc = y - y.mean(1, keepdim=True)
    rc = reference - reference.mean(1, keepdim=True)
    cost = ((yc - rc - delta[None]).square().mean(1))[0]
    edit = (cost * weight * edits).sum() / denoms["edits"]
    gap = y[:, 1:].max(1).values - y[:, 0]
    refgap = reference[:, 1:].max(1).values - reference[:, 0]
    penalty = ((3 / 16) * (gap - refgap).square())[0]
    parts = [
        (penalty * weight * mask).sum() / denoms[name]
        for name, mask in [("background", background), ("foreground", foreground)]
        if denoms[name]
    ]
    anchor = sum(parts) / len(parts)
    return edit, anchor


def replay_viewwise(
    network, layout, references, source_native, delta, edits, scale, amp=True, check_source=False
):
    source_label = source_native.argmax(0)
    pads = dict(
        delta=layout.padded(delta),
        edits=layout.padded(edits),
        background=layout.padded(~edits & (source_label == 0)),
        foreground=layout.padded(~edits & (source_label > 0)),
    )
    denoms = {k: int(v.sum()) for k, v in pads.items() if k != "delta"}
    assert denoms["edits"] > 0 and denoms["background"] + denoms["foreground"] > 0
    edit_sum = anchor_sum = 0.0
    for i, sl, axes, x in layout.views():
        with torch.autocast("cuda", enabled=amp):
            y = network(x)
        if check_source:
            assert torch.equal(y.detach().cpu(), references[i])
        ref = references[i].to(y.device)
        if axes:
            y = torch.flip(y, axes)
            ref = torch.flip(ref, axes)
        e, a = viewwise_loss(
            y,
            ref,
            pads["delta"][sl],
            pads["edits"][sl[1:]],
            pads["background"][sl[1:]],
            pads["foreground"][sl[1:]],
            layout.weight(sl),
            denoms,
        )
        ((e + a) * scale).backward()
        edit_sum += float(e.detach())
        anchor_sum += float(a.detach())
        del y, ref, e, a
    return dict(
        edit=edit_sum,
        anchor=anchor_sum,
        views=8 * len(layout.slicers),
        exact_source_reference_checks=8 * len(layout.slicers) if check_source else 0,
    )
