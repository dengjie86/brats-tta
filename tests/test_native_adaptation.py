import numpy as np
import torch

from brats_tta.native.adaptation import delta_logits, loss_terms
from brats_tta.native.replay import Layout, fused_adjoint, replay_fused


def test_extent_loss_is_invariant_to_class_common_logit_shift():
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn((1, 4, 3, 3, 3), generator=generator, requires_grad=True)
    reference = torch.randn(logits.shape, generator=generator)
    delta = torch.randn(logits.shape, generator=generator)
    edits = torch.zeros((1, 3, 3, 3), dtype=torch.bool)
    edits[:, 1] = True
    valid = torch.ones_like(edits)
    first = loss_terms(logits, reference, delta, edits, valid)
    shifted = loss_terms(logits + 3, reference - 2, delta, edits, valid)
    torch.testing.assert_close(first[0] + first[1], shifted[0] + shifted[1])
    assert first[2]["anchor_BG"] + first[2]["anchor_FG"] == int((~edits).sum())
    (first[0] + first[1]).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_probability_delta_has_zero_class_mean_and_zero_identity():
    p = np.array([0.01, 0.09, 0.2, 0.7], np.float32)[:, None, None, None]
    q = p[::-1].copy()
    np.testing.assert_array_equal(delta_logits(p, p), 0)
    np.testing.assert_allclose(delta_logits(q, p).mean(0), 0, atol=1e-7)
    assert np.isfinite(delta_logits(q, np.zeros_like(p))).all()


def test_replay_gradient_matches_full_eight_view_graph_on_cpu():
    torch.manual_seed(23)
    network = torch.nn.Conv3d(4, 4, 1)
    image = torch.randn(4, 3, 3, 3)
    slicer = (slice(None), slice(0, 3), slice(0, 3), slice(0, 3))
    layout = Layout(image, [slicer], (slice(None),) * 4, torch.ones(3, 3, 3, dtype=torch.float16))
    outputs = []
    for _, _, axes, x in layout.views():
        y = network(x)
        outputs.append(torch.flip(y, axes) if axes else y)
    merged = torch.stack(outputs).mean(0)[0]
    reference = torch.randn_like(merged)
    delta = torch.randn_like(merged)
    edits = torch.zeros(3, 3, 3, dtype=torch.bool)
    edits[1] = True
    valid = torch.ones_like(edits)
    edit, anchor, _ = loss_terms(merged[None], reference[None], delta[None], edits[None], valid[None])
    (edit + anchor).backward()
    expected = [p.grad.clone() for p in network.parameters()]
    network.zero_grad(set_to_none=True)
    gradient, _ = fused_adjoint(merged.detach(), reference, delta, edits, scale=1)
    receipt = replay_fused(network, layout, gradient, amp=False)
    for parameter, target in zip(network.parameters(), expected):
        torch.testing.assert_close(parameter.grad, target, atol=2e-6, rtol=2e-6)
    assert receipt["views"] == 8
