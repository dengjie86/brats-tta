from copy import deepcopy
import pytest
import torch

from brats_tta.native.efficient import LearningLayout, soft_extent_adjoint, replay_learning
from brats_tta.native.replay import Layout


@pytest.mark.parametrize('views', [1, 8])
def test_replay_matches_joint_graph_with_overlap_and_instance_norm(views):
    torch.manual_seed(16)
    model = torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, padding=1),
        torch.nn.InstanceNorm3d(4, affine=True), torch.nn.LeakyReLU(), torch.nn.Conv3d(4, 4, 1))
    model.requires_grad_(False)
    model[1].requires_grad_(True)
    image = torch.randn(4, 6, 4, 4)
    slices = [(slice(None), slice(k,k+4), slice(0,4), slice(0,4)) for k in (0,2)]
    base = Layout(image, slices, (slice(None),)*4, torch.ones(4,4,4,dtype=torch.float16))
    layout = LearningLayout(base, views)
    merged = torch.zeros_like(image)
    for _, sl, axes, patch in layout.views():
        out = model(patch)[0]
        if axes:
            out = torch.flip(out, tuple(a-1 for a in axes))
        merged[sl] = merged[sl] + out * layout.weight(sl)
    gradient = torch.randn_like(merged)
    (merged * gradient).sum().backward()
    expected = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    assert replay_learning(model, layout, gradient, amp=False) == 2*views
    for actual, wanted in zip([p.grad for p in model.parameters() if p.requires_grad], expected):
        torch.testing.assert_close(actual, wanted, atol=3e-5, rtol=3e-5)


def test_soft_extent_source_has_zero_gradient_and_teacher_edit_has_nonzero_gradient():
    torch.manual_seed(17)
    reference = torch.randn(4,3,4,5)
    edits = torch.zeros(3,4,5,dtype=torch.bool)
    edits[1] = True
    grad, info = soft_extent_adjoint(reference, reference, torch.zeros_like(reference), edits, 1.)
    assert torch.count_nonzero(grad) == 0 and info['total'] == 0
    delta = torch.zeros_like(reference)
    delta[1,edits] = 1
    delta[2,edits] = -1
    grad, info = soft_extent_adjoint(reference, reference, delta, edits, 1.)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0 and info['total'] > 0
