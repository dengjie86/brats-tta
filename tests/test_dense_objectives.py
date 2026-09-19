import math

import pytest
import torch

from brats_tta.tta.dense_objectives import (
    dense_entropy_loss,
    dense_prediction_confidence,
    dense_teacher_consistency_loss,
    image_brain_mask,
)


def test_image_brain_mask_uses_any_nonzero_modality() -> None:
    image = torch.zeros(1, 2, 1, 1, 3)
    image[0, 1, 0, 0, 1] = -2
    assert image_brain_mask(image).flatten().tolist() == [False, True, False]


def test_brain_entropy_excludes_confident_padding() -> None:
    image = torch.zeros(1, 1, 1, 1, 4)
    image[..., -1] = 1
    logits = torch.tensor([[[[[-20.0, -20.0, -20.0, 0.0]]]]])
    all_loss = dense_entropy_loss(logits, image, reduction="all", normalize=False)
    brain_loss = dense_entropy_loss(logits, image, reduction="brain", normalize=False)
    assert brain_loss.item() == pytest.approx(math.log(2.0))
    assert brain_loss > all_loss * 3.9


def test_foreground_background_balancing_upweights_sparse_foreground() -> None:
    image = torch.ones(1, 1, 1, 1, 10)
    logits = torch.full((1, 1, 1, 1, 10), -8.0)
    logits[..., -1] = 0.1
    all_loss = dense_entropy_loss(logits, image, reduction="brain", normalize=False)
    balanced = dense_entropy_loss(logits, image, reduction="foreground_background_balanced", normalize=False)
    assert balanced > all_loss * 3


def test_uncertain_confidence_is_not_hidden_by_confident_voxels() -> None:
    image = torch.ones(1, 1, 1, 1, 100)
    logits = torch.full((1, 1, 1, 1, 100), -10.0)
    logits[..., -1] = 0.0
    global_confidence = dense_prediction_confidence(logits, image, reduction="all")
    uncertain = dense_prediction_confidence(
        logits, image, reduction="brain_uncertain", uncertain_fraction=0.01
    )
    assert global_confidence > 0.99
    assert uncertain.item() == pytest.approx(0.5)


def test_identity_teacher_has_zero_gradient_for_all_reductions() -> None:
    image = torch.ones(1, 1, 1, 2, 2)
    for reduction in ("all", "brain", "foreground_background_balanced"):
        logits = torch.randn(1, 2, 1, 2, 2, requires_grad=True)
        teacher = logits.detach().sigmoid()
        loss = dense_teacher_consistency_loss(logits, teacher, image, reduction=reduction)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-8)
