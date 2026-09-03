from __future__ import annotations

import pytest
import torch
from torch import nn

from brats_tta.losses.segmentation import DeepSupervisionLoss, build_loss
from brats_tta.models.unet3d import PlainUNet3D


def _nested_target(size: int) -> torch.Tensor:
    target = torch.zeros((1, 3, size, size, size), dtype=torch.float32)
    target[:, 2, 1:-1, 1:-1, 1:-1] = 1
    target[:, 1, 2:-2, 2:-2, 2:-2] = 1
    target[:, 0, 3:-3, 3:-3, 3:-3] = 1
    return target


def test_canonical_source_model_specification() -> None:
    model = PlainUNet3D()
    instance_norms = [module for module in model.modules() if isinstance(module, nn.InstanceNorm3d)]

    assert model.features == (32, 64, 128, 256, 320)
    assert model.in_channels == 4
    assert model.out_channels == 3
    assert model.required_divisibility == 16
    assert model.parameter_count() == 16_550_668
    assert len(instance_norms) == 18
    assert all(layer.affine and not layer.track_running_stats for layer in instance_norms)
    assert not any(isinstance(module, nn.Dropout3d) for module in model.modules())


def test_deep_supervision_forward_and_backward() -> None:
    model = PlainUNet3D(features=(4, 8, 16), deep_supervision=True)
    image = torch.randn((1, 4, 16, 16, 16))
    target = _nested_target(16)
    loss_function = build_loss(
        {
            "dice_weight": 1.0,
            "bce_weight": 1.0,
            "deep_supervision_weights": [1.0, 0.5],
        },
        number_of_outputs=2,
    )

    model.train()
    outputs = model(image)
    assert isinstance(outputs, list)
    assert [tuple(output.shape) for output in outputs] == [
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
    ]
    loss = loss_function(outputs, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder[0][0][0].weight.grad is not None

    model.eval()
    with torch.inference_mode():
        output = model(image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == target.shape


def test_input_divisibility_is_checked() -> None:
    model = PlainUNet3D(features=(4, 8, 16))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros((1, 4, 15, 16, 16)))


def test_negative_deep_supervision_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DeepSupervisionLoss(nn.Identity(), [1.0, -0.5])
