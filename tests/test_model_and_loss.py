from __future__ import annotations

import pytest
import torch
from torch import nn

from brats_tta.losses.segmentation import DeepSupervisionLoss, build_loss
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.models.cicek_unet3d import CicekUNet3D
from brats_tta.models.original_unet3d import OriginalUNet3D
from brats_tta.models.unet3d import PlainUNet3D


def _nested_target(size: int) -> torch.Tensor:
    target = torch.zeros((1, 3, size, size, size), dtype=torch.float32)
    target[:, 2, 1:-1, 1:-1, 1:-1] = 1
    target[:, 1, 2:-2, 2:-2, 2:-2] = 1
    target[:, 0, 3:-3, 3:-3, 3:-3] = 1
    return target


def test_canonical_source_model_specification() -> None:
    model = PlainUNet3D()
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm3d)]

    assert model.features == (32, 64, 128, 256, 320, 320)
    assert model.in_channels == 4
    assert model.out_channels == 4
    assert model.output_mode == "classes_softmax"
    assert model.required_divisibility == 32
    assert model.parameter_count() == 31_199_796
    assert len(batch_norms) == 22
    assert all(layer.affine and layer.track_running_stats for layer in batch_norms)
    assert not any(isinstance(module, nn.Dropout3d) for module in model.modules())


def test_original_unet_topology_has_no_normalization_or_deep_supervision() -> None:
    model = OriginalUNet3D(features=(4, 8, 16, 32, 64))
    image = torch.randn((1, 4, 32, 32, 32))

    output = model(image)

    assert model.required_divisibility == 16
    assert model.deep_supervision is False
    assert model.norm == "none"
    assert model.padding == "same"
    assert model.parameter_count() == 353_552
    assert output.shape == (1, 4, 32, 32, 32)
    assert not any(isinstance(module, nn.BatchNorm3d) for module in model.modules())
    assert not any(isinstance(module, nn.Dropout3d) for module in model.modules())
    assert all(not isinstance(module, nn.LeakyReLU) for module in model.modules())


def test_original_unet_batch_norm_control_exposes_features_for_tta() -> None:
    model = OriginalUNet3D(features=(4, 8, 16), norm="batch3d")
    features = model.encode_features(torch.randn((1, 4, 16, 16, 16)))

    assert len(features) == 3
    assert model.decode_features(features).shape == (1, 4, 16, 16, 16)
    assert sum(isinstance(module, nn.BatchNorm3d) for module in model.modules()) == 10


def test_cicek_unet3d_matches_the_released_channel_and_bn_topology() -> None:
    model = CicekUNet3D()
    tiny_model = CicekUNet3D(features=(4, 8, 16, 32))
    image = torch.randn((1, 4, 16, 16, 16))

    output = tiny_model(image)

    assert model.features == (64, 128, 256, 512)
    assert model.parameter_count() == 19_075_044
    assert model.required_divisibility == 8
    assert model.deep_supervision is False
    assert model.norm == "batch3d"
    assert model.track_running_stats is False
    assert output.shape == (1, 4, 16, 16, 16)
    assert sum(isinstance(module, nn.BatchNorm3d) for module in model.modules()) == 17
    assert sum(isinstance(module, nn.ConvTranspose3d) for module in model.modules()) == 3
    assert not any(isinstance(module, nn.Dropout3d) for module in model.modules())


def test_deep_supervision_forward_and_backward() -> None:
    model = PlainUNet3D(
        out_channels=3,
        features=(4, 8, 16),
        output_mode="regions_sigmoid",
        deep_supervision=True,
        norm="instance3d",
        track_running_stats=False,
    )
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
    model = PlainUNet3D(
        out_channels=3,
        features=(4, 8, 16),
        output_mode="regions_sigmoid",
        norm="instance3d",
        track_running_stats=False,
    )
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros((1, 4, 15, 16, 16)))


def test_softmax_source_predictions_compare_against_target_regions() -> None:
    logits = torch.full((1, 4, 1, 1, 4), -10.0)
    for voxel, class_index in enumerate((0, 1, 2, 3)):
        logits[0, class_index, 0, 0, voxel] = 10.0
    target_regions = torch.tensor(
        [[[[[0, 0, 0, 1]]], [[[0, 1, 0, 1]]], [[[0, 1, 1, 1]]]]],
        dtype=torch.float32,
    )
    metrics = compute_region_metrics(
        logits,
        target_regions,
        output_mode="classes_softmax",
        label_schema="brats_modern",
    )
    assert metrics["dice_mean"] == pytest.approx(1.0)


def test_negative_deep_supervision_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DeepSupervisionLoss(nn.Identity(), [1.0, -0.5])


def test_zero_weight_deep_supervision_head_receives_zero_gradient() -> None:
    model = PlainUNet3D(
        out_channels=3,
        features=(4, 8, 16),
        output_mode="regions_sigmoid",
        deep_supervision=True,
        norm="instance3d",
        track_running_stats=False,
    )
    loss_function = build_loss(
        {
            "dice_weight": 1.0,
            "bce_weight": 1.0,
            "deep_supervision_weights": [1.0, 0.0],
        },
        number_of_outputs=2,
    )

    outputs = model(torch.randn((1, 4, 16, 16, 16)))
    loss_function(outputs, _nested_target(16)).backward()

    disabled_head_gradient = model.segmentation_heads[0].weight.grad
    assert disabled_head_gradient is not None
    assert torch.count_nonzero(disabled_head_gradient) == 0
