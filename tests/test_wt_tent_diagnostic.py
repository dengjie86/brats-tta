from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from brats_tta.cli.diagnose_wt_tent import (
    accumulate_regional_volume_gradients,
    apply_first_adam_step,
    prediction_report,
    regional_gradient_report,
)
from brats_tta.tta.tent import TentAdapter


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(2, 4, 1),
            nn.InstanceNorm3d(4, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.01),
            nn.Conv3d(4, 3, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def test_joint_all_voxel_gradient_is_equal_channel_mean_and_one_step_resets() -> None:
    torch.manual_seed(31)
    model = TinyModel()
    source = deepcopy(model.state_dict())
    adapter = TentAdapter(model, learning_rate=1e-3, use_amp=False)
    image = torch.randn(1, 2, 4, 4, 8)
    gradients, losses, batches = accumulate_regional_volume_gradients(
        adapter,
        image,
        reduction="all",
        patch_size=(4, 4, 4),
        overlap=0,
    )
    assert batches == 2
    assert set(losses) == {"ET", "TC", "WT", "joint"}
    for joint, expected in zip(
        gradients["joint"],
        [sum(values) / 3 for values in zip(gradients["ET"], gradients["TC"], gradients["WT"])],
    ):
        torch.testing.assert_close(joint, expected, rtol=2e-5, atol=1e-7)
    report = regional_gradient_report(gradients)
    assert report["joint_vs_equal_channel_mean_relative_l2"] < 2e-5
    assert apply_first_adam_step(adapter, gradients["joint"]) > 0
    assert any(not torch.equal(value, source[name]) for name, value in model.state_dict().items())
    adapter.reset()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, source[name])


def test_prediction_report_exposes_region_volume_direction() -> None:
    logits = torch.full((1, 3, 2, 2, 2), -10.0)
    target = torch.zeros_like(logits)
    logits[:, 2, 0] = 10.0
    target[:, 2, 0, 0, 0] = 1.0
    report = prediction_report(logits, target)
    assert report["regions"]["WT"]["predicted_voxels"] == 4
    assert report["regions"]["WT"]["target_voxels"] == 1
    assert report["regions"]["WT"]["predicted_to_target_volume_ratio"] == 4.0
    assert abs(report["metrics"]["dice_WT"] - 0.4) < 1e-7
    assert report["probability_by_truth"]["WT"]["target_positive"]["count"] == 1
    assert "0.05" in report["threshold_sweep"]


def test_first_adam_step_accepts_a_learning_rate_override() -> None:
    torch.manual_seed(32)
    model = TinyModel()
    adapter = TentAdapter(model, learning_rate=1e-4, use_amp=False)
    image = torch.randn(1, 2, 4, 4, 4)
    gradients, _, _ = accumulate_regional_volume_gradients(
        adapter,
        image,
        reduction="all",
        patch_size=(4, 4, 4),
        overlap=0,
    )
    small_delta = apply_first_adam_step(adapter, gradients["WT"], learning_rate=1e-4)
    large_delta = apply_first_adam_step(adapter, gradients["WT"], learning_rate=1e-3)
    assert large_delta > 5 * small_delta
    assert adapter.optimizer.param_groups[0]["lr"] == 1e-3
