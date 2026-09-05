from __future__ import annotations

import torch
from torch import nn

from brats_tta.tta.tent import TentAdapter, binary_prediction_entropy, configure_norm_stats


class TinyInstanceNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=1),
            nn.InstanceNorm3d(4, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.01),
            nn.Conv3d(4, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def test_binary_entropy_is_finite_and_differentiable() -> None:
    logits = torch.tensor([0.0, 2.0, -2.0], requires_grad=True)
    loss = binary_prediction_entropy(logits).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_tent_updates_and_resets_only_instance_norm_affine() -> None:
    torch.manual_seed(3)
    model = TinyInstanceNormModel()
    frozen_before = model.layers[0].weight.detach().clone()
    norm_before = model.layers[1].weight.detach().clone()
    adapter = TentAdapter(model, learning_rate=1e-3)

    result = adapter.predict_and_adapt(torch.randn(1, 2, 8, 8, 8))
    assert result.logits.shape == (1, 3, 8, 8, 8)
    torch.testing.assert_close(model.layers[0].weight, frozen_before)
    assert not torch.equal(model.layers[1].weight, norm_before)

    adapter.reset()
    torch.testing.assert_close(model.layers[1].weight, norm_before)


def test_norm_statistics_is_source_equivalent_for_untracked_instance_norm() -> None:
    model = TinyInstanceNormModel().eval()
    image = torch.randn(1, 2, 8, 8, 8)
    expected = model(image)
    information = configure_norm_stats(model)
    actual = model(image)

    assert information["equivalent_to_source"] is True
    torch.testing.assert_close(actual, expected)
