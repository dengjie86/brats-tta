from __future__ import annotations

from copy import deepcopy

import pytest
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


def test_multiple_steps_return_last_forward_and_reset_replays_episode() -> None:
    torch.manual_seed(9)
    model = TinyInstanceNormModel()
    manual = TentAdapter(deepcopy(model), learning_rate=1e-2, use_amp=False)
    adapter = TentAdapter(model, learning_rate=1e-2, steps=3, use_amp=False)
    image = torch.randn(1, 2, 8, 8, 8)
    first = manual.predict_and_adapt(image).logits
    manual.predict_and_adapt(image)
    expected = manual.predict_and_adapt(image).logits
    actual = adapter.predict_and_adapt(image).logits
    assert not torch.equal(first, expected)
    torch.testing.assert_close(actual, expected)
    adapter.reset()
    torch.testing.assert_close(adapter.predict_and_adapt(image).logits, expected)


def test_one_patch_online_output_is_before_update_not_final_model_prediction() -> None:
    from brats_tta.tta.inference import sliding_window_tent_logits

    torch.manual_seed(17)
    model = TinyInstanceNormModel().eval()
    image = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        source = model(image).clone()
    adapter = TentAdapter(model, learning_rate=1e-2, use_amp=False)
    online, _ = sliding_window_tent_logits(model, adapter, image,
        patch_size=(8, 8, 8), gaussian_weighting=False)
    with torch.no_grad():
        after_update = model(image)
    torch.testing.assert_close(online, source)
    assert not torch.allclose(online, after_update)


def test_five_updates_match_independent_bernoulli_entropy_reference() -> None:
    torch.manual_seed(23)
    model = TinyInstanceNormModel().eval()
    reference = deepcopy(model)
    reference.requires_grad_(False)
    reference.layers[1].requires_grad_(True)
    optimizer = torch.optim.Adam(reference.layers[1].parameters(), lr=1e-3)
    adapter = TentAdapter(model, learning_rate=1e-3, use_amp=False)
    image = torch.randn(1, 2, 8, 8, 8)
    frozen = model.layers[0].weight.detach().clone()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits = reference(image)
        p = logits.sigmoid()
        loss = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        loss.backward()
        optimizer.step()
        actual = adapter.predict_and_adapt(image)
        torch.testing.assert_close(actual.logits, logits.detach())
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(model.layers[0].weight, frozen)


def test_reported_adaptation_updates_match_optimizer_steps() -> None:
    from brats_tta.tta.inference import sliding_window_tent_logits

    model = TinyInstanceNormModel()
    adapter = TentAdapter(model, steps=3, use_amp=False)
    progress_events = []
    _, information = sliding_window_tent_logits(
        model,
        adapter,
        torch.randn(1, 2, 8, 8, 16),
        patch_size=(8, 8, 8),
        overlap=0,
        progress_callback=lambda done, total: progress_events.append((done, total)),
    )
    actual_steps = int(next(iter(adapter.optimizer.state.values()))["step"].item())
    assert information["adaptation_patch_batches"] == 2
    assert information["adaptation_updates"] == actual_steps == 6
    assert progress_events == [(1, 2), (2, 2)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA float16")
def test_fp16_volume_entropy_retains_gradients_and_resets_scaler() -> None:
    class ConfidentVolume(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.InstanceNorm3d(1, affine=True, track_running_stats=False)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Match the real output size and confidence regime. The cast models
            # the final autocast convolution's output in the source U-Net.
            logits = self.norm.bias.view(1, 1, 1, 1, 1) + 4.0
            return logits.to(torch.float16).expand(1, 3, 128, 128, 128)

    model = ConfidentVolume().cuda()
    adapter = TentAdapter(model, use_amp=True)
    initial_scale = adapter.scaler.get_scale()
    images = torch.zeros(1, 1, 1, 1, 1, device="cuda")
    adapter.predict_and_adapt(images)
    reference = torch.tensor(4.0, device="cuda", requires_grad=True)
    binary_prediction_entropy(reference).backward()
    torch.testing.assert_close(model.norm.bias.grad[0], reference.grad, rtol=1e-3, atol=1e-5)
    assert model.norm.bias.item() > 0
    adapter.reset()
    assert model.norm.bias.item() == 0
    assert adapter.scaler.get_scale() == initial_scale
    assert adapter.scaler.state_dict()["_growth_tracker"] == 0
