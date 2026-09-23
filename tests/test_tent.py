from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from brats_tta.tta.tent import TentAdapter, categorical_prediction_entropy


class TinyBatchNormSoftmaxModel(nn.Module):
    output_mode = "classes_softmax"

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=1),
            nn.BatchNorm3d(4, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.01),
            nn.Conv3d(4, 4, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class LegacyInstanceNormModel(nn.Module):
    output_mode = "regions_sigmoid"

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm3d(3, affine=True, track_running_stats=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.norm(image)


def test_categorical_entropy_is_finite_and_differentiable() -> None:
    logits = torch.zeros(1, 4, 1, 1, 1, requires_grad=True)
    loss = categorical_prediction_entropy(logits).mean()
    loss.backward()

    assert loss.item() == pytest.approx(math.log(4.0))
    assert logits.grad is not None


def test_tent_updates_and_resets_only_batch_norm_affine() -> None:
    torch.manual_seed(5)
    model = TinyBatchNormSoftmaxModel()
    frozen_before = model.layers[0].weight.detach().clone()
    norm_before = model.layers[1].weight.detach().clone()
    adapter = TentAdapter(model, learning_rate=1e-2, use_amp=False)

    assert model.layers[1].running_mean is None
    assert model.layers[1].running_var is None
    result = adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    assert result.logits.shape == (1, 4, 4, 4, 4)
    torch.testing.assert_close(model.layers[0].weight, frozen_before)
    assert not torch.equal(model.layers[1].weight, norm_before)

    adapter.reset()
    torch.testing.assert_close(model.layers[1].weight, norm_before)


def test_tent_scope_and_statistics_only_ablation() -> None:
    class ScopedModel(nn.Module):
        output_mode = "classes_softmax"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.BatchNorm3d(2))
            self.decoder = nn.Sequential(nn.BatchNorm3d(2))
            self.head = nn.Conv3d(2, 4, kernel_size=1)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.head(self.decoder(self.encoder(image)))

    model = ScopedModel()
    adapter = TentAdapter(model, bn_scope="encoder", use_amp=False)
    assert adapter.parameter_names == ["encoder.0.weight", "encoder.0.bias"]
    assert model.encoder[0].running_mean is None
    assert model.decoder[0].running_mean is None

    stats_model = ScopedModel()
    stats_adapter = TentAdapter(
        stats_model,
        update_affine=False,
        use_amp=False,
    )
    assert stats_adapter.parameter_names == []
    before = stats_model.head.weight.detach().clone()
    result = stats_adapter.predict_and_adapt(torch.randn(1, 2, 2, 2, 2))
    assert result.logits.shape == (1, 4, 2, 2, 2)
    torch.testing.assert_close(stats_model.head.weight, before)


def test_tent_rejects_the_removed_instance_norm_region_model() -> None:
    with pytest.raises(ValueError, match="BatchNorm3d"):
        TentAdapter(LegacyInstanceNormModel(), use_amp=False)


def test_multiple_steps_return_last_forward_and_reset_replays_episode() -> None:
    torch.manual_seed(9)
    model = TinyBatchNormSoftmaxModel()
    manual = TentAdapter(deepcopy(model), learning_rate=1e-2, use_amp=False)
    adapter = TentAdapter(model, learning_rate=1e-2, steps=3, use_amp=False)
    image = torch.randn(1, 2, 4, 4, 4)
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
    model = TinyBatchNormSoftmaxModel().eval()
    image = torch.randn(1, 2, 4, 4, 4)
    adapter = TentAdapter(model, learning_rate=1e-2, use_amp=False)
    source_model = deepcopy(model).eval()
    with torch.no_grad():
        source = source_model(image)
    online, _ = sliding_window_tent_logits(
        model,
        adapter,
        image,
        patch_size=(4, 4, 4),
        gaussian_weighting=False,
    )
    with torch.no_grad():
        after_update = model(image)
    torch.testing.assert_close(online, source)
    assert not torch.allclose(online, after_update)


def test_five_updates_match_categorical_entropy_reference() -> None:
    torch.manual_seed(23)
    model = TinyBatchNormSoftmaxModel().eval()
    reference = deepcopy(model).eval()
    reference.requires_grad_(False)
    reference_norm = reference.layers[1]
    reference_norm.track_running_stats = False
    reference_norm.running_mean = None
    reference_norm.running_var = None
    reference_norm.requires_grad_(True)
    optimizer = torch.optim.Adam(reference_norm.parameters(), lr=1e-3)
    adapter = TentAdapter(model, learning_rate=1e-3, use_amp=False)
    image = torch.randn(1, 2, 4, 4, 4)
    frozen = model.layers[0].weight.detach().clone()

    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits = reference(image)
        categorical_prediction_entropy(logits).mean().backward()
        optimizer.step()
        actual = adapter.predict_and_adapt(image)
        torch.testing.assert_close(actual.logits, logits.detach())
        for current, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(current, expected)
        torch.testing.assert_close(model.layers[0].weight, frozen)


def test_reported_adaptation_updates_match_optimizer_steps() -> None:
    from brats_tta.tta.inference import sliding_window_tent_logits

    model = TinyBatchNormSoftmaxModel()
    adapter = TentAdapter(model, steps=3, use_amp=False)
    progress_events: list[tuple[int, int]] = []
    _, information = sliding_window_tent_logits(
        model,
        adapter,
        torch.randn(1, 2, 4, 4, 8),
        patch_size=(4, 4, 4),
        overlap=0,
        progress_callback=lambda done, total: progress_events.append((done, total)),
    )
    actual_steps = int(next(iter(adapter.optimizer.state.values()))["step"].item())
    assert information["adaptation_patch_batches"] == 2
    assert information["adaptation_updates"] == actual_steps == 6
    assert progress_events == [(1, 2), (2, 2)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA float16")
def test_fp16_categorical_entropy_retains_gradients_and_resets_scaler() -> None:
    class ConfidentVolume(nn.Module):
        output_mode = "classes_softmax"

        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.BatchNorm3d(4, affine=True, track_running_stats=True)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            offsets = torch.tensor([4.0, -4.0, -4.0, -4.0], device=images.device)
            return (self.norm(images) + offsets.view(1, 4, 1, 1, 1)).to(torch.float16)

    model = ConfidentVolume().cuda()
    adapter = TentAdapter(model, use_amp=True)
    initial_scale = adapter.scaler.get_scale()
    images = torch.zeros(1, 4, 4, 4, 4, device="cuda")
    adapter.predict_and_adapt(images)

    assert model.norm.bias.grad is not None
    assert torch.count_nonzero(model.norm.bias.grad) > 0
    assert torch.count_nonzero(model.norm.bias) > 0
    adapter.reset()
    assert torch.count_nonzero(model.norm.bias) == 0
    assert adapter.scaler.get_scale() == initial_scale
    assert adapter.scaler.state_dict()["_growth_tracker"] == 0
