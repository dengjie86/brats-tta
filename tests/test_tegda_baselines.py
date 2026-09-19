from copy import deepcopy

import pytest
import torch
from torch import nn

from brats_tta.tta.baseline_state import (
    load_baseline_state,
    save_baseline_state,
    source_parameters,
)
from brats_tta.tta.tegda_baselines import (
    CoTTAAdapter,
    SARAdapter,
    _invert_spatial_augmentation,
    _random_source_compatible_augmentation,
    build_tegda_baseline_adapter,
)


def tiny() -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(2, 4, 1),
        nn.InstanceNorm3d(4, affine=True),
        nn.LeakyReLU(),
        nn.Conv3d(4, 3, 1),
    )


def test_sar_updates_norm_affine_without_recovery() -> None:
    torch.manual_seed(3)
    adapter = SARAdapter(tiny(), entropy_margin=1.0, recovery_threshold=None, learning_rate=1e-2)
    before = [parameter.detach().clone() for parameter in adapter.parameters]
    result = adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    assert result.logits.shape == (1, 3, 4, 4, 4)
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters))
    assert adapter.case_diagnostics()["sar_updates"] == 1


def test_sar_recovery_restores_source() -> None:
    torch.manual_seed(4)
    adapter = SARAdapter(tiny(), entropy_margin=1.0, recovery_threshold=10.0)
    adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    for parameter, source in zip(adapter.parameters, adapter._source_parameters):
        torch.testing.assert_close(parameter, source, rtol=0, atol=0)
    assert adapter.case_diagnostics()["sar_recovery_resets"] == 1


def test_dense_sar_balances_brain_foreground_without_recovery() -> None:
    torch.manual_seed(14)
    profiled = build_tegda_baseline_adapter("sar", tiny(), profile="dense_v2")
    assert profiled.recovery_threshold is None
    assert profiled.objective_reduction == "foreground_background_balanced"
    adapter = SARAdapter(
        tiny(),
        entropy_margin=1.0,
        recovery_threshold=None,
        objective_reduction="foreground_background_balanced",
    )
    result = adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    assert torch.isfinite(torch.tensor(result.entropy))
    diagnostics = adapter.case_diagnostics()
    assert diagnostics["sar_updates"] == 1
    assert diagnostics["sar_recovery_resets"] == 0


def test_cotta_state_round_trip_and_source_probe(tmp_path) -> None:
    torch.manual_seed(5)
    source = tiny()
    adapter = CoTTAAdapter(
        deepcopy(source),
        learning_rate=1e-3,
        weight_decay=0.1,
        augmentation_threshold=0.0,
        augmentation_count=2,
    )
    image = torch.randn(1, 2, 4, 4, 4)
    result = adapter.predict_and_adapt(image)
    assert result.logits.shape == (1, 3, 4, 4, 4)
    current = [parameter.detach().clone() for parameter in adapter.parameters]
    with source_parameters(adapter) as source_model:
        with torch.no_grad():
            torch.testing.assert_close(source_model(image), source(image), rtol=0, atol=0)
    for expected, actual in zip(current, adapter.parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    state_path = tmp_path / "state.pt"
    settings = {"method": "cotta"}
    save_baseline_state(state_path, adapter, settings, [{"id": "a"}])
    resumed = CoTTAAdapter(
        deepcopy(source),
        learning_rate=1e-3,
        weight_decay=0.1,
        augmentation_threshold=0.0,
        augmentation_count=2,
    )
    assert load_baseline_state(state_path, resumed, settings) == [{"id": "a"}]
    for expected, actual in zip(adapter.parameters, resumed.parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(adapter.ema_model.parameters(), resumed.ema_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert resumed.optimizer.state[resumed.parameters[0]]["step"].item() == pytest.approx(1)


def test_dense_cotta_uses_augmented_target_and_reset() -> None:
    torch.manual_seed(15)
    adapter = CoTTAAdapter(
        tiny(),
        learning_rate=1e-3,
        weight_decay=0.0,
        restore_probability=0.0,
        augmentation_threshold=1.0,
        augmentation_count=2,
        confidence_reduction="brain_uncertain",
        consistency_reduction="foreground_background_balanced",
        augmentation_mode="source_training",
    )
    before = [parameter.detach().clone() for parameter in adapter.parameters]
    result = adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    assert result.logits.shape == (1, 3, 4, 4, 4)
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters))
    assert adapter.case_diagnostics()["cotta_augmentation_forwards"] == 2
    adapter.reset()
    for parameter, source in zip(adapter.parameters, adapter._source_parameters):
        torch.testing.assert_close(parameter, source, rtol=0, atol=0)
    for parameter, source in zip(adapter.ema_model.parameters(), adapter._source_parameters):
        torch.testing.assert_close(parameter, source, rtol=0, atol=0)


def test_dense_cotta_skips_numerical_zero_signal() -> None:
    torch.manual_seed(16)
    adapter = CoTTAAdapter(
        tiny(),
        learning_rate=1e-3,
        weight_decay=0.0,
        restore_probability=0.01,
        augmentation_threshold=0.0,
        augmentation_count=2,
        confidence_reduction="brain_uncertain",
        consistency_reduction="foreground_background_balanced",
        minimum_consistency_mae=1e-7,
    )
    before = [parameter.detach().clone() for parameter in adapter.parameters]
    adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))
    for expected, actual in zip(before, adapter.parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    diagnostics = adapter.case_diagnostics()
    assert diagnostics["cotta_updates"] == 0
    assert diagnostics["cotta_skipped_updates"] == 1
    assert diagnostics["cotta_consistency_probability_mae"] == pytest.approx(0.0)


def test_cotta_augmentation_preserves_zero_background() -> None:
    torch.manual_seed(17)
    image = torch.zeros(1, 2, 4, 4, 4)
    image[:, :, 1:3, 1:3, 1:3] = 1.0
    augmented, transform = _random_source_compatible_augmentation(image)
    restored = _invert_spatial_augmentation(augmented, transform)
    background = image.abs().sum(dim=1, keepdim=True) == 0
    assert torch.count_nonzero(restored.masked_select(background)) == 0
