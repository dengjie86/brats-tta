from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from brats_tta.cli.evaluate_advanced_tta import build_parser
from brats_tta.models.unet3d import PlainUNet3D
from brats_tta.tta.tegda import FeatureBank, TEGDAAdapter, compute_adic_quality


def _tiny_model() -> PlainUNet3D:
    return PlainUNet3D(
        features=(4, 8),
        deep_supervision=False,
        norm="batch3d",
        track_running_stats=True,
    )


def test_tegda_formal_defaults_follow_paper_braTS_settings() -> None:
    adapter = TEGDAAdapter(_tiny_model())
    metadata = adapter.method_metadata()

    assert metadata["learning_rate"] == pytest.approx(1e-4)
    assert metadata["adaptation_scope"] == "full"
    assert metadata["dropout_samples"] == 10
    assert metadata["dropout_probability"] == pytest.approx(0.5)
    assert metadata["feature_bank_size"] == 10
    assert metadata["feature_percentile"] == pytest.approx(0.90)

    arguments = build_parser().parse_args(
        [
            "--checkpoint",
            "source.pt",
            "--manifest",
            "target.json",
            "--output-dir",
            "result",
            "--method",
            "tegda",
        ]
    )
    assert arguments.tegda_lr == pytest.approx(1e-4)
    assert arguments.tegda_scope == "full"
    assert arguments.tegda_dropout_samples == 10
    assert arguments.tegda_dropout_probability == pytest.approx(0.5)
    assert arguments.tegda_feature_bank_size == 10
    assert arguments.tegda_feature_percentile == pytest.approx(0.90)


def test_feature_bank_uses_similarity_weighted_reference() -> None:
    bank = FeatureBank(capacity=3)
    bank.push(torch.tensor([1.0, 0.0]))
    bank.push(torch.tensor([0.0, 1.0]))

    reference, similarity = bank.similarity_weighted_reference(torch.tensor([1.0, 0.0]))
    expected_weights = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    expected = torch.tensor([expected_weights[0], expected_weights[1]])

    torch.testing.assert_close(reference, expected)
    assert similarity == pytest.approx(float(expected_weights[0]))


def test_adic_quality_is_bounded_and_perfect_for_identical_predictions() -> None:
    logits = torch.full((1, 4, 2, 2, 2), -5.0)
    logits[:, 3] = 5.0
    probabilities = logits.softmax(dim=1).unsqueeze(0).repeat(3, 1, 1, 1, 1, 1)
    quality = compute_adic_quality(logits, probabilities)

    assert quality["adic_mean"] == pytest.approx(1.0)
    assert quality["adic_ET"] == pytest.approx(1.0)
    assert quality["adic_mismatch_fraction"] == 0.0


def test_tegda_norm_scope_updates_bn_and_round_trips_state() -> None:
    torch.manual_seed(41)
    source = _tiny_model()
    model = deepcopy(source)
    frozen_before = model.encoder[0][0][0].weight.detach().clone()
    norm = next(module for module in model.modules() if isinstance(module, nn.BatchNorm3d))
    norm_before = norm.weight.detach().clone()
    adapter = TEGDAAdapter(
        model,
        scope="norm",
        learning_rate=1e-2,
        dropout_samples=2,
        dropout_probability=0.25,
        feature_bank_size=2,
    )
    result = adapter.predict_and_adapt(torch.randn(1, 4, 8, 8, 8))

    assert result.logits.shape == (1, 4, 8, 8, 8)
    torch.testing.assert_close(model.encoder[0][0][0].weight, frozen_before)
    assert not torch.equal(norm.weight, norm_before)
    diagnostics = adapter.case_diagnostics()
    assert diagnostics["tegda_updates"] == 1
    assert diagnostics["tegda_feature_bank_size"] == 1
    assert 0.0 <= float(diagnostics["tegda_quality"]) <= 1.0

    state = adapter.adaptation_state_dict()
    restored = TEGDAAdapter(
        deepcopy(source),
        scope="norm",
        learning_rate=1e-2,
        dropout_samples=2,
        dropout_probability=0.25,
        feature_bank_size=2,
    )
    restored.load_adaptation_state_dict(state)
    for actual, expected in zip(restored.parameters, adapter.parameters):
        torch.testing.assert_close(actual, expected)
    assert len(restored.feature_bank) == len(adapter.feature_bank)
