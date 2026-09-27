from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from brats_tta.tta.sar import SARAdapter


class TinySARModel(nn.Module):
    output_mode = "classes_softmax"

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.01),
            nn.Conv3d(4, 4, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def test_sar_updates_only_batch_norm_and_round_trips_state() -> None:
    torch.manual_seed(31)
    source = TinySARModel()
    model = deepcopy(source)
    frozen_before = model.layers[0].weight.detach().clone()
    norm_before = model.layers[1].weight.detach().clone()
    adapter = SARAdapter(
        model,
        learning_rate=1e-2,
        entropy_margin=10.0,
        recovery_threshold=None,
    )
    result = adapter.predict_and_adapt(torch.randn(1, 2, 4, 4, 4))

    assert result.logits.shape == (1, 4, 4, 4, 4)
    torch.testing.assert_close(model.layers[0].weight, frozen_before)
    assert not torch.equal(model.layers[1].weight, norm_before)
    assert adapter.case_diagnostics()["sar_updates"] == 1

    state = adapter.adaptation_state_dict()
    restored = SARAdapter(
        deepcopy(source),
        learning_rate=1e-2,
        entropy_margin=10.0,
        recovery_threshold=None,
    )
    restored.load_adaptation_state_dict(state)
    for actual, expected in zip(restored.parameters, adapter.parameters):
        torch.testing.assert_close(actual, expected)


def test_sar_skips_patch_when_no_voxel_passes_reliability_filter() -> None:
    adapter = SARAdapter(
        TinySARModel(),
        entropy_margin=1e-8,
        recovery_threshold=None,
    )
    adapter.predict_and_adapt(torch.zeros(1, 2, 4, 4, 4))
    diagnostics = adapter.case_diagnostics()
    assert diagnostics["sar_updates"] == 0
    assert diagnostics["sar_skipped_updates"] == 1
