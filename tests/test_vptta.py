from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from brats_tta.tta.vptta import FourierPrompt3D, VPTTAAdapter


class TinyVPTTAModel(nn.Module):
    output_mode = "classes_softmax"
    in_channels = 2

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


def test_fourier_prompt_of_ones_is_identity_for_four_modalities() -> None:
    prompt = FourierPrompt3D(4, (8, 8, 8), prompt_alpha=0.25)
    inputs = torch.randn(1, 4, 8, 8, 8)
    prompted, low_frequency = prompt(inputs)

    assert prompt.data_prompt.shape == (1, 4, 2, 2, 2)
    assert low_frequency.shape == (1, 4, 2, 2, 2)
    torch.testing.assert_close(prompted, inputs, atol=2e-5, rtol=2e-5)


def test_vptta_updates_prompt_memory_and_round_trips_state() -> None:
    torch.manual_seed(37)
    source = TinyVPTTAModel()
    adapter = VPTTAAdapter(
        deepcopy(source),
        patch_size=(4, 4, 4),
        prompt_alpha=0.5,
        neighbors=1,
        memory_size=2,
        learning_rate=0.05,
    )
    images = torch.randn(1, 2, 4, 4, 4)
    first = adapter.predict_and_adapt(images)
    second = adapter.predict_and_adapt(images * 1.1)

    assert first.logits.shape == second.logits.shape == (1, 4, 4, 4, 4)
    assert len(adapter.memory) == 2
    assert adapter.case_diagnostics()["vptta_updates"] == 2
    assert adapter.adaptive_batch_norms[0].sample_count.item() == 2
    assert torch.count_nonzero(adapter.prompt.data_prompt - 1.0) > 0

    state = adapter.adaptation_state_dict()
    restored = VPTTAAdapter(
        deepcopy(source),
        patch_size=(4, 4, 4),
        prompt_alpha=0.5,
        neighbors=1,
        memory_size=2,
        learning_rate=0.05,
    )
    restored.load_adaptation_state_dict(state)
    torch.testing.assert_close(restored.prompt.data_prompt, adapter.prompt.data_prompt)
    assert len(restored.memory) == len(adapter.memory)
    assert restored.adaptive_batch_norms[0].sample_count.item() == 2
