"""Per-patient adaptation rounds, each evaluated with one fixed final model."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import torch

from brats_tta.engine.inference import sliding_window_logits
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.tent import TentAdapter


def iter_tent_round_predictions(
    adapter: TentAdapter, image: torch.Tensor, *, rounds: int,
    patch_size: Sequence[int], overlap: float = .5, sw_batch_size: int = 1,
    gaussian_weighting: bool = True,
    progress_callback: Callable[[int, str, int, int], None] | None = None,
) -> Iterator[tuple[int, torch.Tensor, dict]]:
    """Yield source (round 0) then post-update full-volume predictions.

    Reset exactly once per patient. Adam state carries across rounds. Each
    adaptation round updates once per patch batch. The separate prediction
    sweep is no-grad and does not call the adapter or change optimizer state.
    This function intentionally has no argument for a segmentation label.
    """
    if rounds < 1 or adapter.steps != 1:
        raise ValueError('Rounds require rounds >= 1 and exactly one update per patch batch')
    adapter.reset()
    settings = dict(patch_size=patch_size, overlap=overlap, sw_batch_size=sw_batch_size,
                    gaussian_weighting=gaussian_weighting)
    cumulative = 0
    for round_index in range(rounds + 1):
        def notify(phase: str, done: int, total: int) -> None:
            if progress_callback is not None:
                if image.device.type == 'cuda':
                    torch.cuda.synchronize(image.device)
                progress_callback(round_index, phase, done, total)

        adaptation = {}
        if round_index:
            # The online stitched output is deliberately NOT the evaluated output.
            discarded, adaptation = sliding_window_tent_logits(adapter.model, adapter, image,
                **settings, progress_callback=lambda done, total: notify('adapt', done, total))
            del discarded
            cumulative += adaptation['adaptation_updates']
            if not all(torch.isfinite(p).all().item() for p in adapter.parameters):
                raise FloatingPointError('Non-finite adapted affine parameters')
        logits = sliding_window_logits(adapter.model, image, **settings, amp=adapter.use_amp,
            progress_callback=lambda done, total: notify('predict', done, total))
        yield round_index, logits, {**adaptation, 'cumulative_updates': cumulative,
            'prediction_timing': 'fixed_model_after_complete_round'}
        # Release the previous full-volume output before allocating the next graph.
        del logits
