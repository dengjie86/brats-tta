from copy import deepcopy

import torch
from test_tent import TinyInstanceNormModel

from brats_tta.cli.evaluate_tta_rounds import summarize
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.rounds import iter_tent_round_predictions
from brats_tta.tta.tent import TentAdapter


def test_rounds_predict_after_update_and_do_not_reset_between_rounds():
    torch.manual_seed(41)
    model = TinyInstanceNormModel().eval()
    reference = deepcopy(model)
    image = torch.randn(1, 2, 8, 8, 16)
    adapter = TentAdapter(model, use_amp=False)
    manual = TentAdapter(reference, use_amp=False)
    kwargs = dict(patch_size=(8,8,8), overlap=0)
    source = sliding_window_logits(reference, image, **kwargs, amp=False)
    frozen = model.layers[0].weight.detach().clone()
    events = []
    observed = []
    for step, logits, info in iter_tent_round_predictions(adapter, image, rounds=5, **kwargs,
        progress_callback=lambda *event: events.append(event)):
        if step:
            sliding_window_tent_logits(reference, manual, image, **kwargs)
        expected = sliding_window_logits(reference, image, **kwargs, amp=False)
        torch.testing.assert_close(logits, expected)
        assert info['cumulative_updates'] == step * 2
        if step:
            assert int(next(iter(adapter.optimizer.state.values()))['step']) == step * 2
        else:
            torch.testing.assert_close(logits, source)
            assert not adapter.optimizer.state
        observed.append(logits.clone())
        torch.testing.assert_close(model.layers[0].weight, frozen)
    assert len(events) == 22  # 12 prediction patches, 10 adaptation patches.
    # A second patient with the same image replays the complete episode exactly.
    for step, logits, _ in iter_tent_round_predictions(adapter, image, rounds=5, **kwargs):
        torch.testing.assert_close(logits, observed[step])


def test_summary_pairs_only_complete_patients():
    def row(value):
        return {k:value for k in ('dice_ET','dice_TC','dice_WT','dice_mean','hierarchy_violation')}
    result = summarize([
        {'complete':True, 'rounds':[row(.5),row(.6)]},
        {'complete':False, 'rounds':[row(.9)]},
    ], rounds=1, expected=2)
    assert not result['complete']
    assert result['rounds']['0']['available_cases'] == 2
    assert result['rounds']['0']['paired_cases'] == 1
    assert result['rounds']['0']['paired_metrics_mean']['dice_mean'] == .5
    assert abs(result['rounds']['1']['paired_delta_mean_vs_source'] - .1) < 1e-6
