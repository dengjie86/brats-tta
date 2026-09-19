import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from brats_tta.cli.evaluate_tta_continual import summarize
from brats_tta.tta.continual_state import load_state, save_state
from brats_tta.tta.tent import TentAdapter, binary_prediction_entropy


def tiny():
    return nn.Sequential(nn.Conv3d(2,4,1), nn.InstanceNorm3d(4, affine=True),
                         nn.LeakyReLU(), nn.Conv3d(4,3,1))


def test_normalized_entropy_weight_decay_matches_reference():
    torch.manual_seed(2)
    model = tiny()
    reference = deepcopy(model)
    adapter = TentAdapter(model, learning_rate=1e-5, weight_decay=.9,
                          normalize_entropy=True, use_amp=False)
    reference.requires_grad_(False)
    reference[1].requires_grad_(True)
    optimizer = torch.optim.Adam(reference[1].parameters(), lr=1e-5, weight_decay=.9)
    for _ in range(3):
        image = torch.randn(1,2,4,4,4)
        optimizer.zero_grad()
        expected = reference(image)
        loss = binary_prediction_entropy(expected).mean()/math.log(2)
        loss.backward()
        optimizer.step()
        actual = adapter.predict_and_adapt(image)
        torch.testing.assert_close(actual.logits, expected.detach())
        assert actual.entropy == pytest.approx(loss.item())
    for a,b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a,b, rtol=0, atol=0)


def test_continual_checkpoint_restores_parameters_momentum_and_records(tmp_path):
    torch.manual_seed(5)
    source = tiny()
    options = dict(learning_rate=1e-5, weight_decay=.9, normalize_entropy=True, use_amp=False)
    first = TentAdapter(deepcopy(source), **options)
    first.predict_and_adapt(torch.randn(1,2,4,4,4))
    path = tmp_path/'state.pt'
    settings = {'order':['a','b']}
    save_state(path, first, settings, [{'id':'a'}])
    next_image = torch.randn(1,2,4,4,4)
    expected = first.predict_and_adapt(next_image)
    resumed = TentAdapter(deepcopy(source), **options)
    assert load_state(path, resumed, settings) == [{'id':'a'}]
    actual = resumed.predict_and_adapt(next_image)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    for a,b in zip(first.parameters, resumed.parameters):
        torch.testing.assert_close(a,b, rtol=0, atol=0)
    assert resumed.optimizer.state[resumed.parameters[0]]['step'].item() == 2
    with pytest.raises(ValueError, match='mismatch'):
        load_state(path, resumed, {'order':['b','a']})


def test_summary_paired_means():
    from brats_tta.cli.evaluate_tta import METRIC_KEYS
    records = [{method:{key:value for key in METRIC_KEYS}
        for method,value in [('source',.5),('tent_online',.6),('tent_post',.4)]}]
    result = summarize(records, 2)
    assert not result['complete']
    assert result['methods']['tent_online']['delta_mean_vs_source'] == pytest.approx(.1)


def test_source_probe_preserves_continual_state_even_on_error():
    from brats_tta.tta.continual_state import source_affine_parameters
    model = tiny().eval()
    image = torch.randn(1,2,4,4,4)
    with torch.no_grad():
        baseline = model(image).clone()
    adapter = TentAdapter(model, use_amp=False)
    adapter.predict_and_adapt(image)
    current = [p.detach().clone() for p in adapter.parameters]
    before_step = adapter.optimizer.state[adapter.parameters[0]]['step'].clone()
    with pytest.raises(RuntimeError, match='test'):
        with source_affine_parameters(adapter) as source:
            with torch.no_grad():
                torch.testing.assert_close(source(image), baseline, rtol=0, atol=0)
            raise RuntimeError('test')
    for a,b in zip(current, adapter.parameters):
        torch.testing.assert_close(a,b, rtol=0, atol=0)
    torch.testing.assert_close(adapter.optimizer.state[adapter.parameters[0]]['step'], before_step)
