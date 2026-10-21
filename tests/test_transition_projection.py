import numpy as np
import pytest

from brats_tta.native.adaptation import project_transition_delta


def test_projection_preserves_all_class_pair_contrasts_and_is_minimum_norm():
    rng = np.random.default_rng(11)
    source, target = np.indices((4, 4))
    delta = rng.normal(size=(4, 4, 4)).astype(np.float32)
    delta -= delta.mean(0)
    projected = project_transition_delta(delta, source, target)
    for a in range(4):
        for b in range(4):
            v = projected[:, a, b]
            if a == b:
                np.testing.assert_array_equal(v, 0)
                continue
            assert v[b] - v[a] == pytest.approx(delta[b, a, b] - delta[a, a, b])
            assert v.sum() == 0
            # Feasible perturbations orthogonal to e_b-e_a cannot lower norm.
            perturbation = rng.normal(size=4)
            perturbation[[a, b]] = perturbation[[a, b]].mean()
            assert np.dot(v, perturbation) == pytest.approx(0, abs=1e-7)
            assert np.linalg.norm(v + perturbation) >= np.linalg.norm(v) - 1e-7
    np.testing.assert_allclose((projected * (delta - projected)).sum(0), 0, atol=4e-7)


def test_projection_is_class_symmetric_and_ignores_common_logit_shift():
    source, target = np.indices((4, 4))
    rng = np.random.default_rng(12)
    delta = rng.normal(size=(4, 4, 4)).astype(np.float32)
    baseline = project_transition_delta(delta, source, target)
    permutation = np.array([3, 1, 0, 2])  # old class -> new class
    shuffled = np.empty_like(delta)
    shuffled[permutation] = delta
    actual = project_transition_delta(shuffled, permutation[source], permutation[target])
    np.testing.assert_array_equal(actual[permutation], baseline)
    np.testing.assert_allclose(project_transition_delta(delta + 2, source, target), baseline, atol=2e-7)


def test_projection_no_edits_and_invalid_inputs():
    labels = np.zeros((2, 3, 4), dtype=np.uint8)
    delta = np.ones((4, *labels.shape), dtype=np.float32)
    np.testing.assert_array_equal(project_transition_delta(delta, labels, labels), 0)
    with pytest.raises(ValueError):
        project_transition_delta(delta[:3], labels, labels)
    with pytest.raises(ValueError):
        project_transition_delta(delta, labels, labels + 4)
