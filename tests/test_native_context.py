import numpy as np
import pytest

from brats_tta.cli.context_teacher import main
from brats_tta.native.context import make_contexts, teacher
from brats_tta.native.controls import geometry, matched
from brats_tta.native.coordinates import crop, crop_patch, restore


def test_context_preserves_protected_tissue_and_zero_padding():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(4, 56, 56, 56)).astype(np.float32)
    image[:, :2] = 0
    source = np.zeros(image.shape[1:], np.uint8)
    source[26:30, 26:30, 26:30] = 1
    contexts, protected, info = make_contexts(image, source)
    assert info["reason"] == "ok"
    assert protected[26:30, 26:30, 26:30].all()
    for observed in contexts.values():
        np.testing.assert_array_equal(observed[:, protected], image[:, protected])
        np.testing.assert_array_equal(observed[image == 0], 0)
        assert np.isfinite(observed).all()
        assert not np.array_equal(observed, image)


def test_no_lesion_has_no_accepted_region_or_input_edit():
    image = np.ones((4, 5, 6, 7), np.float32)
    contexts, protected, info = make_contexts(image, np.zeros(image.shape[1:], np.uint8))
    assert info["reason"] == "no_source_WT"
    assert not protected.any()
    for observed in contexts.values():
        np.testing.assert_array_equal(observed, image)


def test_teacher_requires_agreement_confidence_and_protection_for_all_classes():
    source = np.full((1, 1, 7), 2, dtype=np.uint8)
    probability = np.full((4, 1, 1, 7), 0.025, np.float32)
    for i, label in enumerate([0, 1, 2, 3, 1, 1, 1]):
        probability[label, 0, 0, i] = 0.925
    second = probability.copy()
    second[:, 0, 0, 4] = [0.025, 0.025, 0.925, 0.025]  # disagreement
    second[:, 0, 0, 5] = [0.1, 0.7, 0.1, 0.1]  # low confidence
    protected = np.ones_like(source, bool)
    protected[..., 6] = False
    prediction, accepted = teacher(source, [probability, second], protected)
    np.testing.assert_array_equal(prediction.ravel(), [0, 1, 2, 3, 2, 2, 2])
    np.testing.assert_array_equal(accepted.ravel(), [True, True, True, True, False, False, False])


@pytest.mark.parametrize("kind", ["margin", "random", "geometry"])
def test_controls_preserve_edit_direction_counts_and_eligibility(kind):
    rng = np.random.default_rng(5)
    source = rng.integers(0, 4, (8, 9, 7), dtype=np.uint8)
    eligible = np.indices(source.shape)[0] < 5
    target = source.copy()
    target[2, 2, 2:5] = (target[2, 2, 2:5] + 1) % 4
    probabilities = rng.random((4, *source.shape)).astype(np.float32)
    probabilities /= probabilities.sum(0)
    result = (
        geometry(source, target, probabilities, eligible)
        if kind == "geometry"
        else matched(source, target, probabilities, seed=19, random=kind == "random", eligible=eligible)
    )
    np.testing.assert_array_equal(result[~eligible], source[~eligible])
    for src in range(4):
        for dest in range(4):
            assert np.count_nonzero((source == src) & (result == dest)) == np.count_nonzero(
                (source == src) & (target == dest)
            )


def test_coordinates_roundtrip_with_nontrivial_axis_order_and_padding():
    image = np.arange(7 * 8 * 9, dtype=np.int16).reshape(7, 8, 9)
    fw = np.array([2, 0, 1])
    bbox = np.array([[1, 6], [2, 8], [1, 7]])
    cropped = crop(image, bbox, fw)
    restored = restore(cropped, image.shape, bbox, fw)
    np.testing.assert_array_equal(crop(restored, bbox, fw), cropped)
    assert restored.dtype == image.dtype
    patch = crop_patch(np.ones((4, 3, 3, 3), np.float32), [-1, 0, 0], size=4)
    assert not patch[:, 0].any()
    assert patch[:, 1:, :3, :3].all()


def test_cached_cli_creates_prediction_and_refuses_overwrite(tmp_path):
    source = np.zeros((4, 2, 2, 2), np.float32)
    source[2] = 1
    view = np.zeros_like(source)
    view[1] = 1
    for name, key, value in [
        ("source", "view0", source),
        ("blur", "prob", view),
        ("neutral", "prob", view),
        ("geometry", "protected", np.ones((2, 2, 2), bool)),
    ]:
        np.savez(tmp_path / f"{name}.npz", **{key: value})
    args = [
        part
        for name in ["source", "blur", "neutral", "geometry"]
        for part in [f"--{name}", str(tmp_path / f"{name}.npz")]
    ]
    args += ["--output", str(tmp_path / "result.npz")]
    assert main(args) == 0
    original = (tmp_path / "result.npz").read_bytes()
    with np.load(tmp_path / "result.npz") as result:
        assert (result["prediction"] == 1).all() and result["accepted"].all()
    with pytest.raises(SystemExit):
        main(args)
    assert (tmp_path / "result.npz").read_bytes() == original
