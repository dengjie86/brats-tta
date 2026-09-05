from __future__ import annotations

import numpy as np
import pytest

from brats_tta.data.preprocessing import labelmap_to_regions, regions_to_labelmap, zscore_nonzero


def test_nonzero_zscore_preserves_background() -> None:
    image = np.asarray([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float32)
    normalized = zscore_nonzero(image)
    foreground = normalized[image != 0]

    assert np.all(normalized[image == 0] == 0)
    assert abs(float(foreground.mean())) < 1e-6
    assert abs(float(foreground.std()) - 1.0) < 1e-6


@pytest.mark.parametrize(
    ("schema", "enhancing_value"),
    [("brats_modern", 3), ("brats_legacy", 4)],
)
def test_labelmap_region_round_trip(schema: str, enhancing_value: int) -> None:
    label = np.asarray([[[0, 1, 2, enhancing_value]]], dtype=np.uint8)
    regions = labelmap_to_regions(label, schema)

    np.testing.assert_array_equal(regions[0, 0, 0], [0, 0, 0, 1])
    np.testing.assert_array_equal(regions[1, 0, 0], [0, 1, 0, 1])
    np.testing.assert_array_equal(regions[2, 0, 0], [0, 1, 1, 1])
    np.testing.assert_array_equal(regions_to_labelmap(regions, schema), label)


def test_modern_schema_rejects_legacy_enhancing_value() -> None:
    with pytest.raises(ValueError, match="not defined"):
        labelmap_to_regions(np.asarray([[[4]]]), "brats_modern")


def test_pediatric_2024_schema_merges_four_tissues_into_nested_regions() -> None:
    label = np.asarray([[[0, 1, 2, 3, 4]]], dtype=np.uint8)
    regions = labelmap_to_regions(label, "brats_ped_2024")

    np.testing.assert_array_equal(regions[0, 0, 0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(regions[1, 0, 0], [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(regions[2, 0, 0], [0, 1, 1, 1, 1])


def test_output_conversion_repairs_region_hierarchy() -> None:
    regions = np.zeros((3, 2, 2, 2), dtype=np.float32)
    regions[0, 0, 0, 0] = 1.0
    label = regions_to_labelmap(regions, "brats_modern", enforce_hierarchy=True)
    assert label[0, 0, 0] == 3
