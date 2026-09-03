from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn

from brats_tta.engine.inference import save_brats_prediction, sliding_window_logits


class PointwiseModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        channel = image[:, :1]
        return torch.cat((channel, 2.0 * channel, -channel), dim=1)


def test_sliding_window_matches_pointwise_full_volume() -> None:
    image = torch.randn((1, 1, 17, 18, 19))
    model = PointwiseModel().eval()
    expected = model(image)
    actual = sliding_window_logits(
        model,
        image,
        patch_size=(8, 8, 8),
        overlap=0.5,
        sw_batch_size=3,
        gaussian_weighting=True,
        amp=False,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_sliding_window_crops_padding_to_original_shape() -> None:
    image = torch.randn((1, 1, 5, 6, 7))
    model = PointwiseModel().eval()
    actual = sliding_window_logits(model, image, patch_size=(8, 8, 8), amp=False)
    torch.testing.assert_close(actual, model(image), atol=1e-6, rtol=1e-6)


def test_save_prediction_preserves_nifti_geometry(tmp_path: Path) -> None:
    shape = (5, 6, 7)
    affine = np.diag([1.1, 1.2, 1.3, 1.0])
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference_path)
    probabilities = torch.zeros((1, 3, *shape))
    probabilities[:, 2, 1:4, 1:5, 1:6] = 1
    probabilities[:, 1, 2:4, 2:4, 2:5] = 1
    probabilities[:, 0, 3:4, 3:4, 3:4] = 1
    destination = tmp_path / "prediction.nii.gz"

    save_brats_prediction(
        probabilities,
        reference_path,
        destination,
        label_schema="brats_modern",
    )
    prediction = nib.load(destination)
    assert prediction.shape == shape
    np.testing.assert_allclose(prediction.affine, affine)
    assert prediction.get_data_dtype() == np.dtype(np.uint8)
    assert set(np.unique(np.asanyarray(prediction.dataobj))) == {0, 1, 2, 3}
