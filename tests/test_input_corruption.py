import torch

from brats_tta.tta.input_corruption import apply_mri_mixed_corruption


def test_mri_corruption_is_deterministic_preserves_background_and_zscore() -> None:
    torch.manual_seed(4)
    image = torch.zeros(1, 4, 4, 4, 4)
    image[:, :, 1:3, 1:3, 1:3] = torch.randn(1, 4, 2, 2, 2)
    first = apply_mri_mixed_corruption(image, seed=17)
    second = apply_mri_mixed_corruption(image, seed=17)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[:, :, 0]) == 0
    mask = image.abs().sum(dim=1) > 0
    for channel in range(4):
        values = first[0, channel][mask[0]]
        assert abs(values.mean().item()) < 1e-5
        torch.testing.assert_close(values.std(unbiased=False), torch.tensor(1.0))
