# Original U-Net Topology Ablation

This experiment separates the effect of model capacity/topology from the effect
of BatchNorm-based test-time adaptation.

## Paired models

Both variants use the same five-level 3D U-Net topology:

- channels: `[32, 64, 128, 256, 512]`
- four `MaxPool3d(2)` downsamplings
- two `3 x 3 x 3` convolutions and ReLU at each level
- `ConvTranspose3d(2, 2)` upsampling
- concatenated skip connections
- `1 x 1 x 1` four-class output head
- same padding, so a `128^3` patch has a `128^3` output
- no residual block, dropout, deep supervision, or LeakyReLU

The plain U-Net topology control uses no normalization and has 22,578,020
parameters. The paired TENT control adds only `BatchNorm3d` after each
convolution and has 22,580,964 parameters. It is therefore not a capacity
comparison: the parameter-count difference is only the normalization affine
parameters and both models have the same receptive-field topology.

The original U-Net paper used valid convolutions and cropped skip features.
This repository uses the same topology with 3D operators and same padding so a
`128^3` BraTS patch keeps a `128^3` label geometry. A literal five-level valid
convolution port would shrink a `128^3` patch too aggressively before the
bottleneck, so it would require a different patch size and target-cropping
protocol. The present experiment should therefore be called an **original
U-Net topology 3D adaptation**, not an exact reproduction of the 2D paper.

The channel widths `[32, 64, 128, 256, 512]` are the practical 3D version used
for this ablation. They preserve the paper's doubling pattern while avoiding
the much larger 2D paper width `[64, 128, 256, 512, 1024]`, which is not a
like-for-like choice for `128^3` volumetric patches.

## Expected TENT interpretation

- strict model: standard affine-BatchNorm TENT is an explicit no-op because it
  has no BatchNorm parameters; source and TENT predictions should match up to
  numerical noise.
- BatchNorm control: TENT can update only BatchNorm scale/shift parameters and
  can use target-patch statistics. Any improvement or degradation is evidence
  about the adaptation mechanism, not a different U-Net capacity.

## Formal source run

```bash
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats_tta_original_unet_ablation_20260922 \
bash scripts/autodl_original_unet_ablation.sh
```

The script filters the mixed flat archive to `BraTS-GLI-*`, splits by patient,
uses batch size 2, 300 epochs, 250 iterations per epoch, strict FP32, and
saves periodic checkpoints every 20 epochs plus `best.pt`, `last.pt`, and
`latest.pt` for each variant.
