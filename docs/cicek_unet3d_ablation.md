# Cicek 3D U-Net Ablation

This is the source-model comparison requested after distinguishing the 2015
2D U-Net from the 2016 3D U-Net by Cicek et al.

## Implemented topology

The primary model follows the channel topology in the authors' released
`3dUnet_miccai2016_with_BN.prototxt`:

- four resolution levels with encoder outputs `[64, 128, 256, 512]`
- two `3 x 3 x 3` convolutions per encoder level
- `MaxPool3d(2)` downsampling
- `ConvTranspose3d(2, 2)` followed by BatchNorm and ReLU
- concatenated skip connections
- two convolution, BatchNorm and ReLU operations after each concatenation
- 17 `BatchNorm3d` layers in total
- a `1 x 1 x 1` four-class BraTS output head

The released Caffe network uses valid convolutions and crops its labels to the
smaller output. This project changes only that geometry to same padding so a
`128^3` BraTS input remains aligned with its `128^3` target. The source uses
four MRI modalities rather than the paper's microscopy input and predicts the
four mutually exclusive BraTS classes.

The Caffe BatchNorm layers specify `use_global_stats: false`, including at test
time. The PyTorch port therefore uses `track_running_stats: false`; affine
scale and shift remain trainable and available to TENT.

## Capacity comparison

| Model | Parameters | 128-cubed forward MACs | Relative parameters |
|---|---:|---:|---:|
| Cicek 3D U-Net + BN | 19,075,044 | 1.960 T | 1.00x |
| Cicek 3D U-Net without BN | 19,071,780 | 1.960 T | 1.00x |
| Current six-stage plain U-Net | 31,199,796 | 0.483 T | 1.64x |

The current model has 12,124,752 more parameters. However, the same-padded
Cicek model has 64 full-resolution feature channels, compared with 32 in the
current model. It therefore requires substantially more compute and activation
memory even though its stored parameter count is lower.

Local RTX 4060 Laptop GPU synthetic training-step measurements, using strict
FP32, SGD, forward, loss, backward and optimizer step, were:

| Patch and batch | Cicek BN peak allocated / time | Current peak allocated / time |
|---|---:|---:|
| `64^3`, batch 1 | 1,423 MiB / 0.97 s | 622 MiB / 0.52 s |
| `96^3`, batch 1 | 4,632 MiB / 2.39 s | 1,820 MiB / 1.05 s |

These are implementation smoke measurements rather than throughput benchmarks.
The AutoDL script runs a full `128^3`, batch-2 synthetic step on the target GPU
before beginning the 300-epoch run, because cuDNN workspace selection and peak
memory vary by GPU and PyTorch build.

An end-to-end real-NIfTI pilot was also completed for both models with the same
patient manifests, one `64^3` training iteration, one full validation case,
batch 1 and FP32. The Cicek run reported 5,301 MiB peak allocated and the
current model reported 5,036 MiB. Their first-iteration times were 34.9 s and
10.1 s respectively, but those values include data loading, augmentation,
cuDNN algorithm selection and cold-start overhead. The pilot validates the
pipeline only; its randomly initialized Dice values are not experimental
results. Pilot artifacts are outside the repository at
`D:\PersonalData\Desktop\brats_tta_cicek_unet3d_ablation_20260923\pilot`.

## Kaggle T4 probe (September 23, 2026)

The current source snapshot was submitted to the private Kaggle kernel
`nagisa0v0/brats-gli-cicek-3d-u-net-128-fp32-probe`. Kaggle exposed two Tesla
T4 GPUs, each with about 14.56 GiB usable memory. The probe used one GPU and a
strict FP32 forward, loss, backward, and optimizer step:

| Patch and batch | Result | Peak allocated / reserved | Step time |
|---|---|---:|---:|
| `64^3`, batch 1 | passed | 1,423 / 1,660 MiB | 1.75 s |
| `96^3`, batch 1 | passed | 4,634 / 5,390 MiB | 2.13 s |
| `128^3`, batch 1 | passed | 10,862 / 12,620 MiB | 4.37 s |
| `128^3`, batch 2 | OOM | 12,370 / 12,398 MiB before failure | not applicable |

The real-NIfTI pilot also completed one augmented `64^3`, batch-1 training
step. Therefore Kaggle is suitable for a per-GPU batch of 1 at `128^3`; using
both T4s could provide a global batch of 2 through DDP, but that is a
distributed experiment and is not identical to the planned single-GPU 32 GiB
run. The downloaded probe report is outside the repository at
`D:\PersonalData\Desktop\kaggle_cicek_probe_results_20260923\cicek_memory_probe_results.json`.

## Formal comparison

All variants use the same patient-level split, four-class loss, augmentations,
batch size 2, `128^3` patches, 300 epochs, 250 iterations per epoch, and strict
FP32. The launch order is:

1. Cicek 3D U-Net with BN
2. current six-stage BN model, retrained on the same valid patient split
3. Cicek 3D U-Net without BN

```bash
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats_tta_cicek_unet3d_ablation_20260923 \
bash scripts/autodl_cicek_unet3d_ablation.sh
```

The no-BN run can be skipped with `RUN_NO_BN_ABLATION=0`. The current-model
control can be skipped with `RUN_CURRENT_CONTROL=0`, but it is required for a
valid capacity comparison because the old checkpoint used a different split
and incomplete validation selection.
