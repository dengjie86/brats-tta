# 历史 Source 训练与评估协议

从根目录 AGENTS 于 2026-10-22 归档。保留实验协议和限制；旧远程状态不是实时状态。当前工作规则和状态见[AGENTS](../../AGENTS.md)与[HANDOFF](../../HANDOFF.md)。

## Historical custom source-model experiment (September 2026)

- Architecture: nnU-Net-style plain six-stage 3D U-Net, features `[32, 64, 128, 256, 320, 320]`, two convolutions per stage, patch `128^3` (31,199,796 parameters).
- Historical experiment: four mutually exclusive classes (`background`, `NCR/NET`, `edema`, `ET`) with Softmax-compatible logits and `BatchNorm3d`. This section describes the old custom trainer, not the October native nnU-Net controls below.
- Training on AutoDL: single-GPU batch 2, strict FP32 with AMP and TF32 disabled, 300 epochs and 250 iterations per epoch on an RTX 4080 SUPER with 32 GiB VRAM.
- AutoDL runtime quota: 16 CPU cores and 80 GiB RAM; use 12 DataLoader workers, training prefetch factor 4, validation prefetch factor 1, and one PyTorch thread per process by default.
- GLI labels use `brats_modern` (`0,1,2,3`). The historical PED2024 five-value schema cannot be used directly with this four-class head; current PED2023 instead uses the four-value mapping above.
- The model remains the plain U-Net for this comparison; do not silently substitute the deeper residual TEGDA architecture.
- New GLI source runs must split by patient ID and evaluate the full validation set for checkpoint selection. The September 19, 2026 checkpoint used a scan-level split and only the first 64 sorted validation scans, so retain it for exploratory TTA only rather than final reported experiments.

## Recovered AutoDL checkpoint and TENT diagnostics

- The complete September 19 source-training run was recovered from AutoDL to `D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn\source_training`. All 29 files matched the remote SHA-256 inventory; `run\checkpoints\best.pt` is the epoch-300 checkpoint. The historical remote code snapshot is at `D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn\autodl_source_snapshot.tar.gz`.
- At the September recovery, old TENT target-result folders were not found remotely. The replacement 10-case PED/SSA diagnostic results, deterministic manifests, logs, and analysis are now under `D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn`; read `TENT_DIAGNOSTIC_REPORT.md` before citing these historical scores.
- `evaluate_tta.py` supports `--tent-inference-mode online|two_pass`. Online is still the default. The two-pass variant made identical adaptation updates but worsened mean Dice on masked PED and SSA pilots; do not silently use it as the standard TENT baseline.
- The raw-versus-HD-BET-masked PED input is a major protocol variable. In the 10-case pilot, masking raised Source mean ET/TC/WT Dice from 0.1098 to 0.5845 and removed most of the BN-statistics-only gain. These are small diagnostic samples, not full-dataset results.
- A separate 159-scan GLI validation subset with no patient ID in training or the first 64 checkpoint-selection scans gave mean ET/TC/WT Dice 0.8934 using the recovered checkpoint. The original logged 0.9154 came from the flawed 64-scan selection set.

## Original U-Net topology ablation

- `src/brats_tta/models/original_unet3d.py` is the plain U-Net topology requested for the ablation: five resolution levels, two `3x3x3` Conv/ReLU operations per block, `MaxPool3d` downsampling, `ConvTranspose3d` upsampling, concatenated skips, and a `1x1x1` four-class head.
- The strict control uses no normalization, dropout, residual block, LeakyReLU, or deep supervision. It has `22,578,020` parameters with features `[32, 64, 128, 256, 512]`.
- The paired BN control keeps the exact same topology and adds only `BatchNorm3d`; it has `22,580,964` parameters and 36 affine BN parameters available to TENT.
- This is a topology-faithful 3D adaptation with same padding, not a literal 2D-paper valid-convolution/crop reproduction. Same padding is required to keep `128^3` BraTS patches aligned with `128^3` labels.
- Formal launch script: `scripts/autodl_original_unet_ablation.sh`. The September 23 record says the run had not started because SSH failed before authentication. No completed Original/Cicek source checkpoint was found in the October 10 project inventory; historical SSH failures do not establish current remote status.

## Cicek 3D U-Net ablation

- The user clarified that the requested standard network is the 2016 Cicek et al. 3D U-Net, not the 2015 2D U-Net topology.
- `src/brats_tta/models/cicek_unet3d.py` follows the authors' released channel topology: encoder outputs `[64, 128, 256, 512]`, three max pools, three transposed convolutions, and 17 BatchNorm layers. Same padding and a four-class head are BraTS-specific adaptations.
- The BN model has `19,075,044` parameters. The current six-stage `[32, 64, 128, 256, 320, 320]` model has `31,199,796`, so the current model has 12,124,752 more parameters and is 1.64 times larger by parameter count.
- The Cicek model is heavier in compute and activations despite fewer parameters: about 1.960 T forward MACs at `128^3`, versus 0.483 T for the current model. Do not infer runtime or VRAM from parameter count alone.
- Formal script: `scripts/autodl_cicek_unet3d_ablation.sh`. It runs a full `128^3`, batch-2 FP32 memory smoke before training, then Cicek BN, the current six-stage BN control on the same patient split, and the Cicek no-BN ablation.
- The previously recorded local pilot path `outputs/brats_tta_cicek_unet3d_ablation_20260923/pilot` is absent as of October 10; do not present it as an available artifact or infer where it went. Historical pilots validated the pipeline only, not reportable Dice results.
- Kaggle memory probe completed on September 23, 2026 with two Tesla T4 GPUs (14.56 GiB each). On one T4, Cicek BN passed `128^3`, batch 1, strict FP32 at 10,862 MiB peak allocated, while `128^3`, batch 2 OOMed. The private kernel is `nagisa0v0/brats-gli-cicek-3d-u-net-128-fp32-probe`; use batch 1 per GPU for Kaggle DDP rather than treating the two cards as pooled memory.

## Native nnU-Net four-class BatchNorm control (October 4, 2026)

- The user requested a norm-only control against the completed native Dataset503 four-class InstanceNorm model. The new experiment changes only `3d_fullres` plans `norm_op` to `torch.nn.modules.batchnorm.BatchNorm3d`; it does not restore the historical custom BN model's training protocol.
- Local bundle and monitor: `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_4class_bn_20261004`. Remote root on the second SSH endpoint: `/root/autodl-fs/hdtta_4class_bn_20261004`.
- Verified locally and remotely: 22 normalization layers, 31,199,796 parameters, identical trainable initialization and module tree except normalization. BN uses eps 1e-5, affine parameters, momentum 0.1 and running statistics.
- Keep native Dice+CE (Dice excludes background), patient-disjoint 998/253 split, batch 2, patch 128 cubed, AMP, seed 2025, native augmentation and 250 epochs x 250 iterations. Metadata/plans/results were isolated; training shared unchanged deterministic Dataset503 preprocessing arrays with the IN control. Retired local training arrays were subsequently cleaned; reconstruct them from retained originals/plans before rerunning that historical workflow.
- The remote queue completed WT, then BN from random initialization. BN finished all 250 epochs and final 253-case validation on October 4, 2026 at 11:44 Beijing time. The second AutoDL machine had 12 CPU cores, 90 GiB system RAM and 32 GiB GPU VRAM; native runs used AMP and eight augmentation workers.
- The run preserved latest each epoch, numbered checkpoints each 25 epochs, and best selected using full 253-volume native tissue-class mean Dice. This selection metric is distinct from WT/TC/ET region mean. Its `status.ps1` / `status.cmd` are historical monitoring tools; they do not imply an active job.

## Completed source models (October 5 records, status corrected October 10)

- Read root `HANDOFF.md` for current experiment results and paths; the older timeline is in `docs/archive/HANDOFF_before_maintenance_20261010.md`. The current Kaggle PED2023 Source/context experiments use native preprocessing without additional HD-BET. Keep historical HD-TTA and masked TENT protocols separate.
- Completed native IN binary/four-class Dice+CE backups: `D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece`; both use final/best epoch 250. IN four-class GLI region mean WT/TC/ET is 91.06%, while native tissue-class mean is 83.90%.
- Completed pure-Dice WT backup: `D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice`. Best epoch 225 WT Dice 93.64%; final epoch 250 WT Dice 93.61%. PED evaluation uses final epoch 250.
- Completed BN backup: `D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece`. All 55 files / 13 checkpoints matched the remote SHA-256 inventory (`download_verified.json`); best/final optimizer and AMP states and 22 unique BN buffers verified (`checkpoint_inspection.json`). Best epoch 225 tissue-class mean 83.55969%; final epoch 250 mean 83.51887%. Original remote BN validation masks were not downloaded.
- Local BN best-225 region reevaluation finished 253/253 cases on October 5 at 15:29 Beijing time: `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_GLI_BN_regions_20261005`. GLI WT 93.75471%, TC 91.69634%, ET 86.02339%; region mean 90.49148%. These are source validation scores, not PED. Native argmax masks are merged as WT={1,2,3}, TC={1,3}, ET={3}; both-empty region Dice=1. BN statistics unchanged; local native tissue mean matches remote within 0.00027 percentage points. Small per-case differences exist across GPU/software.
- BN PED statistics evaluation completed 24 cases on October 8; see `docs/ped2023_bn_statistics_pilot_20261008.md`. WT improved on average, TC/ET deteriorated relative to its own BN Source. A full 99-case BN evaluation has not been established. This is distinct from both the IN PED run at `artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003` and the CPU-only BN input-style proposal.
- Pure-Dice PED official-code evaluation: `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_pure_dice_20261004`; completed 99 cases, one gate trigger. Paper-equation control: `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_paper_20261004`; completed both sum/mean reduction conventions, still one trigger. The paper and public code differ; see their protocols before claiming strict reproduction.
- October 5 SSH checks failed before authentication with `Error reading SSH protocol banner`; current remote power state is unconfirmed. Do not assume a job is still training or start a paid machine without authorization.

## Completed PED2023 per-case analysis (October 5, 2026)

- Local root: `D:/Workspace/deeplearn/3dunet/artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005`. User selected four-class IN Dice+CE final/best epoch 250. All 99 cases finished at 19:37 Beijing time, exit code 0. Uses native eight-view mirroring and four-class argmax masks, with frozen weights and no HD-TTA refinement.
- WT/TC/ET mean Dice: 89.3334% / 48.8627% / 54.6713%; region mean 64.2891%. Mean HD95: 5.2672 / 14.7466 / 9.3240 mm, valid counts 99 / 92 / 85. Both-empty Dice=1 and HD95=0; one-empty Dice=0 and HD95=null. WT is merged after argmax, unlike the earlier HD-TTA marginal probability threshold.
- `outputs/full/index.html` browses all cases; `outputs/full/ANALYSIS.md`, `analysis.html` and `cohort_analysis.png` document cohort findings. Each case has prediction/reference/error NIfTI, metrics, two PNG figures and an HTML page. Root-level `verification.json` and `review_verification.json` both passed, including 198 case figures, links, artifact hashes and unchanged model weights. These receipts are not inside `outputs/full/`.
- Main finding: 45 cases have WT Dice >=0.8 but TC Dice <0.5; 88/99 have TC volume underestimation >10%. NC-to-ED confusion affects 58.50% of pooled NC voxels, not a patient-average rate. Do not infer a causal mechanism from these observations alone.
- The 2023 Kaggle mirror has labels 0/1/2/3 and uses project mapping 1=NC, 2=ED, 3=ET; do not apply the PED2024 five-value schema. On October 8, official numeric definitions were verified in the BraTS-PED2023 results paper, https://arxiv.org/html/2407.08855v3 section 2.2. Voxelwise equivalence between the Kaggle mirror and official original files remains unverified. Read the analysis directory's `PROTOCOL.md` and this repository's root `HANDOFF.md` before comparisons. Existing masks and reports are sufficient for follow-up analysis; do not rerun full inference just to inspect results.
