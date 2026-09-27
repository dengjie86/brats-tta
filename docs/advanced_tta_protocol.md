# Advanced TTA protocol

This document defines the reproducible SAR, VPTTA and TEGDA evaluation path for the
current four-class BatchNorm3d source model. The existing Source/TENT evaluator is
kept separate so completed official results cannot be mixed with continual method
state.

## Common evaluation rules

- Source checkpoint, target manifest and case order are fingerprinted before a run.
- One method is evaluated per output directory and per process.
- Inference is strict FP32; AMP and TF32 are disabled.
- Target labels are never used by adaptation. They are loaded only after prediction
  to calculate ET/TC/WT Dice and optional physical-space HD95.
- The default protocol is continual across patches and cases. `--reset-per-case`
  provides an explicitly episodic ablation.
- `cases.jsonl`, `summary.json`, `run_settings.json`, `case_order.json`,
  `evaluation.log`, `progress.json` and `adapter_state.pt` are retained.
- A pending-state transaction keeps JSONL records and continual state aligned after
  interruption. Resume only in the same output directory with identical settings.

## SAR

Reference: `mr-eggplant/SAR`, commit
`20f6e24b17525f34503510afccedc0629b67b7c4`, plus the BraTS comparison in the
TEGDA repository.

Defaults:

- SAM-SGD, learning rate `1e-3`, momentum `0.9`, rho `0.05`.
- reliable entropy margin `0.4 * ln(4)`.
- recovery threshold `0.2`.
- one update per sliding-window patch batch.
- only BatchNorm3d affine scale and shift are trainable.

The classification sample filter is applied per voxel for dense segmentation. This
is therefore a documented 3D dense port rather than an exact classification run.
Because low entropy can make the official recovery rule reset every patch on this
source model, final experiments retain both the default recovery run and a separate
`--sar-no-recovery` ablation. They must use different output directories.

## VPTTA

Reference: `Chen-Ziyang/VPTTA`, commit
`7e0a3776100d1350cd2bd0c9c3431522c5940027`.

Defaults follow the TEGDA BraTS fork where it provides a 3D setting:

- four-modality 3D Fourier prompt, alpha `0.03`.
- Adam, learning rate `0.05`, betas `(0.9, 0.99)`, one update.
- memory capacity `40`, nearest neighbors `16`, temperature `0.2`.
- AdaBN warm-up constant `5`.
- batch size one.

The original implementation is 2D RGB. This port applies FFT over all three spatial
axes and uses a prompt with shape `(1, 4, pD, pH, pW)`. It intentionally does not
reproduce the TEGDA fork's malformed one-channel fallback prompt.

## TEGDA compatibility port

Reference: `HiLab-git/TEGDA`, commit
`2f515079a69f8e7312084ab6974162afb5622315`.

The formal defaults follow the MICCAI paper's BraTS implementation details:

- input patch `4 x 128 x 128 x 128` and TTA batch size `1`.
- update all model parameters once per test batch.
- Adam, learning rate `1e-4`, betas `(0.5, 0.999)` in this compatibility port.
- 10 stochastic inferences, bottleneck dropout probability `0.5`.
- ADIC quality over ET/TC/WT.
- quality-selected feature bank of 10 entries, 90th-percentile admission.
- cosine-softmax fusion over all retained bank entries, followed by ADIC-weighted
  blending with the current bottleneck feature.
- quality-weighted AFFR target and quality-guided mean teacher.
- formal default scope `full`; `--tegda-scope norm` is a memory-saving ablation only.

The common inference geometry intentionally differs from the paper. The paper
resizes each BraTS volume to `4 x 128 x 128 x 128` and adapts once per whole-volume
batch. This project keeps each target on its native grid and applies adaptation to
every `128^3` sliding-window patch batch with overlap `0.5`, matching the already
completed Source and TENT protocol. The comparison is therefore internally fair,
but its absolute scores are not a direct reproduction of the paper's table.

This result must be named `TEGDA 3D compatibility port`, not an exact TEGDA
reproduction. The trained plain U-Net has no dropout modules, while the released
TEGDA source network does. Stochastic dropout is therefore applied only to the
bottleneck. The released 3D TEGDA file also imports a 2D feature-update module; this
port uses native 3D bottleneck features. Refined and teacher predictions are detached
targets so only one adaptation graph is retained in GPU memory.

The paper describes Dice+CE segmentation consistency, while the released 3D code
uses symmetric soft cross-entropy. This compatibility port follows the released
optimization objective and records that choice in `run_settings.json`; the paper's
feature-bank length, admission percentile and all-parameter scope are retained.
The released 3D ADIC implementation also computes hard ET/TC/WT agreement and a
region confidence weight, whereas the paper writes ADI as a class-probability
soft-Dice expression. This port follows the released 3D region-wise ADIC path and
records that difference as well.

## Pilot versus final experiment

`scripts/run_advanced_tta_pilot.ps1` uses a fixed prefix of each manifest, a `96^3`
patch and overlap zero to validate execution, memory, state resume and metric output
on the local 8 GiB GPU. It also calculates Source and TENT under the same inference
settings so pilot scores have a matching baseline. It retains both SAR recovery
settings. Its TEGDA pilot uses three dropout samples for speed. Pilot scores are not
final-cohort results.

For final reporting, evaluate all 60 SSA and all 260 valid brain-masked PED cases,
restore overlap `0.5`, use 10 TEGDA dropout samples, keep the same case order, and
place each method/dataset pair in a fresh output directory. On AutoDL, use
`scripts/autodl_evaluate_advanced_tta.sh`; it runs the default SAR, SAR without
recovery, VPTTA and the TEGDA compatibility port in resumable directories.

```bash
CHECKPOINT=/root/autodl-fs/results/checkpoint/best.pt \
SSA_MANIFEST=/root/autodl-fs/results/manifests/ssa_raw.json \
PED_MANIFEST=/root/autodl-fs/results/manifests/ped_brain.json \
WORK_ROOT=/root/autodl-fs/brats_advanced_tta_full_20260921 \
bash scripts/autodl_evaluate_advanced_tta.sh
```

The AutoDL script uses the paper-aligned TEGDA settings by default: full-parameter
adaptation, learning rate `1e-4`, dropout `0.5`, 10 stochastic inferences, feature
bank length `10`, and admission percentile `0.90`. A one-case memory smoke test is
run automatically before the full matrix. The lighter SAR and VPTTA runs are
completed for both datasets before the substantially slower TEGDA runs. Do not mix
a `norm`-scope ablation with this output directory.
