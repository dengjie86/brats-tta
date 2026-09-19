# Why the classic TTA baselines were ineffective

## Scope and controls

- Source checkpoint: `best.pt`, SHA256
  `c8107f2c3e2d79e22d3ab5aa10fce01e28f55985854dcb85157b9faf0abd83c9`.
- Evaluation is FP32 with AMP and TF32 disabled.
- Adaptation never reads target labels. Labels are used only after a complete
  prediction has been produced, to report ET/TC/WT Dice.
- Hyperparameter and mechanism checks use five fixed BraTS 2021 source
  validation cases. PED and SSA labels are not used to select a variant.
- The source network predicts three overlapping sigmoid regions and contains
  18 affine `InstanceNorm3d(track_running_stats=False)` layers, not BatchNorm.

## Root causes established by diagnostics

| Failure mechanism | Direct evidence | Consequence |
|---|---|---|
| Optimizer decay overwhelmed the TTA signal | The inherited Adam setting used `weight_decay=0.9`. The source IN-affine norm is 27.3803, so the decay-gradient norm is 24.6423. First-patch all-voxel entropy-gradient norms were only 0.000189--0.000459: decay was about 53,700--130,451 times larger. | The apparent TENT/CoTTA parameter movement was largely data-independent shrinkage rather than adaptation. |
| Dense background diluted entropy | Across 90 clean source patches, normalized all-voxel entropy averaged 0.000650, brain-only entropy 0.001704, and foreground/background-balanced entropy 0.057215. Corresponding first-patch IN-affine gradient norms increased from 0.000189--0.000459 to as much as 2.096 after balancing. | A classification-style mean over all voxels mostly optimizes already-certain background. |
| SAR recovery threshold was on the wrong scale | The inherited recovery threshold was 0.02, while the clean-source SAR objective averaged about 0.000608 and never exceeded 0.002091 in the five-case diagnostic. All 108 target pilot patch updates therefore triggered recovery; parameter delta was exactly zero. | SAR mathematically erased every update. |
| CoTTA's confidence gate measured background | Global Bernoulli confidence was 0.9977--0.99998 on target patches, so the augmentation branch opened 0 times in 108 pilot updates. At initialization teacher and student are identical, so unaugmented consistency has zero theoretical gradient. | CoTTA had no image-dependent teaching signal; decay/restoration accounted for its movement. |
| Numerically zero CoTTA gradients were still amplified | On source case `BraTS2021_00046`, no augmentation fired, yet Adam moved the parameters by L2 about 0.07. | Tiny floating-point residuals were normalized into optimizer steps despite no teacher/student disagreement. |
| The first CoTTA port violated preprocessing | Intensity shift/noise made exact-zero cranial background nonzero, and it added 90-degree rotations absent from source training. | Untracked InstanceNorm statistics were changed by synthetic padding/background and unfamiliar transforms. |
| Continual updates drifted across unrelated patients | For balanced TENT, continual IN-affine L2 drift grew from about 0.06 to 0.19 across five source cases. The final case lost about 0.00429 Dice; with per-case reset the loss shrank to about 0.00088. | Patient order, rather than only the current image, controlled the result. |

The absence of BatchNorm is therefore only part of the compatibility story.
Methods whose defining state is BN running statistics cannot be reproduced
exactly on this checkpoint. TENT and SAR can still adapt affine InstanceNorm
parameters, and CoTTA can still use a teacher, but their classification-style
reductions and thresholds must be mapped to dense sigmoid segmentation.

This diagnosis agrees with the original TENT formulation, which adapts
normalization affine parameters by entropy minimization, and with InTEnt's
observation that direct entropy minimization often produces only minor changes
for single-image medical segmentation and is sensitive to foreground/background
imbalance. See the [TENT paper](https://openreview.net/pdf?id=uXl3bZLkr3c),
[SAR paper](https://openreview.net/pdf?id=g2YraF75Tj),
[CoTTA paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf),
and [InTEnt paper](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Dong_Medical_Image_Segmentation_with_InTEnt_Integrated_Entropy_Weighting_for_Single_CVPRW_2024_paper.pdf).

## Implemented architecture-compatible fixes

### Dense TENT

- Set Adam weight decay to zero.
- Compute entropy inside the image-derived brain support.
- Average predicted foreground and background groups equally for ET/TC/WT.
- Reset model and optimizer state at each patient.
- Keep the online and post-adaptation outputs separate.

### Dense SAR

- Retain SAM-SGD, the entropy margin, and IN-affine-only updates.
- Apply reliability filtering inside brain support and balance predicted
  foreground/background groups.
- Disable the invalid `0.02` recovery threshold for this dense Bernoulli scale.
- Reset at every patient.

### Dense CoTTA

- Set Adam weight decay to zero and stochastic restoration probability to 0.01.
- Gate augmentation using the least-confident 1% of brain voxels rather than a
  global background-dominated mean.
- Use foreground/background-balanced Bernoulli consistency.
- Use exactly source-training-compatible transforms: independent three-axis
  flips, per-modality intensity scale/shift and probabilistic noise; preserve
  exact-zero background and do not rotate.
- Skip optimizer updates when brain teacher/student probability MAE is at most
  `1e-7`. On the real no-signal case this changed 18 false updates and L2 drift
  0.07 into 18 skips, exactly zero drift, and bit-identical Dice.
- Seed stochastic adaptation from the case ID so episodic results do not depend
  on patient order.

## Fixed three-case target pilot

The same seed-selected cases and source checkpoint were used for every method.
These are pilot results, not population estimates.

| Domain | Method/output | Source mean Dice | TTA mean Dice | Delta |
|---|---|---:|---:|---:|
| PED | dense TENT online | 0.233066499 | 0.235130856 | +0.002064357 |
| PED | dense TENT post | 0.233066499 | 0.236728181 | +0.003661682 |
| PED | dense SAR online | 0.233066499 | 0.236435056 | +0.003368556 |
| PED | dense SAR post | 0.233066499 | 0.249223126 | +0.016156627 |
| PED | dense CoTTA online | 0.233066499 | 0.232648770 | -0.000417729 |
| PED | dense CoTTA post | 0.233066499 | 0.233482172 | +0.000415673 |
| SSA | dense TENT online | 0.549649060 | 0.550362329 | +0.000713269 |
| SSA | dense TENT post | 0.549649060 | 0.550960879 | +0.001311819 |
| SSA | dense SAR online | 0.549649060 | 0.550384561 | +0.000735501 |
| SSA | dense SAR post | 0.549649060 | 0.552205145 | +0.002556086 |
| SSA | dense CoTTA online | 0.549649060 | 0.550842285 | +0.001193225 |
| SSA | dense CoTTA post | 0.549649060 | 0.549388369 | -0.000260691 |

The fixes make TENT and SAR genuinely image-dependent and positive on this
small fixed pilot. CoTTA remains domain-dependent: its online teacher helps SSA
but not PED. A larger frozen-protocol cohort is required before claiming a
general improvement.

## Frozen SAR protocol enlarged to ten cases per domain

The dense SAR configuration was kept unchanged and evaluated on the first ten
cases of the same seed-1337 shuffled order. This set includes the initial three
pilot cases plus seven additional cases, so it is an enlarged check rather than
an independent holdout.

| Domain | Output | Source mean Dice | SAR mean Dice | Delta | Improved / tied / degraded | Worst case delta |
|---|---|---:|---:|---:|---:|---:|
| PED | online | 0.504206705 | 0.508562249 | +0.004355544 | 9 / 1 / 0 | 0.000000000 |
| PED | post | 0.504206705 | 0.519870291 | +0.015663586 | 10 / 0 / 0 | +0.000353456 |
| SSA | online | 0.759205562 | 0.759554148 | +0.000348586 | 7 / 1 / 2 | -0.005483687 |
| SSA | post | 0.759205562 | 0.759472603 | +0.000267041 | 7 / 1 / 2 | -0.033304691 |

PED's online ET/TC/WT means all increased, by approximately +0.000648,
+0.003535, and +0.008883. The much larger post-adaptation mean improvement is
not uniformly desirable: post ET fell from 0.540015 to 0.441443 while TC and WT
rose. SSA online ET rose by about +0.002480, but TC and WT fell by about
0.000613 and 0.000822. Paired mean-delta 95% t intervals remain wide at this
sample size: PED online `[-0.001484, 0.010195]` and SSA online
`[-0.001438, 0.002135]`.

The safest current candidate is therefore **dense SAR online**, not the final
post-adaptation model. It shows a consistent PED direction and limits the worst
SSA loss compared with post adaptation. The full 260-case PED and 60-case SSA
cohorts are still required for a population-level claim.

## WT-specific follow-up: why increasing the WT signal is not enough

The apparent remaining WT problem was tested with a label-free, volume-level
diagnostic. For each case, all 18 sliding-window gradients were accumulated
before applying one Adam step; the model was reset to the source checkpoint for
the next case. The objective was WT-only entropy with equal predicted
foreground/background weighting, and the learning rate was selected from ten
source validation cases (`1e-3`, not from PED labels).

On the first 20 cases of the already persisted seed-1337 PED order, this
configuration produced:

| Quantity | Result |
|---|---:|
| WT delta, mean | `-0.006046493` |
| WT delta, standard deviation | `0.007080531` |
| Cases improved / degraded | `4 / 16` |
| Worst WT delta | `-0.026029617` |
| Mean-Dice delta | `-0.001046513` |

The result is not a threshold artifact. Recomputing WT Dice after the update at
thresholds from `0.05` through `0.70` gave negative mean deltas at every
threshold (`-0.00492` to `-0.00608`). The same cases under the existing
patch-wise dense TENT had a smaller but still negative WT delta (`-0.002869`)
online and `-0.005441` post-update. Thus, giving WT a larger entropy weight
does not recover the missing WT region; it usually moves the WT probability in
the wrong spatial direction.

This separates two effects that had been mixed together:

1. The original implementation did have correctable protocol defects (large
   weight decay, background-dominated entropy, patch-wise state drift, and a
   SAR threshold on the wrong loss scale). Fixing those defects makes the
   update nonzero and image-dependent.
2. The remaining PED WT failures are information failures. In several cases the
   source output has no spatial overlap with the true WT, or only a very weak
   signal. Entropy minimization can make an already wrong region more confident,
   but it has no target-free term that tells it where the missing lesion is.
   Cases with source WT around `0.1--0.3` were the most damaging in this pilot;
   a high entropy value alone did not identify recoverable cases. The label-free
   objective is therefore not a reliable WT repair mechanism for this cohort.

The next viable direction is a conservative, label-free gate based on
augmentation agreement or a source-prediction anchor, with adaptation skipped
when the proposed update changes the prediction without improving agreement.
That should be evaluated as a separate method. A genuine reproduction of the
paper's WT behavior would additionally require retraining a source model with
the paper's mutually exclusive four-class softmax, BatchNorm, resized-volume
protocol, and matching cohort; those properties cannot be recovered by
post-hoc changes to this three-channel sigmoid/InstanceNorm checkpoint.

The first agreement-gate screen does not yet justify deploying such a gate. On
the same 20-case PED pilot, the WT-only update delta had near-zero linear
correlation with source WT entropy (`-0.127`), augmentation MAE (`-0.025`),
half-threshold crossing rate (`+0.070`), or the first-patch balanced gradient
norm (`+0.011`). Restricting updates to cases with lower augmentation MAE or
lower uncertain-WT fraction still left a negative mean WT change. These are
useful rejection signals for extreme instability, but not a demonstrated WT
improvement criterion.

The practical fix at this point is therefore a **no-regression default**:
retain the source prediction unless a separately validated, label-free method
shows a measurable gain on a frozen validation protocol. The current WT-only
variant is retained as a diagnostic artifact, not enabled in the full evaluator.

## SSA WT-only external check

The same source-selected WT-only volume-step protocol was run on the first 20
cases in the fixed SSA manifest order. Adaptation still used no target labels;
labels were read only for the post-hoc reports. This cohort is an external
check, not a tuned holdout, and its source WT is already high (`0.949170`,
compared with `0.899710` over all 60 SSA cases).

| Quantity | Result |
|---|---:|
| Source WT mean | `0.949169692` |
| WT-only post-step mean | `0.950204644` |
| WT delta mean | `+0.001034951` |
| WT delta standard deviation | `0.002224419` |
| Paired 95% normal interval | `[+0.000060,+0.002010]` |
| Cases improved / degraded | `14 / 6` |
| Worst / best WT delta | `-0.002611339 / +0.006162047` |
| Mean-Dice delta | `+0.000909340` |

The direction depends on the reporting threshold: mean WT delta was
`-0.001613` at threshold `0.05`, `-0.000953` at `0.10`, `-0.000157` at
`0.20`, `+0.000313` at `0.30`, and `+0.001760` at `0.70`. Therefore this
small positive result at the default `0.5` threshold is not evidence that
entropy found missing WT structure; it is consistent with moving already
near-threshold probabilities and is not stable enough to enable by default.
The contrast with the PED 20-case result (`-0.006046493`) confirms that the
remaining WT behavior is domain- and cohort-dependent rather than a universal
BatchNorm or learning-rate failure.

## SSA WT baseline audit: high values are real, but `0.949` is a selected subset

The `0.949170` WT source result above is **not** the full SSA baseline. It is
the first 20 cases in raw manifest order used by the small WT-only diagnostic.
The correct source-only, all-60-case SSA aggregate is:

| Metric | All 60 SSA cases | First diagnostic 20 | Remaining 40 |
|---|---:|---:|---:|
| WT mean Dice | `0.899710` | `0.949170` | `0.874981` |
| WT median Dice | `0.946119` | `0.959633` | -- |
| Mean Dice (ET/TC/WT) | `0.809230` | -- | -- |

Thus the diagnostic subset is easier, but the high WT values are not a single
case artifact: 46/60 cases have WT Dice at least `0.90`, and 32/60 have WT
Dice at least `0.94`. Conversely, the full cohort contains meaningful failures,
including one all-zero prediction and three WT scores below `0.60`.

Three independent checks were made before interpreting this result:

1. The 20 source records in `ssa/source_cases.jsonl` and the FP32 diagnostic
   were compared case by case. Their largest reported regional-metric
   difference was `0.000150` (the old source export used AMP); independently
   recomputing every regional Dice from stored TP, predicted-voxel, and
   target-voxel counts matched within `2.91e-8`.
2. A fresh FP32, TF32-disabled inference of `BraTS-SSA-00046-000`, performed
   without passing its target to the model, yielded ET/TC/WT Dice
   `0.285652 / 0.037616 / 0.957546`. The accompanying overlay shows that the
   source prediction aligns with the large FLAIR-visible WT/edema area while
   severely overpredicting the core. A high WT therefore does not imply a
   high-quality three-region segmentation.
3. The modern SSA schema was verified as `ET=3`, `TC=1 or 3`, and
   `WT=1,2,or 3`; checked image/label affines and shapes agree. A direct
   source/SSA leakage audit found no duplicated image or label. Although 39
   SSA numbers also occur as a BraTS2021 number, none of those matched pairs
   had identical FLAIR data or labels (foreground-label IoU median `0.0149`).
   Across all 60 SSA images against all 1,251 source images, the strongest
   z-normalized low-resolution FLAIR signature correlation was only `0.444`.
   This rules out direct copies or simple intensity re-encodings in the local
   data; anonymized local files alone cannot prove a patient-level provenance
   claim beyond that.

The reproducible QC artifacts are
`outputs/target_eval/qc/ssa_00046_source_regions_fp32.png` and
`outputs/target_eval/qc/ssa_00046_source_regions_fp32.json`. The complete
pure-FP32 audit is in `outputs/target_eval/ssa/source_fp32_audit/`.

## Reproducible artifacts

- Label-free diagnostics: `outputs/target_eval/tta_mechanism_study/signals_*.json`
- Source calibration: `outputs/target_eval/tta_mechanism_study/source5_*`
- Fixed target pilots: `outputs/target_eval/tta_mechanism_study/target_pilot_*`
- Ten-case SAR check:
  `outputs/target_eval/tta_mechanism_study/target_confirm_sar_dense_v2_case_reset/`
- WT regional-gradient diagnostic and source learning-rate calibration:
  `outputs/target_eval/wt_diagnostic/volume_step_balanced_lr_sweep_source10/`
  and
  `outputs/target_eval/wt_diagnostic/volume_step_wt_only_source_selected_ped20/`
  (the latter is supplemented by the short-batch directories with suffixes
  `_08_10`, `_11_13`, `_14_16`, and `_17_20`).
- SSA external WT-only check:
  `outputs/target_eval/wt_diagnostic/volume_step_wt_only_source_selected_ssa20/`
- Evaluators: `src/brats_tta/cli/evaluate_tent_variant.py` and
  `src/brats_tta/cli/evaluate_tegda_baseline.py`
- Dense objectives and adapters: `src/brats_tta/tta/dense_objectives.py` and
  `src/brats_tta/tta/tegda_baselines.py`
