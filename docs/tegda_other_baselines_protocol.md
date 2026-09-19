# Other TEGDA comparison baselines on the project source model

This study uses the project's own source checkpoint, source-compatible target
preprocessing, native-resolution sliding-window inference, and strict FP32 with
TF32 disabled. It does not use the separate local TEGDA project's checkpoint,
data, or results.

The TEGDA paper compares Source, TENT, SAR, CoTTA, InTEnt, VPTTA, and TEGDA.
Only SAR and CoTTA have update rules that can be ported to this project's source
network without inventing source statistics that do not exist.

## Compatible implementations

- SAR uses SAM-SGD (`lr=0.001`, momentum `0.9`, rho `0.05`), one update per
  sliding-window patch, reliable Bernoulli-entropy filtering with margin
  `0.4*ln(2)`, the released TEGDA recovery threshold `0.02`, and only affine
  InstanceNorm parameters.
- CoTTA uses a student, EMA teacher, and frozen anchor; Adam (`lr=1e-5`,
  weight decay `0.9`), EMA momentum `0.99`, stochastic restoration probability
  `0.1`, and up to 32 augmentation teacher forwards when anchor confidence is
  below `0.9`. Its categorical consistency is replaced by independent
  Bernoulli cross-entropy. Shape-preserving, invertible 3D augmentations retain
  the source model's z-score input convention instead of applying the released
  code's `[0,1]` TorchIO pipeline.

Both are compatible comparisons, not exact reproductions. PED and SSA start
independently from the source model. Target labels are used only after each
prediction for evaluation.

## Methods that are not defined for this source network

- InTEnt constructs an ensemble by interpolating source and target BatchNorm
  statistics. The source model uses `InstanceNorm3d(track_running_stats=False)`,
  so there are no source running statistics to interpolate. All nominal ensemble
  members collapse to the same per-instance normalization already used by Source.
- VPTTA optimizes its visual prompt by aligning target features to stored source
  BatchNorm statistics. With zero BatchNorm/AdaBN layers, the released objective
  is undefined. Running it would require changing/retraining the source model or
  inventing a new source-statistics objective.

These incompatibilities are architectural, not runtime errors, and should be
reported rather than silently replacing the methods with different algorithms.

## FP32 pilot results (2026-09-08)

The first three cases in the fixed seed-1337 continual order were evaluated in
each target domain. These are mechanism/runtime pilots, not full-cohort
estimates; in particular, the three-case SSA subset contains one case whose
Source Dice is zero.

| Domain | Method | Source mean | Online mean | Post mean | Online delta | Post delta |
|---|---|---:|---:|---:|---:|---:|
| PED | SAR | 0.233066499 | 0.233066499 | 0.233066499 | 0 | 0 |
| PED | CoTTA | 0.233066499 | 0.233078410 | 0.233077059 | +0.000011911 | +0.000010560 |
| SSA | SAR | 0.549649060 | 0.549649060 | 0.549649060 | 0 | 0 |
| SSA | CoTTA | 0.549649060 | 0.549676716 | 0.549675326 | +0.000027657 | +0.000026266 |

SAR performed 18 SAM updates per case, but every update triggered the released
`ema < 0.02` recovery rule. All 108 attempted patch updates across the PED and
SSA pilots were therefore restored exactly to the source affine parameters;
the final parameter delta was zero for every pilot case. Since the unchanged
method demonstrably reduces to Source on this cross-domain pilot, a full run
was not launched without first defining a preregistered recovery-rule ablation.

CoTTA updated all student parameters and retained a nonzero continual parameter
delta, but its augmentation gate never opened in any of the 108 patch updates.
Mean anchor confidence was above the `0.9` threshold because dense independent
Bernoulli outputs are dominated by confident background voxels. The small
three-case deltas show that the port executes and is numerically stable, but do
not establish a population-level benefit. A full CoTTA run was not launched
automatically because it would take many GPU-hours while the defining
augmentation branch is inactive on this architecture.

Machine-readable pilot outputs are under
`outputs/target_eval/tegda_other_baselines_pilot_fp32/`.
