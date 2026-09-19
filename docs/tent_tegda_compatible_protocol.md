# TEGDA-compatible continual comparison (not exact reproduction)

Reference: official HiLab-git/TEGDA, `code/sota/_tta.py`, `code/sota/tent.py`,
and `code/test_time_adaptation_3D_online_eval.py`.

This run uses our epoch-200 source checkpoint, not another local project's model.
Adam lr=1e-5, betas=(0.9,0.999), weight_decay=0.9 follow the official code,
not the paper's stated lr=1e-4. Hyperparameters are fixed before observing results.

Pure FP32, TF32 off. One full adaptation sweep per patient, one update per patch.
Affine parameters and Adam momentum persist between patients; each domain starts
from source. Order is a saved seed-1337 permutation (not claimed to reproduce the
author's exact DataLoader order). No labels enter adaptation or select a model.

Retained source-compatible differences: IN 3D U-Net architecture and checkpoint,
three overlapping sigmoid regions rather than four mutually exclusive softmax
classes, native-resolution sliding windows (128^3, overlap 0.5, Gaussian),
nonzero z-score preprocessing and corrected PED brain masks. Bernoulli entropy
is divided by ln(2), giving unit maximum per region; this is not categorical entropy.
PED=260, SSA=60; these are not the paper's target cohort.

For each patient, recompute frozen Source, then record online pre-update patch
predictions and a separate fixed-model post-sweep prediction. Both are reported,
not selected by target Dice. ET/TC/WT means use the exact same completed patients.

`continual_state.pt` atomically commits completed records, affine parameters,
Adam state, scaler and RNG. Restart restores that state and repeats only an
uncommitted patient. JSON reports are derived and may lag the authoritative
checkpoint after an interruption. A 300-second progress watchdog allows two
stall restarts. Ordinary errors stop the run. Original datasets are read-only;
size/mtime snapshots check all original files and derived masks.

Output: `outputs/target_eval/tent_tegda_compatible_fp32/{ped,ssa}`.
