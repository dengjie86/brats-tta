# Five patient-wise TENT rounds (requested 2026-09-07)

## Definition, fixed before full evaluation

The user explicitly selected **five whole-patient adaptation rounds**, not five
updates on each patch before moving to the next patch. Evaluate all 260 corrected
PED cases and all 60 SSA cases. All five settings/results will be reported; target
labels never select a round, learning rate, stopping point or mask.

For each patient:

1. Restore source InstanceNorm affine parameters and clear Adam state once.
2. Round 0: no-grad sliding-window source prediction and ET/TC/WT/mean Dice.
3. Round 1: visit every patch batch, doing one Adam entropy update each. Discard
   online stitched predictions. Use the resulting **fixed model** for a separate,
   no-grad sliding-window prediction of the whole patient, then compute Dice.
4. Repeat that same adaptation/prediction sequence for rounds 2–5, preserving
   model parameters and optimizer momentum between rounds.
5. Restore the source state before the next patient.

With 18 patch batches, rounds 0–5 correspond to 0/18/36/54/72/90 optimizer
updates. Exact counts are recorded rather than assumed. Prediction sweeps do not
perform updates. A round is not a single full-volume gradient step.

## Fixed settings

- FP32 throughout, no autocast, TF32 disabled.
- Same source checkpoint, preprocessed PED manifest, and original SSA manifest.
- InstanceNorm affine-only TENT; independent Bernoulli entropy for overlapping
  ET/TC/WT sigmoid outputs, averaged over all patch voxels/channels.
- Adam LR 0.001, betas (0.9, 0.999), weight decay 0; one update per patch batch.
- 128-cube patch, overlap 0.5, patch batch size 1, Gaussian logit blending,
  probability threshold 0.5. Region mappings/ground truth remain unchanged.

This is a new **post-round evaluation protocol**, not a claim that the previous
official-style online loop had the optimizer step in the wrong order. The
[official TENT loop](https://github.com/DequanWang/tent/blob/master/tent.py) returns
the forward output obtained before its final update. In a sliding-window setup,
that yields a composite prediction from multiple parameter states. It is not
the same observable as a prediction made with the final adapted model.

The existing source model uses InstanceNorm and overlapping regions, so this
remains a segmentation-specific TENT variant, not an exact BN/softmax replication.
Independent tests confirm five parameter updates match a manually implemented
Bernoulli entropy/Adam reference. More adaptation is not assumed to improve Dice.

## Outputs and interruption handling

`brats-evaluate-tta-rounds` writes a case JSON after each round, including region
Dice, mean Dice, cumulative update count, affine change, memory and elapsed time.
`summary.json` includes both available per-round samples and a paired summary
restricted to patients completing every round. `report.md` uses the latter,
avoiding comparisons across different partial cohorts.

A complete patient is skipped on resume. An incomplete patient is recomputed
from round 0, because its adapted model and Adam state are not checkpointed;
its partial case file is replaced as rounds complete. No patient is dropped.
Original data and existing studies stay untouched in separate output directories.

`scripts/run_tent_round_study.py` runs PED then SSA with a separate supervisor.
Five minutes without real patch progress triggers a worker restart (at most two
retries per domain); repeated stalls fail visibly. Source/round parameters are
fingerprinted and input sizes/modification times checked across the study.

Example (from repository root, with the project environment active):

```powershell
python scripts/run_tent_round_study.py `
  --checkpoint outputs/kaggle/brats-2021-3d-u-net-dual-t4-source-resume-4/brats2021_source_ddp/run/checkpoints/best.pt `
  --ped-manifest outputs/target_eval/rerun_brain_fp32/ped_full.json `
  --ssa-manifest outputs/target_eval/manifests/ssa_raw.json `
  --output-dir outputs/target_eval/tent_patient_rounds_fp32
```
