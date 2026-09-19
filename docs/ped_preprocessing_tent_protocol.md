# PED preprocessing and TENT step protocol (2026-09-06)

## Correcting the input mismatch

BraTS-PEDs 2024 distributed defaced images, not the skull-stripped images used by
the BraTS 2021 source model. Normalizing every nonzero voxel therefore includes
extracranial tissues. This is an input-pipeline mismatch, not evidence for or
against TENT. See the [organizer clarification](https://www.synapse.org/Synapse:syn53708249/discussion/threadId=11140).

Use the official [HD-BET](https://github.com/MIC-DKFZ/HD-BET) pretrained brain
extractor (software 2.0.1, release_2.0.0 weights, nnUNetv2 2.6.2). The original
method is published in Human Brain Mapping (2019). This is an additional external
pretrained preprocessing model, not part of the source segmentation checkpoint;
the exact v2 release must be recorded rather than described as identical to the
2019 network. HD-BET's adult-data origin means pediatric masks still require QC.

- Predict a mask from T1c alone, with mirroring enabled; no tumor annotation enters extraction.
- Validate native shape/affine, binary values, and a broad image-only size sanity check.
- Apply the same mask to all four co-registered modalities **before** nonzero z-score.
- Keep tumor labels, voxel grid, region mapping and Dice denominator unchanged.
- Do not force segmentation predictions inside the mask or repair masks using tumor labels.
- Original dataset files are read-only; save masks, provenance and manifests under `outputs/`.
- Keep inference/adaptation/accumulation FP32, disable TF32. Do not replace the user's environment.
- SSA is already brain-extracted; retain its original preprocessing.

## What published open-source code actually specifies

| Work | Verified TENT setting | Important difference |
|---|---|---|
| TENT, ICLR 2021 | Adam, LR 1e-3, weight decay 0, 1 step per input batch | Official example is BN + categorical softmax, not our IN + overlapping sigmoid regions |
| TEGDA, MICCAI 2025 | `setup_tent` uses `Tent(...steps=1, episodic=False)` defaults | Brain implementation adapts BatchNorm3d; optimizer code uses LR 1e-5 and weight decay 0.9 |
| PT-TEA, ICCV 2025 | README lists 10 iterations for PT-TEA itself | This is **not evidence** that its TENT baseline used 10 steps |

Primary sources, inspected 2026-09-06:

- [TENT official configuration](https://github.com/DequanWang/tent/blob/master/cfgs/tent.yaml)
- [TENT official forward/update loop](https://github.com/DequanWang/tent/blob/master/tent.py)
- [TEGDA MICCAI paper page](https://papers.miccai.org/miccai-2025/0906-Paper2263.html)
- [TEGDA TENT wrapper](https://github.com/HiLab-git/TEGDA/blob/main/code/sota/tent.py)
- [TEGDA setup and optimizer](https://github.com/HiLab-git/TEGDA/blob/main/code/sota/_tta.py)
- [PT-TEA official repository](https://github.com/Voldemort108X/pttea_seg)

The local earlier PT-TEA reproduction used a controlled 10-step episodic TENT
baseline, but explicitly documented that exact paper baseline hyperparameters
were unavailable. Do not promote that local choice into a published setting.

## Predeclared experiment

1. Full corrected PED: source and **1-step-per-patch-batch TENT**, retaining LR
   1e-3 and per-patient episodic reset from the earlier run. This isolates preprocessing.
2. Fixed sensitivity pilot: first 3 cases in each manifest, all **1 / 5 / 10**
   steps per patch batch, same LR and all other settings. These are diagnostic
   cases, not an unbiased estimate for the complete dataset. Report every setting;
   do not pick the largest target-label Dice for the full-cohort configuration.
3. A typical 240 x 240 x 155 case gives 18 patch batches with 128-cube windows,
   overlap 0.5, sliding-window batch size 1: 18 / 90 / 180 actual optimizer updates
   per patient, respectively. Exact count depends on image geometry.
4. Carry adapted affine parameters across patches, reset both model and optimizer
   before each patient. Stitch the final forward within each patch's step loop,
   as in official TENT (that forward precedes the final optimizer update).
5. Our IN-affine/Bernoulli-entropy implementation is a segmentation-specific
   TENT adaptation, not an exact reproduction of a BN/softmax source model.

`scripts/run_brain_tta_study.py` automates pilot extraction/evaluation, full PED
extraction and full source/TENT evaluation, with per-stage logs, case-level resume,
protocol fingerprints and original input size/mtime checks. Multiple GPU stages
run sequentially. Completed source/1-step SSA results from the previous full FP32
run remain valid; the new SSA pilot is for the step-count comparison.
