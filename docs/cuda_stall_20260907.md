# PED TENT CUDA stall diagnosis and recovery (2026-09-07)

## Observed failure

The full PED source evaluation finished (260 cases; mean Dice 0.5787945025).
The TENT worker saved case 1 at 2026-09-06 23:49:10 local time and then stopped
making case progress. At 02:45 on September 7, both supervisor and worker still
existed. A `running` state and 100% GPU utilization were therefore misleading.

Read-only `py-spy` snapshots of worker PID 40484 repeatedly located the main
thread at `TentAdapter.predict_and_adapt`, `loss.detach().item()`. The native
stack showed `c10::cuda::memcpy_and_sync` -> `cudaStreamSynchronize` ->
`cuStreamSynchronize` / `cuStreamQuery` in the CUDA runtime/driver. CUDA launches
are asynchronous, so this identifies the blocking synchronization point, not
the precise previously queued kernel responsible for the failure.

The process was not waiting on NIfTI loading or Dice computation. Task-specific
GPU memory counters were about 5.91 GB dedicated and 0.17 GB shared; this does
not establish memory fallback as the cause. A system-event query did not produce
supporting sleep/display-reset events. The exact trigger (runtime context,
driver, or a particular kernel) remains unproven.

## Recovery verified on actual data

Only the identified study supervisor (46064) and evaluation worker (40484) were
terminated after checking their command lines. No other applications, original
datasets, masks, or saved evaluation records were deleted or changed.

A fresh worker evaluated the same first three PED cases with unchanged source
weights, native geometry, patch size, step count, thresholds and pure FP32:

| Case | Mean Dice | GPU inference/adaptation seconds |
|---|---:|---:|
| BraTS-PED-00001-000 | 0.5133972 | 21.5 |
| BraTS-PED-00002-000 | 0.6172860 | 21.1 |
| BraTS-PED-00003-000 | 0.8890343 | 21.3 |

The metrics match the earlier 3-case pilot. This verifies recovery of the
previously blocked case without changing the adaptation algorithm; it does not
prove the underlying CUDA/driver issue can never recur.

## Operational fixes

- Record `progress.json` with PID, current case, stage and completed patch count.
  Progress is written after actual computation, not a periodic liveness timer.
- Log patch 1, 6, 12 and 18 for the usual 18-patch case.
- Dump Python stacks after 240 seconds without progress in the study worker.
- A separate supervisor kills only its own worker after 300 seconds with no
  progress and restarts it at most twice, retaining completed episodic records.
  Exhausted retries fail visibly; input/configuration errors are not retried.
- Skip already-completed IDs before loading their images when resuming.
- Preserve FP32/TF32-off settings and all existing experiment fingerprints.

The watchdog has automated tests for healthy progress, bounded timeout/retry,
preservation of completed work and non-retry of ordinary failures. The patch
callback has an optimizer-count/progress test. Full-case resumption is verified
separately in the live run, not inferred from the process merely existing.
