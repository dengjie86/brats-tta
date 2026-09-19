"""TEGDA-code-compatible optimizer/continual ablation, NOT an exact reproduction."""
import argparse
import faulthandler
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.cli.evaluate_tta import METRIC_KEYS, _validate_run_settings, _write_json
from brats_tta.cli.extract_brain_masks import _sha256, validate_destinations
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.continual_state import load_state, save_state, source_affine_parameters
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.tent import TentAdapter

LOGGER = logging.getLogger(__name__)
METHODS = ('source', 'tent_online', 'tent_post')


def summarize(records, expected):
    return dict(completed_cases=len(records), expected_cases=expected,
        complete=len(records) == expected,
        methods={method: dict(metrics_mean={key: float(np.mean([
            r[method][key] for r in records])) for key in METRIC_KEYS},
            delta_mean_vs_source=float(np.mean([
                r[method]['dice_mean']-r['source']['dice_mean'] for r in records])))
            for method in METHODS} if records else {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    configure_logging()
    faulthandler.enable()
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    dataset = BraTSDataset(args.manifest, training=False)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if count < 1:
        raise ValueError('Empty cohort or invalid limit')
    # Fixed shuffled order, persisted verbatim: restart must not reshuffle/skip state.
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(1337)).tolist()[:count]
    root = Path(args.output_dir).resolve()
    validate_destinations(dataset.manifest, root, root/'summary.json')
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(protocol='tegda_compatible_continual_v1',
        checkpoint_sha256=_sha256(Path(args.checkpoint)),
        manifest_sha256=_sha256(Path(args.manifest)),
        case_order=[dataset.cases[i]['id'] for i in order], seed=1337,
        precision='fp32', tf32=False, optimizer='Adam', learning_rate=1e-5,
        weight_decay=.9, betas=[.9,.999], entropy='all_voxel_bernoulli_div_ln2',
        reset='domain_start_only', adaptation='one_sweep_one_update_per_patch',
        patch_size=[128,128,128], overlap=.5, sw_batch_size=1, gaussian_weighting=True,
        threshold=.5, source='recomputed_frozen_model', predictions=list(METHODS),
        labels_used_for_adaptation=False, exact_tegda_reproduction=False,
        retained_differences=['our_source_checkpoint_and_IN_architecture',
            'three_sigmoid_regions_not_four_softmax_classes',
            'source_compatible_preprocessing_and_native_resolution_sliding_windows',
            'our_cohorts_and_explicit_fixed_shuffle_not_identical_author_loader_order'])
    state_path = root/'continual_state.pt'
    _validate_run_settings(root/'run_settings.json', settings,
        has_records=state_path.exists(), overwrite=False)
    records = []

    def progress(stage, **details):
        _write_json(root/'progress.json', dict(stage=stage, pid=os.getpid(),
            updated_at_unix=time.time(), completed_cases=len(records), expected_cases=count, **details))
        faulthandler.dump_traceback_later(240, repeat=True)

    def report():
        _write_json(root/'summary.json', {**summarize(records, count), 'settings':settings})
        _write_json(root/'cases.json', records)

    try:
        progress('loading_model')
        model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        del checkpoint
        adapter = TentAdapter(model, learning_rate=1e-5, weight_decay=.9,
            steps=1, use_amp=False, normalize_entropy=True)
        if state_path.exists():
            records = load_state(state_path, adapter, settings)
            if [r['id'] for r in records] != settings['case_order'][:len(records)]:
                raise ValueError('Committed records do not match continual order')
        report()
        inference = dict(patch_size=(128,128,128), overlap=.5, sw_batch_size=1)
        for sequence_index in range(len(records), count):
            index = order[sequence_index]
            case_id = dataset.cases[index]['id']
            progress('loading_case', case_id=case_id)
            started = time.perf_counter()
            sample = dataset[index]
            image = sample['image'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0)
            if target.shape[1] != 3:
                raise ValueError('Requires ET/TC/WT labels for evaluation only')
            torch.cuda.reset_peak_memory_stats(device)
            record = dict(id=case_id, sequence_index=sequence_index)

            def callback(phase):
                def notify(done, total):
                    torch.cuda.synchronize(device)
                    progress(phase, case_id=case_id, patch_done=done, patch_total=total)
                return notify

            with source_affine_parameters(adapter) as source_model:
                logits = sliding_window_logits(source_model, image, **inference, amp=False,
                    progress_callback=callback('source_predict'))
            record['source'] = compute_region_metrics(logits.cpu(), target, threshold=.5)
            del logits
            torch.cuda.empty_cache()
            logits, adaptation = sliding_window_tent_logits(model, adapter, image, **inference,
                progress_callback=callback('tent_adapt_online'))
            record['tent_online'] = compute_region_metrics(logits.cpu(), target, threshold=.5)
            del logits
            if not all(torch.isfinite(p).all().item() for p in adapter.parameters):
                raise FloatingPointError('Nonfinite adapted affine parameters')
            logits = sliding_window_logits(model, image, **inference, amp=False,
                progress_callback=callback('tent_post_predict'))
            record['tent_post'] = compute_region_metrics(logits.cpu(), target, threshold=.5)
            del logits
            record.update(adaptation=adaptation, seconds=time.perf_counter()-started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
                affine_delta_l2=sum((p.detach()-s).square().sum().item()
                    for p,s in zip(adapter.parameters, adapter._source_parameters))**.5)
            # This single atomic file is authoritative. On interruption replay only
            # the uncommitted patient using its predecessor's parameters AND Adam.
            progress('committing', case_id=case_id)
            save_state(state_path, adapter, settings, records+[record])
            records.append(record)
            report()
            progress('case_complete', case_id=case_id)
            LOGGER.info('%d/%d %s source=%.5f online=%.5f post=%.5f seconds=%.1f',
                len(records), count, case_id, record['source']['dice_mean'],
                record['tent_online']['dice_mean'], record['tent_post']['dice_mean'], record['seconds'])
            del image, target, sample
            torch.cuda.empty_cache()
        progress('complete')
    except BaseException as exc:
        progress('failed', error=str(exc))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
