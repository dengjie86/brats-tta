"""Evaluate source plus post-round TENT Dice, with read-only target labels."""
from __future__ import annotations

import argparse
import faulthandler
import json
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
from brats_tta.data.preprocessing import _safe_case_id
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.rounds import iter_tent_round_predictions
from brats_tta.tta.tent import TentAdapter
from brats_tta.utils.atomic_io import atomic_write_text

LOGGER = logging.getLogger(__name__)


def summarize(records: list[dict], rounds: int, expected: int) -> dict:
    paired = [r for r in records if r['complete'] and len(r['rounds']) == rounds + 1]
    result = {}
    for step in range(rounds + 1):
        available = [r['rounds'][step] for r in records if len(r['rounds']) > step]
        common = [r['rounds'][step] for r in paired]
        result[str(step)] = {
            'available_cases': len(available),
            'available_metrics_mean': {k: float(np.mean([r[k] for r in available]))
                                       for k in METRIC_KEYS} if available else {},
            'paired_cases': len(common),
            'paired_metrics_mean': {k: float(np.mean([r[k] for r in common]))
                                    for k in METRIC_KEYS} if common else {},
            'paired_delta_mean_vs_source': float(np.mean([
                r['rounds'][step]['dice_mean'] - r['rounds'][0]['dice_mean']
                for r in paired])) if paired else None,
        }
    return dict(complete=len(paired) == expected, completed_cases=len(paired),
                expected_cases=expected, rounds=result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--tent-lr', type=float, default=.001)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.rounds < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('rounds and limit must be positive')
    configure_logging()
    faulthandler.enable()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    dataset = BraTSDataset(args.manifest, training=False)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if not count:
        raise ValueError('empty cohort')
    root = Path(args.output_dir).resolve()
    validate_destinations(dataset.manifest, root, root/'summary.json')
    root.mkdir(parents=True, exist_ok=True)
    (root/'cases').mkdir(exist_ok=True)
    settings = dict(protocol='patient_rounds_post_update_v1', rounds=args.rounds,
        checkpoint_sha256=_sha256(Path(args.checkpoint)), manifest_sha256=_sha256(Path(args.manifest)),
        patch_size=[128,128,128], overlap=.5, sw_batch_size=1, threshold=.5,
        precision='fp32', tf32=False, optimizer='Adam', learning_rate=args.tent_lr,
        weight_decay=0, reset='per_patient_only', updates_per_patch_batch=1,
        entropy='all_voxel_bernoulli', labels_used_for_adaptation=False,
        prediction='full_volume_no_grad_after_each_complete_round', device=str(device))
    _validate_run_settings(root/'run_settings.json', settings,
        has_records=any((root/'cases').glob('*.json')), overwrite=False)
    records = {}
    for case in dataset.cases[:count]:
        path = root/'cases'/f"{_safe_case_id(case['id'])}.json"
        if path.exists():
            records[case['id']] = json.loads(path.read_text())

    def progress(stage: str, **details) -> None:
        _write_json(root/'progress.json', dict(pid=os.getpid(), stage=stage,
            updated_at_unix=time.time(), expected_cases=count,
            completed_cases=sum(r['complete'] for r in records.values()), **details))
        faulthandler.dump_traceback_later(240, repeat=True)

    def report() -> None:
        summary = {**summarize(list(records.values()), args.rounds, count), 'settings':settings}
        _write_json(root/'summary.json', summary)
        lines = ['# Post-round TENT evaluation (FP32)', '',
            'Paired columns use only patients with all rounds complete; labels do not select rounds.', '',
            '| Round | Paired N | ET | TC | WT | Mean | Delta vs source |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for step, row in summary['rounds'].items():
            m = row['paired_metrics_mean']
            if m:
                lines.append(f"| {step} | {row['paired_cases']} | " +
                    ' | '.join(f'{m[k]:.6f}' for k in ('dice_ET','dice_TC','dice_WT','dice_mean')) +
                    f" | {row['paired_delta_mean_vs_source']:+.6f} |")
        atomic_write_text(root/'report.md', '\n'.join(lines)+'\n')

    progress('loading_model')
    try:
        model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        del checkpoint
        adapter = TentAdapter(model, learning_rate=args.tent_lr, steps=1, use_amp=False)
        source_parameters = [p.detach().clone() for p in adapter.parameters]
        for index, case in enumerate(dataset.cases[:count]):
            case_id = case['id']
            if case_id in records and records[case_id]['complete']:
                continue
            progress('loading_case', case_id=case_id)
            sample = dataset[index]
            image = sample['image'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0)
            if target.shape[1] != 3:
                raise ValueError('evaluation requires three region labels')
            # Incomplete patients restart from Source because Adam state is not saved.
            record = dict(id=case_id, complete=False, rounds=[])
            records[case_id] = record
            path = root/'cases'/f'{_safe_case_id(case_id)}.json'
            _write_json(path, record)
            def patch_progress(step, phase, done, total):
                progress(phase, case_id=case_id, round=step, patch_done=done, patch_total=total)
                if done == total:
                    LOGGER.info('%s round=%d %s %d/%d', case_id, step, phase, done, total)

            start = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step, logits, adaptation in iter_tent_round_predictions(adapter, image,
                rounds=args.rounds, patch_size=(128,128,128), progress_callback=patch_progress):
                progress('metrics', case_id=case_id, round=step)
                scores = compute_region_metrics(logits.cpu(), target, threshold=.5)
                delta = sum((p.detach()-s).square().sum().item()
                            for p,s in zip(adapter.parameters, source_parameters)) ** .5
                record['rounds'].append(dict(round=step, **scores, **adaptation,
                    affine_delta_l2_vs_source=delta, cumulative_seconds=time.perf_counter()-start,
                    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device)
                        if device.type == 'cuda' else 0))
                record['complete'] = step == args.rounds
                _write_json(path, record)
                report()
                LOGGER.info('%d/%d %s round=%d ET=%.4f TC=%.4f WT=%.4f mean=%.4f updates=%d',
                    index+1, count, case_id, step, scores['dice_ET'], scores['dice_TC'],
                    scores['dice_WT'], scores['dice_mean'], adaptation['cumulative_updates'])
                del logits
            del image, target, sample
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        report()
        progress('complete')
    except BaseException as exc:
        progress('failed', error=str(exc))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
