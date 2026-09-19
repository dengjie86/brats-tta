"""Resumable, read-only-input FP32 PED preprocessing and TENT sensitivity study.

Run from the repository root. All step counts are declared before evaluation;
the pilot is the first three manifest cases, not a label-selected subset.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def snapshot(manifests: list[Path]) -> dict:
    records = {}
    for manifest in manifests:
        for case in json.loads(manifest.read_text(encoding='utf-8'))['cases']:
            for raw in [*case['images'].values(), *([case['label']] if case.get('label') else [])]:
                path = Path(raw).resolve()
                stat = path.stat()
                records[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ped-manifest', 'ssa-manifest', 'checkpoint', 'weights', 'output-dir', 'mask-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--brain-python', required=True)
    parser.add_argument('--eval-python', default=sys.executable)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifests = [args.ped_manifest.resolve(), args.ssa_manifest.resolve()]
    # Reuse the extractor's protection for every destination before writing.
    from brats_tta.cli.extract_brain_masks import validate_destinations
    from brats_tta.utils.process_watchdog import run_with_progress_watchdog
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding='utf-8'))
        validate_destinations(data, root, root/'ped_full.json')
        validate_destinations(data, args.mask_dir, root/'ped_pilot.json')
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(ped_manifest=str(manifests[0]), ssa_manifest=str(manifests[1]),
        checkpoint=str(args.checkpoint.resolve()), weights=str(args.weights.resolve()),
        mask_dir=str(args.mask_dir.resolve()), pilot_selection='first 3 cases in each manifest',
        steps_per_patch_batch=[1, 5, 10], learning_rate=0.001, episodic_per_patient=True,
        full_ped_methods=['source', 'tent_steps_1'], precision='fp32', tf32=False,
        note='All pilot settings reported; target Dice does not select full-cohort hyperparameters.')
    protocol_path = root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Study protocol changed; choose a new output directory')
    save(protocol_path, protocol)
    before_path = root/'input_snapshot.json'
    current = snapshot(manifests)
    if before_path.exists():
        if json.loads(before_path.read_text()) != current:
            raise ValueError('Original dataset identities changed since study began')
    else:
        save(before_path, current)
    status = dict(status='running', started_at=datetime.now(timezone.utc).isoformat())
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),
                       OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    def run(stage: str, command: list[str]) -> None:
        status.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        save(root/'status.json', status)
        print(f'\nSTAGE {stage}', flush=True)
        with (root/f'{stage}.log').open('a', encoding='utf-8') as log:
            log.write('\n' + json.dumps(command) + '\n')
            log.flush()
            if 'brats_tta.cli.evaluate_tta' in command:
                # All study evaluations reset per patient and can safely resume.
                run_with_progress_watchdog(command, env=environment, log=log,
                    progress_path=root/stage/'progress.json', timeout=300, max_restarts=2)
            else:
                subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    def extract(output: Path, limit: bool = False) -> None:
        command = [args.brain_python, '-m', 'brats_tta.cli.extract_brain_masks',
            '--manifest', str(manifests[0]), '--output-root', str(args.mask_dir.resolve()),
            '--output-manifest', str(output), '--weights', str(args.weights.resolve())]
        run('extract_pilot' if limit else 'extract_full', command + (['--limit','3'] if limit else []))
    def evaluate(domain: str, manifest: Path, steps: int, pilot: bool) -> None:
        name = f'{domain}_{"pilot" if pilot else "full"}_steps{steps}'
        command = [args.eval_python, '-m', 'brats_tta.cli.evaluate_tta',
            '--checkpoint', str(args.checkpoint.resolve()), '--manifest', str(manifest),
            '--output-dir', str(root/name), '--methods', *(['source','tent'] if steps == 1 else ['tent']),
            '--no-amp', '--device', 'cuda', '--patch-size','128','128','128',
            '--overlap','0.5','--sw-batch-size','1','--threshold','0.5',
            '--stall-trace-seconds','240',
            '--tent-lr','0.001','--tent-steps', str(steps)]
        run(name, command + (['--limit','3'] if pilot else []))
        report()
    def report() -> None:
        rows = []
        for summary in sorted(root.glob('*/*_summary.json')):
            payload = json.loads(summary.read_text())
            rows.append(dict(experiment=summary.parent.name, method=payload['method'],
                cases=payload['case_count'], complete=payload['complete'], **payload['metrics_mean']))
        save(root/'comparison.json', {'protocol':protocol, 'results':rows})
        lines = ['# FP32 brain-extraction / TENT study', '',
            'Pilot = first 3 cases/domain, descriptive only. Full PED uses predeclared 1-step TENT.',
            'SSA preprocessing is unchanged (already brain-extracted). Original labels remain unchanged.', '',
            '| Experiment | Method | N | ET | TC | WT | Mean |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in rows:
            lines.append(f"| {row['experiment']} | {row['method']} | {row['cases']} | " +
                ' | '.join(f"{row[k]:.6f}" for k in ('dice_ET','dice_TC','dice_WT','dice_mean')) + ' |')
        (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    try:
        pilot_manifest = root/'ped_pilot.json'
        extract(pilot_manifest, limit=True)
        for steps in (1, 5, 10):
            evaluate('ped', pilot_manifest, steps, True)
            evaluate('ssa', manifests[1], steps, True)
        full_manifest = root/'ped_full.json'
        extract(full_manifest)
        evaluate('ped', full_manifest, 1, False)
        after = snapshot(manifests)
        integrity = dict(files_checked=len(current), unchanged=after == current,
                         check='file size and mtime_ns, not a full content hash')
        save(root/'input_integrity.json', integrity)
        if not integrity['unchanged']:
            raise RuntimeError('Original dataset identities changed during the study')
        status.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        save(root/'status.json', status)
        report()


if __name__ == '__main__':
    main()
