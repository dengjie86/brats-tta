"""Run PED then SSA five-round FP32 study with bounded stall recovery."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from brats_tta.cli.evaluate_tta import _validate_run_settings, _write_json
from brats_tta.cli.extract_brain_masks import _sha256, validate_destinations
from brats_tta.utils.process_watchdog import run_with_progress_watchdog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ped-manifest','ssa-manifest','checkpoint','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifests = {'ped':args.ped_manifest.resolve(), 'ssa':args.ssa_manifest.resolve()}
    inputs = []
    for path in manifests.values():
        manifest = json.loads(path.read_text())
        validate_destinations(manifest, root, root/'status.json')
        for case in manifest['cases']:
            inputs.extend([*case['images'].values(), case['label']])
            if case.get('brain_mask'):
                inputs.append(case['brain_mask'])
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(rounds=5, precision='fp32', tf32=False, learning_rate=.001,
        checkpoint_sha256=_sha256(args.checkpoint),
        manifests={k:dict(path=str(v),sha256=_sha256(v)) for k,v in manifests.items()},
        step_definition='one complete adaptation sweep then fixed-model full-volume prediction',
        source_round=0, report_all_rounds=True, label_based_round_selection=False)
    _validate_run_settings(root/'protocol.json', protocol,
                          has_records=(root/'status.json').exists(), overwrite=False)
    def snapshot():
        return {str(Path(p).resolve()): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
                for p in inputs}
    before = snapshot()
    _validate_run_settings(root/'input_snapshot.json', before, has_records=False, overwrite=False)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),
                       OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    status = dict(status='running', started_at=datetime.now(timezone.utc).isoformat())
    try:
        for domain, manifest in manifests.items():
            status['stage'] = domain
            _write_json(root/'status.json', status)
            command = [sys.executable, '-m','brats_tta.cli.evaluate_tta_rounds',
                '--checkpoint',str(args.checkpoint.resolve()), '--manifest',str(manifest),
                '--output-dir',str(root/domain), '--rounds','5', '--tent-lr','.001']
            with (root/f'{domain}.log').open('a',encoding='utf-8') as log:
                log.write(json.dumps(command)+'\n')
                log.flush()
                run_with_progress_watchdog(command, env=environment, log=log,
                    progress_path=root/domain/'progress.json', timeout=300, max_restarts=2)
        unchanged = snapshot() == before
        _write_json(root/'input_integrity.json',dict(unchanged=unchanged,files_checked=len(before),
            check='size and mtime_ns; includes original files and derived brain masks'))
        if not unchanged:
            raise RuntimeError('Input identities changed during evaluation')
        status.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        _write_json(root/'status.json',status)


if __name__ == '__main__':
    main()
