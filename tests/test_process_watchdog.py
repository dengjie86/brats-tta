import os
import subprocess
import sys
from pathlib import Path

import pytest

from brats_tta.utils.process_watchdog import run_with_progress_watchdog


def run_worker(tmp_path: Path, code: str, **kwargs) -> str:
    with (tmp_path/'worker.log').open('w', encoding='utf-8') as log:
        run_with_progress_watchdog([sys.executable, '-c', code], env=os.environ.copy(),
            log=log, progress_path=tmp_path/'progress.json', poll_interval=.02, **kwargs)
    return (tmp_path/'worker.log').read_text()


def test_healthy_worker_progress_prevents_timeout(tmp_path):
    path = repr(str(tmp_path / "progress.json"))
    code = (
        f"import time; from pathlib import Path; p=Path({path});\n"
        "for i in range(8):\n p.write_text(str(i)); time.sleep(.15)"
    )
    run_worker(tmp_path, code, timeout=.6, max_restarts=0)


def test_stalled_worker_is_killed_and_retries_are_bounded(tmp_path):
    with pytest.raises(TimeoutError, match='exhausted 1 restarts'):
        run_worker(tmp_path, 'import time; time.sleep(60)', timeout=.3, max_restarts=1)
    log = (tmp_path/'worker.log').read_text()
    assert log.count('WATCHDOG attempt=') == 2
    assert log.count('terminating owned worker') == 2


def test_restart_preserves_completed_work(tmp_path):
    marker = tmp_path/'completed_case'
    code = ('from pathlib import Path; import time; p=Path(' + repr(str(marker)) + ');\n'
            'if not p.exists():\n p.write_text("saved"); time.sleep(60)\n'
            'else:\n print(p.read_text())')
    log = run_worker(tmp_path, code, timeout=.6, max_restarts=1)
    assert 'saved' in log
    assert marker.read_text() == 'saved'


def test_real_failure_is_not_blindly_retried(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_worker(tmp_path, 'raise RuntimeError("bad input")', timeout=5, max_restarts=2)
    assert (tmp_path/'worker.log').read_text().count('WATCHDOG attempt=') == 1
