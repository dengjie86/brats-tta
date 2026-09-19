"""Bound GPU worker stalls without mistaking process existence for progress."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TextIO


def run_with_progress_watchdog(
    command: list[str], *, env: dict[str, str], log: TextIO, progress_path: Path,
    timeout: float = 300, max_restarts: int = 2, poll_interval: float = 2,
) -> None:
    """For resumable *episodic* workers only; kill only the child we created.

    The worker writes progress after real work, not from a timer heartbeat.
    A cached old progress file does not count as progress on a new invocation.
    Never skip a stalled case or discard its earlier completed predecessors.
    """
    if timeout <= 0 or poll_interval <= 0 or max_restarts < 0:
        raise ValueError("Invalid watchdog limits")
    for attempt in range(max_restarts + 1):
        last_signature = _signature(progress_path)
        last_progress = time.monotonic()
        log.write(f"\nWATCHDOG attempt={attempt+1} timeout={timeout}s\n")
        log.flush()
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                time.sleep(poll_interval)
                signature = _signature(progress_path)
                if signature != last_signature:
                    last_signature = signature
                    last_progress = time.monotonic()
                if time.monotonic() - last_progress > timeout:
                    log.write(f"\nWATCHDOG stalled pid={child.pid}; terminating owned worker\n")
                    log.flush()
                    child.kill()
                    child.wait(timeout=30)
                    break
            else:
                if child.returncode == 0:
                    return
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)
        if attempt == max_restarts:
            raise TimeoutError(f"No real progress for {timeout}s; exhausted {max_restarts} restarts")


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None
