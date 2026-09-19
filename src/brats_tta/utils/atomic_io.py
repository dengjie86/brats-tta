"""Atomic output writes tolerant of brief Windows reader/antivirus locks."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, attempts: int = 20) -> None:
    atomic_write_bytes(path, text.encode('utf-8'), attempts=attempts)


def atomic_write_bytes(path: Path, data: bytes, *, attempts: int = 20) -> None:
    """Keep the prior file intact until replacement succeeds; bound lock retries."""
    if attempts < 1:
        raise ValueError('attempts must be positive')
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(.05 * 2 ** attempt, .5))
    finally:
        if temporary is not None:
            # This is only the unique temporary file created by this invocation.
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                # Do not mask an earlier write failure with cleanup failure.
                pass
