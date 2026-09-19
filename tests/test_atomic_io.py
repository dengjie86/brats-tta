import json
import os

import pytest

from brats_tta.cli.evaluate_tta import _write_json
from brats_tta.utils import atomic_io


def test_transient_access_denied_retries_and_preserves_previous_output(tmp_path, monkeypatch):
    path = tmp_path/'progress.json'
    _write_json(path, {'old':True})
    replace = os.replace
    calls = []
    def briefly_locked(source, destination):
        calls.append(source)
        if len(calls) <= 3:
            assert json.loads(path.read_text()) == {'old':True}
            raise PermissionError(13, 'simulated Windows access denied')
        replace(source, destination)
    monkeypatch.setattr(atomic_io.os, 'replace', briefly_locked)
    monkeypatch.setattr(atomic_io.time, 'sleep', lambda _: None)
    _write_json(path, {'completed':47})
    assert len(calls) == 4
    assert json.loads(path.read_text()) == {'completed':47}
    assert not list(tmp_path.glob('*.tmp'))


def test_permanent_denial_is_bounded_and_does_not_destroy_old_file(tmp_path, monkeypatch):
    path = tmp_path/'progress.json'
    atomic_io.atomic_write_text(path, 'old')
    calls = []
    def locked(*args):
        calls.append(args)
        raise PermissionError(13, 'persistent denial')
    monkeypatch.setattr(atomic_io.os, 'replace', locked)
    monkeypatch.setattr(atomic_io.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_text(path, 'new', attempts=3)
    assert len(calls) == 3
    assert path.read_text() == 'old'
    assert not list(tmp_path.glob('*.tmp'))


def test_other_io_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []
    def disk_full(*args):
        calls.append(args)
        raise OSError(28, 'disk full')
    monkeypatch.setattr(atomic_io.os, 'replace', disk_full)
    with pytest.raises(OSError, match='disk full'):
        atomic_io.atomic_write_text(tmp_path/'progress.json', 'new')
    assert len(calls) == 1
