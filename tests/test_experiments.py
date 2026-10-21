import copy
import json
import sys

import pytest
import yaml

from brats_tta.cli.experiments import main
from brats_tta.experiments import load_catalog


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "src/brats_tta").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    row = {
        "id": "example", "title": "Example", "model": "test",
        "status": "paused", "recorded_on": "2026-10-22", "entry_kind": "frozen_runner",
        "note": "User paused", "planned_cases": 98, "completed_cases": 66,
        "paths": {"directory": "outputs/example", "entrypoint": "outputs/example/run.py",
                  "protocol": "outputs/example/PROTOCOL.md", "report": "outputs/example/REPORT.md"},
    }
    data = {"schema_version": 1, "experiments": [row]}
    path = tmp_path / "configs/experiments.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path, path, data


def test_metadata_validation_works_without_local_assets(catalog, capsys):
    root, _, _ = catalog
    assert main(["--root", str(root), "check"]) == 0
    assert json.loads(capsys.readouterr().out)["local_checked"] is False
    assert main(["--root", str(root), "check", "--local"]) == 1
    assert len(json.loads(capsys.readouterr().out)["missing"]) == 4
    assert not (root / "outputs").exists()


def test_show_is_read_only_and_never_executes_referenced_runner(catalog, monkeypatch, capsys):
    root, _, _ = catalog
    output = root / "outputs/example"
    output.mkdir(parents=True)
    runner = output / "run.py"
    runner.write_text("raise RuntimeError('must never import this runner')\n", encoding="utf-8")
    monkeypatch.chdir(root.parent)
    modules_before = set(sys.modules)
    assert main(["--root", str(root), "show", "example"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "paused" and result["completed_cases"] == 66
    assert result["local_references"]["entrypoint"]["exists"]
    assert not any("outputs" in name for name in set(sys.modules) - modules_before)
    assert sorted(p.name for p in output.iterdir()) == ["run.py"]


@pytest.mark.parametrize("bad_path", ["../outside.py", "/tmp/run.py", "C:/outside.py", "..\\x.py"])
def test_catalog_rejects_external_or_parent_paths(catalog, bad_path):
    root, path, data = catalog
    data["experiments"][0]["paths"]["entrypoint"] = bad_path
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(root)


def test_duplicate_identity_and_false_completion_are_rejected(catalog):
    root, path, data = catalog
    original = copy.deepcopy(data)
    data["experiments"].append(copy.deepcopy(data["experiments"][0]))
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(root)
    original["experiments"][0]["status"] = "completed"
    path.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_catalog(root)


def test_unknown_identity_exits_without_creating_output(catalog):
    root, _, _ = catalog
    with pytest.raises(SystemExit) as error:
        main(["--root", str(root), "show", "missing"])
    assert error.value.code == 2
    assert not (root / "outputs").exists()
