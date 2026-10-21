
import pytest

from brats_tta.cli.project import main
from brats_tta.paths import resource_paths


def checkout(tmp_path):
    (tmp_path / "src/brats_tta").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/paths.yaml").write_text("model: models/source\n", encoding="utf-8")
    return tmp_path


def test_paths_follow_checkout_when_cwd_changes(tmp_path, monkeypatch):
    root = checkout(tmp_path / "checkout")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("BRATS_TTA_ROOT", str(root))
    assert resource_paths()["model"] == root / "models/source"


def test_check_command_reports_missing_then_present(tmp_path, capsys):
    root = checkout(tmp_path)
    assert main(["--root", str(root), "--check"]) == 1
    assert '"exists": false' in capsys.readouterr().out
    (root / "models/source").mkdir(parents=True)
    assert main(["--root", str(root), "--check"]) == 0


def test_invalid_path_config_is_rejected(tmp_path):
    root = checkout(tmp_path)
    config = root / "configs/paths.yaml"
    config.write_text("model: [one, two]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        resource_paths(root=root)
