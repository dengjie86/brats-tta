from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast

import numpy as np
import pytest
import torch
from scipy.special import softmax

from brats_tta.native.context import VIEWS, teacher
from brats_tta.native.coordinates import restore
from brats_tta.native.efficient import factor_scores
from brats_tta.native.replay import objective
from brats_tta.pipeline import run_case
from brats_tta.pipeline.backends import SeniorBackend
from brats_tta.pipeline.decoder import decode_no_come
from brats_tta.pipeline.probability_fusion import predict_volume
from brats_tta.pipeline.probability_replay import Layout
from brats_tta.pipeline.state import SourceState
from brats_tta.pipeline.teacher import assemble_teacher
from reference.tta_pipeline import decoder as legacy_decoder
from reference.tta_pipeline import teacher_rules as legacy_teacher


def geometry(shape):
    return dict(shape=np.array(shape), bbox=np.array([[0, n] for n in shape]), fw=np.array([2, 1, 0]))


@pytest.mark.parametrize("seed", [7, 19, 31])
def test_union2_matches_frozen_functions_and_cached_reconstruction(seed):
    rng = np.random.default_rng(seed)
    shape = (6, 5, 4)
    p = rng.dirichlet(np.ones(4), size=np.prod(shape)).T.astype(np.float32).reshape(4, *shape)
    source = p.argmax(0).astype(np.uint8)
    base = [
        rng.dirichlet(np.full(4, 0.05), size=source.size).T.astype(np.float32).reshape(4, *shape)
        for _ in VIEWS
    ]
    protected = rng.random(shape) > 0.2
    old, accepted = teacher(source, base, protected)
    idx = np.flatnonzero(protected)
    evidence = {
        view: rng.dirichlet(np.full(4, 0.02), size=(4, len(idx))).transpose(0, 2, 1).astype(np.float32)
        for view in VIEWS
    }
    evidence["index"] = idx
    # Deliberately add two supporters, a conflicting proposal, and old edits.
    for j, flat in enumerate(idx[:12]):
        target = (int(source.ravel()[flat]) + 1) % 4
        for view in VIEWS:
            evidence[view][:2, :, j] = 0.02
            evidence[view][:2, target, j] = 0.94
    data = dict(
        p=p,
        source=source,
        teacher=old,
        accepted=accepted,
        geo=geometry(shape),
        qteacher=(base[0].astype(np.float64) + base[1].astype(np.float64)) * 0.5,
    )
    expected, expected_index, expected_q = legacy_teacher.proposal(data, evidence)
    independent, _, _ = legacy_teacher.proposal(data, evidence, independent=True)
    actual, delta, edits, _, (index, q) = assemble_teacher(p, base, protected, data["geo"], evidence=evidence)
    assert len(index) > 0
    for key in ("teacher", "accepted", "qteacher"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(actual["teacher"], independent["teacher"])
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(q, expected_q)
    cached = assemble_teacher(p, base, protected, data["geo"], additions=(index, q))
    np.testing.assert_array_equal(cached[1], delta)
    np.testing.assert_array_equal(cached[2], edits)
    np.testing.assert_array_equal(actual["teacher"][old != source], old[old != source])


@pytest.mark.parametrize("cropped", [False, True])
def test_decoder_matches_legacy_and_independent_branch(cropped):
    rng = np.random.default_rng(121)
    shape = (7, 7, 7)
    geo = (
        dict(shape=np.array([12, 14, 16]), bbox=np.array([[2, 9], [1, 8], [3, 10]]), fw=np.array([1, 2, 0]))
        if cropped
        else geometry(shape)
    )
    for trial in range(4):
        p = softmax(rng.normal(size=(4, *shape)), axis=0).astype(np.float32)
        student = softmax(rng.normal(size=p.shape), axis=0).astype(np.float32)
        if trial == 0:
            student = p.copy()
        q = softmax(rng.normal(size=p.shape) * 5, axis=0)
        data = dict(
            p=p,
            source=restore(p.argmax(0).astype(np.uint8), **geo),
            geo=geo,
            accepted=rng.random(shape) > 0.5,
            qteacher=q,
        )
        for bias in (0.0, -0.7, 0.9):
            actual = decode_no_come(data, student, bias)
            np.testing.assert_array_equal(actual, legacy_decoder.decode_no_come(data, student, bias))
            np.testing.assert_array_equal(actual, legacy_decoder.decode_no_come(data, student, bias, True))


def toy_network():
    torch.manual_seed(129)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(4, 4, 1), torch.nn.BatchNorm3d(4), torch.nn.LeakyReLU(), torch.nn.Conv3d(4, 4, 1)
    )
    model.eval().requires_grad_(False)
    model[1].requires_grad_(True)
    return model


def fused_graph(network, layout):
    output = torch.zeros((4, *layout.shape))
    for sl in layout.slices:
        x = torch.from_numpy(np.ascontiguousarray(layout.image[(slice(None), *sl)]))[None]
        p = network(x).float().softmax(1)[0]
        output[(slice(None), *sl)] = output[(slice(None), *sl)] + p * torch.from_numpy(layout.weight)
    output = output / torch.from_numpy(layout.count)[None]
    return output[(slice(None),) + tuple(slice(0, n) for n in layout.original)]


def test_probability_replay_matches_direct_graph_with_padding_and_overlap():
    network = toy_network()
    rng = np.random.default_rng(17)
    image = rng.normal(size=(4, 5, 6, 3)).astype(np.float32)
    layout = Layout(image, (4, 4, 4))
    merged = fused_graph(network, layout)
    old, _ = predict_volume(network, image, torch.float32, patch=(4, 4, 4))
    np.testing.assert_array_equal(old, merged.detach().numpy())
    gradient = torch.from_numpy(rng.normal(size=old.shape).astype(np.float32))
    (merged * gradient).sum().backward()
    expected = [p.grad.clone() for p in network.parameters() if p.requires_grad]
    network.zero_grad(set_to_none=True)
    layout.replay(network, gradient.numpy(), torch.float32)
    for actual, wanted in zip([p.grad for p in network.parameters() if p.requires_grad], expected):
        torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=2e-5)


def test_senior_eight_updates_match_independent_full_graph():
    network = toy_network()
    direct = deepcopy(network)
    rng = np.random.default_rng(4)
    image = rng.normal(size=(4, 5, 5, 3)).astype(np.float32)
    backend = SeniorBackend(network, torch.float32, (4, 4, 4))
    source, _ = backend.predict(image)
    delta = rng.normal(size=source.shape).astype(np.float32) * 0.1
    edits = np.zeros(image.shape[1:], bool)
    edits[1:3, 1:3, 1] = True
    delta[:, ~edits] = 0
    student, selected, history = backend.learn(image, source, None, delta, edits, lambda *args: None)
    reference = torch.from_numpy(source).clamp_min(1e-30).log()
    source_scores = factor_scores(reference, reference)
    target = factor_scores(reference + torch.from_numpy(delta), reference) - source_scores
    target -= target.mean(0, keepdim=True)
    optimizer = torch.optim.Adam([p for p in direct.parameters() if p.requires_grad], lr=0.001)
    layout = Layout(image, (4, 4, 4))
    for record in history:
        optimizer.zero_grad(set_to_none=True)
        output = fused_graph(direct, layout).clamp_min(1e-30).log()
        loss, _ = objective(factor_scores(output, reference), source_scores, target, torch.from_numpy(edits))
        assert float(loss.detach()) == pytest.approx(record["total"], abs=2e-7)
        (loss * 1024).backward()
        for p in direct.parameters():
            if p.requires_grad:
                p.grad.div_(1024)
        optimizer.step()
    assert len(history) == 8
    for name, parameter in direct.named_parameters():
        if name in selected:
            torch.testing.assert_close(parameter, selected[name], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(student, fused_graph(direct, layout).detach().numpy(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("fail", [False, True])
def test_episode_restores_buffers_parameters_flags_and_gradients(fail):
    network = toy_network()
    network.train()
    network[1].eval()
    network[1].weight.grad = torch.ones_like(network[1].weight)
    original = deepcopy(network.state_dict())
    flags = [m.training for m in network.modules()]
    required = [p.requires_grad for p in network.parameters()]
    try:
        with SourceState(network):
            with torch.no_grad():
                network[1].weight.add_(1)
                network[1].running_mean.add_(2)
            if fail:
                raise RuntimeError("synthetic failure")
    except RuntimeError:
        assert fail
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert flags == [m.training for m in network.modules()]
    assert required == [p.requires_grad for p in network.parameters()]
    assert torch.equal(network[1].weight.grad, torch.ones(4))


def test_full_orchestration_is_episodic_and_uses_ten_observations():
    network = toy_network()
    original = deepcopy(network.state_dict())
    image = np.ones((4, 6, 5, 4), np.float32)
    p = np.full(image.shape, 0.025, np.float32)
    p[2] = 0.925
    q = p.copy()
    q[:, 2:4] = 0.025
    q[1, 2:4] = 0.925
    calls = []

    def predict(x):
        calls.append(1)
        return (p if len(calls) == 1 else q).copy(), None

    def learn(x, probability, reference, delta, edits, progress):
        assert edits.any() and not edits.all()
        with torch.no_grad():
            network[1].bias.add_(0.1)
        return p.copy(), {"1.bias": network[1].bias}, [dict(step=i) for i in range(1, 9)]

    backend = SimpleNamespace(
        network=network,
        modalities=("a", "b", "c", "d"),
        predict=predict,
        extent_target=lambda *args: 0.9,
        learn=learn,
    )
    result = run_case(backend, image, geometry(image.shape[1:]))
    assert len(calls) == 11 and result["updates"] == 8 and result["source_restored"]
    assert result["affine"]["1.bias"].sum() != 0
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_production_pipeline_has_no_dated_imports_or_repository_path_search():
    package = Path(__file__).resolve().parents[1] / "src/brats_tta/pipeline"
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "spec_from_file_location" not in source and "sys.path" not in source
        assert "outputs/" not in source and "artifacts/experiments/" not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("reference.")


def test_no_edit_skips_optimization_and_preserves_source():
    backend = SeniorBackend(toy_network(), torch.float32, (4, 4, 4))
    image = np.ones((4, 3, 3, 3), np.float32)
    source, _ = backend.predict(image)
    before = deepcopy(backend.network.state_dict())
    with SourceState(backend.network):
        result, _, history = backend.learn(
            image,
            source,
            None,
            np.zeros_like(source),
            np.zeros(image.shape[1:], bool),
            lambda *args: pytest.fail("No update expected"),
        )
    assert not history
    np.testing.assert_array_equal(result, source)
    for name, value in backend.network.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("output_kind", ["existing", "outside", "missing_inputs"])
def test_cli_refuses_invalid_destinations_or_inputs_without_loading_model(tmp_path, monkeypatch, output_kind):
    from brats_tta.cli import adapt_context as cli

    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src/brats_tta").mkdir(parents=True)
    output = tmp_path / "outputs/new"
    if output_kind == "existing":
        output.mkdir(parents=True)
        (output / "sentinel").write_text("preserve")
    elif output_kind == "outside":
        output = tmp_path / "outside"
    monkeypatch.setattr(cli, "load_backend", lambda *args: pytest.fail("Must reject before model loading"))
    args = [
        "--root",
        str(tmp_path),
        "--profile",
        "senior188-single-view8",
        "--model",
        "missing.pth",
        "--output",
        str(output),
    ]
    for name in ("t1c", "t1n", "t2w", "t2f"):
        args += ["--" + name, str(tmp_path / (name + ".nii.gz"))]
    with pytest.raises(SystemExit) as failure:
        cli.main(args)
    assert failure.value.code == 2
    if output_kind == "existing":
        assert (output / "sentinel").read_text() == "preserve"
        assert list(output.iterdir()) == [output / "sentinel"]
    else:
        assert not output.exists()


def test_senior_input_order_normalization_and_grid_validation(tmp_path):
    import nibabel as nib
    from brats_tta.pipeline.io import load_case

    files, expected = {}, []
    rng = np.random.default_rng(3)
    backend = SeniorBackend(toy_network())
    for name in backend.modalities:
        raw = rng.normal(size=(5, 4, 3)).astype(np.float32)
        raw[0] = 0
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(raw, np.eye(4)), path)
        files[name] = path
        foreground = raw[raw != 0]
        normalized = np.zeros_like(raw)
        normalized[raw != 0] = (foreground - float(foreground.mean())) / max(float(foreground.std()), 1e-6)
        expected.append(normalized)
    actual, geo, _ = load_case(backend, files)
    np.testing.assert_array_equal(actual, np.stack(expected))
    np.testing.assert_array_equal(restore(actual[0], **geo), expected[0])
    nib.save(nib.Nifti1Image(raw, np.diag([1.1, 1, 1, 1])), files[backend.modalities[0]])
    with pytest.raises(ValueError, match="1 mm"):
        load_case(backend, files)


def test_cli_writes_bound_artifacts_and_completion_for_isolated_case(tmp_path, monkeypatch):
    import json
    import nibabel as nib
    from brats_tta.cli import adapt_context as cli

    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src/brats_tta").mkdir(parents=True)
    backend = SeniorBackend(toy_network(), torch.float32, (4, 4, 4))
    monkeypatch.setattr(cli, "load_backend", lambda *args: (backend, {"profile": "synthetic_cpu_test"}))
    output = tmp_path / "outputs/test_case"
    args = [
        "--root",
        str(tmp_path),
        "--profile",
        "senior188-single-view8",
        "--model",
        "synthetic",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    for name in backend.modalities:
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4)), path)
        args += ["--" + name, str(path)]
    assert cli.main(args) == 0
    result = json.loads((output / "complete.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    assert result["source_restored"] and not result["labels_used"]
    assert "torch" in inputs["runtime"]["packages"]
    assert set(inputs["inputs"]) == set(backend.modalities)
    assert nib.load(output / "prediction.nii.gz").shape == (4, 4, 4)
    for name, digest in result["files"].items():
        assert cli.file_sha(output / name) == digest
