"""Entry-point contracts with injected dependencies, not real Cube validation."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cube_reference
import research_cube
from cube_research import model


@pytest.fixture
def reference_inputs(tmp_path, monkeypatch):
    dataset, weights = tmp_path / "cube.h5", tmp_path / "weights.pt"
    dataset.write_bytes(b"entrypoint fixture, not HDF5")
    weights.write_bytes(b"entrypoint fixture, not weights")
    (tmp_path / "config.json").write_text("{}")
    config = tmp_path / "local.yaml"
    config.write_text(yaml.safe_dump(dict(
        dataset=str(dataset), weights=str(weights), device="cpu", render_backend="osmesa",
    )))
    manifest = {"cases": [{"case_id": "fixture"}], "schema": "fixture-only"}
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps(manifest))
    state = SimpleNamespace(calls=[], loaded=[], closed=False)
    reference = ModuleType("cube_research.reference")
    reference.CEM, reference.EVAL, reference.PLAN = {}, {}, SimpleNamespace(horizon=5)
    reference.PROTOCOLS = {"paper-300x10-top30": 10, "public-300x30-top30": 30}
    reference.fixed_manifest = lambda data: manifest
    reference.fit_actions = lambda data: SimpleNamespace(
        mean_=np.zeros(5), scale_=np.ones(5), n_samples_seen_=25,
    )
    reference.open_dataset = lambda *args: SimpleNamespace(
        h5_file=SimpleNamespace(close=lambda: setattr(state, "closed", True)),
    )
    reference.run_policy = lambda *args: state.calls.append(args[3])
    reference.runtime_identity = lambda device, bridge: {"device": device}

    def frozen(cfg, stats):
        state.loaded.append(dict(cfg))
        return SimpleNamespace(device=cfg["device"], base=object())

    monkeypatch.setitem(sys.modules, "cube_research.reference", reference)
    monkeypatch.setattr(model, "FrozenModel", frozen)
    for key in ("MUJOCO_GL", "HF_HUB_OFFLINE", "SDL_VIDEODRIVER", "MPLBACKEND"):
        monkeypatch.setenv(key, "")
    return config, case_file, tmp_path / "output", state, reference


def test_evaluate_needs_manifest_but_no_historical_audit():
    with pytest.raises(SystemExit):
        cube_reference.main(["evaluate-50", "--out", "unused"])
    for obsolete in ("--prior", "--verified-audit"):
        with pytest.raises(SystemExit):
            cube_reference.parser().parse_args(["manifest", "--out", "unused", obsolete, "old"])
    with pytest.raises(SystemExit):
        cube_reference.parser().parse_args(["audit-two", "--out", "unused"])
    with pytest.raises(SystemExit):
        research_cube.parser().parse_args(["perturb-e1"])


@pytest.mark.parametrize("check_only", [True, False])
def test_current_manifest_runs_without_old_artifacts(reference_inputs, check_only):
    config, manifest, out, state, _ = reference_inputs
    argv = ["evaluate-50", "--config", str(config), "--manifest", str(manifest), "--out", str(out)]
    if check_only:
        argv += ["--check-only"]
    cube_reference.main(argv)
    assert state.calls == ([] if check_only else ["cem", "zero", "random"])
    assert state.closed and state.loaded[0]["device"] == "cpu"
    status = json.loads((out / "STATUS.json").read_text())["status"]
    assert status == ("ready-no-rollouts" if check_only else "completed")
    assert json.loads((out / "protocol.json").read_text())["mujoco_gl"] == "osmesa"


def test_manifest_mismatch_stops_before_model_and_rollouts(reference_inputs):
    config, manifest, out, state, _ = reference_inputs
    manifest.write_text('{"cases": []}')
    with pytest.raises(ValueError, match="Manifest"):
        cube_reference.main(["evaluate-50", "--config", str(config),
                             "--manifest", str(manifest), "--out", str(out)])
    assert not state.calls and not state.loaded and state.closed
    assert json.loads((out / "STATUS.json").read_text())["status"] == "unfinished"


def test_manifest_mode_does_not_load_weights_or_execute(reference_inputs):
    config, _, out, state, _ = reference_inputs
    cube_reference.main(["manifest", "--config", str(config), "--out", str(out)])
    assert not state.loaded and not state.calls and state.closed
    assert (out / "reference-50.json").is_file()
    with pytest.raises(FileExistsError):
        cube_reference.main(["manifest", "--config", str(config), "--out", str(out)])


def test_failure_is_recorded_and_dataset_closed(reference_inputs):
    config, manifest, out, state, reference = reference_inputs

    def fail(*args):
        raise RuntimeError("injected execution failure")

    reference.run_policy = fail
    with pytest.raises(RuntimeError, match="injected"):
        cube_reference.main(["evaluate-50", "--config", str(config),
                             "--manifest", str(manifest), "--out", str(out),
                             "--device", "mps", "--render-backend", "glfw"])
    assert state.closed and state.loaded[0]["device"] == "mps"
    status = json.loads((out / "STATUS.json").read_text())
    assert status["status"] == "unfinished" and "injected" in status["error"]
