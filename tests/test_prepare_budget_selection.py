import json
from pathlib import Path

import pytest
import yaml

from scripts import prepare_budget_selection as preparation
from scripts.make_pilot_grid import _fingerprint
from scripts.run_experiment import sha256


@pytest.fixture
def project(tmp_path, monkeypatch):
    base_path = tmp_path / preparation.BASE_CONFIG
    base_path.parent.mkdir(parents=True)
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["protocol_amendment"] = "budget120-v1.4"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    for name in (base["stats"]["analysis_list"], base["noise"]["rho_file"],
                 base["prompt"]["system_file"], base["prompt"]["fewshot_file"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    ids = tmp_path / "data/splits/dev_select.ids"
    ids.parent.mkdir(parents=True)
    ids.write_text("".join(f"train-{i:05d}\n" for i in range(200)), encoding="ascii")
    grid_path = tmp_path / "outputs/analysis/budget120_pilot_C_grid.json"
    grid_path.parent.mkdir(parents=True)
    plan_path = base_path.parent / "grid_plan.json"
    plan = {"root": "../../../..", "conditions": {"C": {"run_configs": [f"C{i}" for i in range(8)],
             "output": grid_path.relative_to(tmp_path).as_posix()}},
            "inputs_sha256": {"data/splits/dev_select.ids": sha256(ids)}}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    evidence = {"base_yaml_sha256": sha256(base_path)}
    decision = {"scope": "dev_base_pilot_only", "condition": "C", "plan_sha256": sha256(plan_path),
                "expected_alphas": base["noise"]["pilot_grid_C"],
                "inputs": {"A": evidence, "pilot_runs": [evidence for _ in range(8)]}}
    decision["input_sha256"] = _fingerprint(decision)
    decision["grid"] = {"launch_allowed": True, "status": "ok", "grid": [.01, .02, .04, .08, .16, .32]}
    grid_path.write_text(json.dumps(decision), encoding="utf-8")
    calls = []

    def revalidate(path):
        calls.append(path)
        return {str(grid_path): json.loads(grid_path.read_text(encoding="utf-8"))}

    monkeypatch.setattr(preparation, "make_pilot_grids", revalidate)
    return tmp_path, grid_path, plan_path, tmp_path / "configs/generated/budget120/selection", calls


def test_fixed_subset_reuses_frozen_base_and_prepares_only_four_runs(project):
    root, grid_path, plan_path, destination, calls = project
    before = (root / preparation.BASE_CONFIG).read_bytes()
    result = preparation.prepare(root, grid_path, plan_path, destination)
    assert preparation.prepare(root, grid_path, plan_path, destination) == result
    assert calls == [plan_path, plan_path]
    assert result["run_count"] == 4
    selection = json.loads((destination / "selection_plan.json").read_text())
    assert selection["original_grid"] == [.01, .02, .04, .08, .16, .32]
    assert selection["selected_grid"] == [.01, .04, .32]
    assert selection["selected_indices_zero_based"] == [0, 2, 5]
    assert selection["edge_extrapolation"] == {
        "source": "original_sorted_six_point_grid", "lower_ratio": 2, "upper_ratio": 2}
    assert (root / preparation.BASE_CONFIG).read_bytes() == before
    assert selection["base_config_sha256"] == sha256(root / preparation.BASE_CONFIG)
    assert selection["grid_decision_sha256"] == sha256(grid_path)
    configs = [selection["a_run_config"], *selection["c_run_configs"]]
    assert len(set(configs)) == 4
    for index, name in enumerate(configs):
        run = yaml.safe_load((root / name).read_text())
        assert run["base_config"] == preparation.BASE_CONFIG
        assert run["split"] == "dev_select" and run["n_rows"] == 16
        assert run["temperature"] == .5 and run["seed_set"] == 0
        assert run["row_start"] == 0 and run["id_version"] == 2
        assert run["basis"] == (None if index == 0 else "B0")
        assert run["alpha"] == (None if index == 0 else selection["selected_grid"][index - 1])
        assert selection["run_configs_sha256"][name] == sha256(root / name)
    assert all(sha256(root / name) == digest for name, digest in selection["inputs_sha256"].items())
    stage = json.loads((destination / "stage_plan.json").read_text())
    assert stage["commands"] == [["-m", "scripts.build_bank", configs[1]],
                                  *[["-m", "scripts.run_experiment", name] for name in configs]]
    assert set((root / "outputs").rglob("*")) == {grid_path.parent, grid_path}


@pytest.mark.parametrize("change", ["blocked", "underfull", "unsorted", "duplicate"])
def test_invalid_or_incomplete_six_point_grids_are_blocked(project, change):
    root, grid_path, plan_path, destination, _ = project
    decision = json.loads(grid_path.read_text())
    grid = decision["grid"]
    if change == "blocked":
        grid["launch_allowed"] = False
    elif change == "underfull":
        grid["grid"] = grid["grid"][:5]
    elif change == "unsorted":
        grid["grid"].reverse()
    else:
        grid["grid"][1] = grid["grid"][0]
    grid_path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="six-point"):
        preparation.prepare(root, grid_path, plan_path, destination)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["plan_sha", "fingerprint"])
def test_forged_decision_is_rejected_before_evidence_revalidation(project, change):
    root, grid_path, plan_path, destination, calls = project
    decision = json.loads(grid_path.read_text())
    decision["plan_sha256" if change == "plan_sha" else "input_sha256"] = "changed"
    grid_path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="SHA|fingerprint"):
        preparation.prepare(root, grid_path, plan_path, destination)
    assert not destination.exists() and not calls


def test_evidence_revalidation_failure_cannot_prepare_runs(project, monkeypatch):
    root, grid_path, plan_path, destination, _ = project

    def fail(path):
        raise ValueError("raw output SHA changed")

    monkeypatch.setattr(preparation, "make_pilot_grids", fail)
    with pytest.raises(ValueError, match="raw output SHA"):
        preparation.prepare(root, grid_path, plan_path, destination)
    assert not destination.exists()


def test_conflicting_file_preflights_whole_batch(project):
    root, grid_path, plan_path, destination, _ = project
    destination.mkdir(parents=True)
    conflict = destination / "stage_plan.json"
    conflict.write_text("conflict")
    with pytest.raises(ValueError, match="Conflicting"):
        preparation.prepare(root, grid_path, plan_path, destination)
    assert list(destination.iterdir()) == [conflict]
