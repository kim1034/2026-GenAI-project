import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts import finalize_budget_selection as finalizer
from scripts.run_experiment import run_id, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def selection(tmp_path, monkeypatch):
    base_rel = "configs/generated/budget120/pilot/base.yaml"
    base_path = tmp_path / base_rel
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["protocol_amendment"] = "budget120-v1.4"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    for rel in (base["stats"]["analysis_list"], base["noise"]["rho_file"],
                base["prompt"]["system_file"], base["prompt"]["fewshot_file"],
                "data/splits/dev_select.ids", "data/processed/train.jsonl"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic pinned evidence", encoding="utf-8")
    folder = tmp_path / "configs/generated/budget120/selection"
    folder.mkdir()
    calls, metrics = [], {"A": .6, .01: .60, .04: .65, .32: .61, .005: .705, "D": .64, "E": .63}

    def config(condition, alpha, target=None):
        run = {"base_config": base_rel, "split": "dev_select", "condition": condition, "alpha": alpha,
               "temperature": 0.0 if condition == "greedy" else .5,
               "basis": None if condition in ("A", "greedy") else "B0", "seed_set": 0,
               "row_start": 0, "n_rows": 1 if condition == "greedy" else 16, "id_version": 2}
        path = target or folder / f"{run_id(run)}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
        return path

    def complete(path):
        run = yaml.safe_load(path.read_text())
        name = run_id(run)
        output = tmp_path / "outputs/samples" / f"{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"synthetic complete {name}\n", encoding="utf-8")
        write_json(tmp_path / "outputs/manifests" / f"{name}.json",
                   {"status": "complete", "run_id": name, "output_sha256": sha256(output)})

    paths = [config("A", None), *[config("C", alpha) for alpha in (.01, .04, .32)]]
    greedy = config("greedy", None, tmp_path / "configs/runs/dev_select_greedy.yaml")
    for path in [greedy, *paths]:
        complete(path)
    write_json(tmp_path / "grid.json", {"grid": {"grid": [.01, .02, .04, .08, .16, .32], "status": "ok", "launch_allowed": True}})
    write_json(tmp_path / "pilot_plan.json", {"scope": "synthetic_completed_pilot"})
    plan = {"protocol_amendment": "budget120-v1.4", "split": "dev_select", "original_grid": [.01, .02, .04, .08, .16, .32],
            "selected_grid": [.01, .04, .32], "selected_indices_zero_based": [0, 2, 5],
            "edge_extrapolation": {"source": "original_sorted_six_point_grid", "lower_ratio": 2, "upper_ratio": 2},
            "base_config": base_rel, "base_config_sha256": sha256(base_path), "grid_decision": "grid.json",
            "grid_decision_sha256": sha256(tmp_path / "grid.json"), "pilot_plan": "pilot_plan.json",
            "pilot_plan_sha256": sha256(tmp_path / "pilot_plan.json"), "inputs_sha256": {base_rel: sha256(base_path)},
            "a_run_config": paths[0].relative_to(tmp_path).as_posix(),
            "c_run_configs": [path.relative_to(tmp_path).as_posix() for path in paths[1:]],
            "run_configs_sha256": {path.relative_to(tmp_path).as_posix(): sha256(path) for path in paths}}
    plan_path = folder / "selection_plan.json"
    write_json(plan_path, plan)

    # Raw/config/runtime verification is delegated to project_power._load_run,
    # whose real synthetic 200-problem tests cover that reader. Here isolate
    # adaptive orchestration and use real select_alpha/edge decision functions.
    def load(root, path, condition, alpha, **kwargs):
        run = yaml.safe_load(path.read_text())
        assert run["condition"] == condition and run["alpha"] == alpha
        manifest = tmp_path / "outputs/manifests" / f"{run_id(run)}.json"
        entry = json.loads(manifest.read_text())
        output = tmp_path / "outputs/samples" / f"{run_id(run)}.jsonl"
        if entry["output_sha256"] != sha256(output):
            raise ValueError("Synthetic completed output changed")
        item = {"run_id": run_id(run), "manifest_sha256": sha256(manifest), "output_sha256": sha256(output),
                "execution_signature": {"base": {"model": base["model"]}}}
        if condition == "greedy":
            return item, {"verified_greedy": True}, []
        auroc = metrics[alpha] if condition == "C" else metrics[condition]
        summary = {"alpha": alpha, "condition": condition, "split": "dev_select", "status": "complete",
                   "run_id": run_id(run), "auroc": auroc, "maj_acc": .8, "invalid": .01,
                   "label_classes_complete": True}
        return item, summary

    def child(argv, *, cwd, check):
        assert argv[:4] == [sys.executable, "-u", "-m", "scripts.run_experiment"]
        assert check is True and cwd == tmp_path
        path = tmp_path / argv[4]
        run = yaml.safe_load(path.read_text())
        calls.append((run["condition"], run["alpha"]))
        complete(path)
    monkeypatch.setattr(finalizer, "_load", load)
    monkeypatch.setattr(finalizer, "_ensure_bank", lambda *args, **kwargs: None)
    monkeypatch.setattr(finalizer, "_require_outer_lock", lambda root: None)
    monkeypatch.setattr(finalizer.subprocess, "run", child)
    return tmp_path, plan_path.relative_to(tmp_path).as_posix(), plan, paths, metrics, calls


def test_initial_queue_no_edge_and_same_alpha_de_then_identical_resume(selection):
    root, plan_path, _, _, _, calls = selection
    result = finalizer.finalize(root, plan_path, execute=True, run_initial=True)
    assert result["common_alpha"] == .04
    assert not result["edge_used"]
    assert result["edge_checks_complete"]
    assert result["feasible_count"] == 3
    assert calls == [("A", None), ("C", .01), ("C", .04), ("C", .32), ("D", .04), ("E", .04)]
    assert set(result["runs"]) == {"A", "C", "D", "E"}
    assert "alpha_S" not in result
    assert all(sha256(root / path) == digest for path, digest in result["inputs_sha256"].items())
    assert finalizer.finalize(root, plan_path) == result
    assert len(calls) == 6


@pytest.mark.parametrize("edge_auroc,expected", [(.705, .01), (.7051, .005)])
def test_edge_uses_original_six_point_ratio_and_strict_improvement(selection, edge_auroc, expected):
    root, plan_path, _, _, metrics, calls = selection
    metrics.update({.01: .7, .04: .6, .32: .59, .005: edge_auroc})
    result = finalizer.finalize(root, plan_path, execute=True)
    assert result["common_alpha"] == expected
    assert result["edge_used"]
    assert calls == [("C", .005), ("D", expected), ("E", expected)]
    edge = json.loads((root / "outputs/analysis/budget120_selection_edge.json").read_text())
    assert edge["accepted"] is (expected == .005)
    assert finalizer.finalize(root, plan_path) == result
    assert len(calls) == 3


def test_bad_initial_settings_stop_before_any_child(selection):
    root, plan_path, plan, paths, _, calls = selection
    run = yaml.safe_load(paths[1].read_text())
    run["temperature"] = .8
    paths[1].write_text(yaml.safe_dump(run), encoding="utf-8")
    plan["run_configs_sha256"][paths[1].relative_to(root).as_posix()] = sha256(paths[1])
    write_json(root / plan_path, plan)
    with pytest.raises(ValueError, match="does not match"):
        finalizer.finalize(root, plan_path, execute=True, run_initial=True)
    assert not calls
    assert not (root / "outputs/analysis/budget120_selection.json").exists()


def test_resume_conflict_preserves_existing_decisions(selection):
    root, plan_path, plan, _, _, calls = selection
    finalizer.finalize(root, plan_path, execute=True)
    initial = root / "outputs/analysis/budget120_selection_initial.json"
    original = initial.read_bytes()
    plan["unexpected_source_change"] = True
    write_json(root / plan_path, plan)
    with pytest.raises(ValueError, match="conflicts"):
        finalizer.finalize(root, plan_path)
    assert initial.read_bytes() == original
    assert len(calls) == 2


def test_verify_only_never_launches_or_claims_final_completion(selection):
    root, plan_path, _, _, metrics, calls = selection
    metrics.update({.01: .7, .04: .6, .32: .59})
    result = finalizer.finalize(root, plan_path)
    assert result["status"] == "pending_runs"
    assert not calls
    assert (root / "outputs/analysis/budget120_selection_initial.json").exists()
    assert not (root / "outputs/analysis/budget120_selection.json").exists()


def test_existing_bank_builder_is_waited_for_without_second_writer(tmp_path, monkeypatch):
    run = {"condition": "D"}
    path = finalizer.bank_path(tmp_path, "D", "B0", 0, "dev_select", 0, 16)
    monkeypatch.setattr(finalizer, "_other_bank_builders", lambda root: [123])
    def finish(_):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
        path.with_suffix(".json").write_text("{}")
    monkeypatch.setattr(finalizer.time, "sleep", finish)
    monkeypatch.setattr(finalizer, "_command", lambda *args: pytest.fail("Must not start a second bank writer"))
    finalizer._ensure_bank(tmp_path, tmp_path / "config.yaml", run, 8)
