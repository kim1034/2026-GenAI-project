import json

import pytest
import yaml

from scripts import run_budget_rehearsal as rehearsal
from scripts.run_experiment import sha256


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    folder = tmp_path / "configs/generated/budget120/final"
    folder.mkdir(parents=True)
    configs = {}
    for c in ("C", "D", "E"):
        path = folder / f"dev_anal_{c}.yaml"
        path.write_text(yaml.safe_dump({"condition": c, "split": "dev_anal", "n_rows": 16,
            "alpha": .1, "temperature": .5, "basis": "B0", "seed_set": 0,
            "row_start": 0, "id_version": 2, "base_config": "base.yaml"}), encoding="utf-8")
        configs[c] = path.relative_to(tmp_path).as_posix()
    commands = [["-m", "scripts.build_bank", configs[c]] for c in ("C", "D", "E")]
    commands += [["-m", "scripts.run_experiment", configs[c]] for c in ("C", "D", "E")]
    (folder / "stage_rehearsal.json").write_text(json.dumps({"commands": commands}), encoding="utf-8")
    (folder / "rehearsal_analysis_plan.json").write_text(json.dumps({"root": "../../../..",
        "split": "dev_anal", "base_config": "base.yaml", "selection_file": "selection.json",
        "run_configs": {"greedy": "configs/runs/dev_anal_greedy.yaml", **configs}}), encoding="utf-8")
    plan = {"selection": "selection.json", "base_config": "base.yaml", "common_alpha": .1,
            "run_configs": {"dev_anal": configs},
            "files_sha256": {p.relative_to(tmp_path).as_posix(): sha256(p) for p in folder.iterdir()}}
    path = folder / "final_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    calls = []
    monkeypatch.setattr(rehearsal, "prepare", lambda *args: calls.append("verify_selection"))
    monkeypatch.setattr(rehearsal, "_require_outer_lock", lambda root: calls.append("lock"))
    monkeypatch.setattr(rehearsal, "_command", lambda root, argv: calls.append(argv))
    return tmp_path, path.relative_to(tmp_path), calls, plan


def test_only_frozen_development_runs_then_analysis(prepared):
    root, path, calls, _ = prepared
    rehearsal.run_rehearsal(root, path)
    assert calls[:2] == ["verify_selection", "lock"]
    assert [call[0] for call in calls[2:]] == ["scripts.build_bank"] * 3 + ["scripts.run_experiment"] * 3 + ["scripts.analyze_budget120"]
    assert all("test" not in " ".join(call) for call in calls[2:])


@pytest.mark.parametrize("change", ["hash", "test_command", "alpha"])
def test_changed_evidence_or_test_command_cannot_launch(prepared, change):
    root, path, calls, plan = prepared
    if change == "test_command":
        target = root / path.parent / "stage_rehearsal.json"
        target.write_text(json.dumps({"commands": [["-m", "scripts.run_experiment", "test.yaml"]]}))
        plan["files_sha256"][target.relative_to(root).as_posix()] = sha256(target)
        (root / path).write_text(json.dumps(plan))
    else:
        target = root / plan["run_configs"]["dev_anal"]["C"]
        value = yaml.safe_load(target.read_text())
        value["alpha"] = .2
        target.write_text(yaml.safe_dump(value))
        if change == "alpha":
            plan["files_sha256"][target.relative_to(root).as_posix()] = sha256(target)
            (root / path).write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        rehearsal.run_rehearsal(root, path)
    assert calls == ["verify_selection"]


def test_failed_inference_prevents_analysis(prepared, monkeypatch):
    root, path, calls, _ = prepared
    def fail(root, command):
        calls.append(command)
        if command[0] == "scripts.run_experiment":
            raise RuntimeError("inference failure")
    monkeypatch.setattr(rehearsal, "_command", fail)
    with pytest.raises(RuntimeError, match="inference failure"):
        rehearsal.run_rehearsal(root, path)
    assert not any(isinstance(call, list) and call[0] == "scripts.analyze_budget120" for call in calls)
