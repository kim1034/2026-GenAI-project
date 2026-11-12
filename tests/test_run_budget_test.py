import json

import pytest

from scripts import run_budget_test as runner
from scripts.run_experiment import sha256


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path
    folder = root / runner.FINAL
    folder.parent.mkdir(parents=True)
    configs, rows, commands = {}, [], []
    for c in ("greedy", "C", "D", "E"):
        path = folder.parent / f"test_{c}.yaml"
        path.write_text("synthetic pinned configuration")
        name = path.relative_to(root).as_posix()
        configs[c] = name
        rows.append({"condition": c, "run_config": name, "run_config_sha256": sha256(path),
                     "rows": 1 if c == "greedy" else 16, "problem_count": 1319})
        if c != "greedy":
            commands.append(["-m", "scripts.build_bank", name])
        commands.append(["-m", "scripts.run_experiment", name])
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(folder, {"base_config": runner.BASE, "common_alpha": .1, "selection": "selection.json",
                   "run_configs": {"test": configs}})
    write(root / runner.RECEIPT, {"common_alpha": .1, "required_test_runs": rows})
    write(folder.parent / "stage_test.json", {"commands": commands})
    write(folder.parent / "test_analysis_plan.json", {"root": "../../../..", "split": "test",
        "base_config": runner.BASE, "selection_file": "selection.json", "run_configs": configs,
        "output_dir": "outputs/analysis/budget120/test"})
    calls = []
    monkeypatch.setattr(runner, "verify_lock", lambda root: {"LOCK_sha256": "sealed-sha"})
    monkeypatch.setattr(runner, "_require_outer_lock", lambda root: calls.append("lock"))
    def execute(root, command):
        calls.append(command)
        if command[0] == "scripts.analyze_budget120":
            write(root / "outputs/analysis/budget120/test/analysis.json", {
                "status": "complete", "scope": "confirmatory_test", "n": 1319, "common_alpha": .1,
                "inputs": {"LOCK_sha256": "sealed-sha"}, "input_sha256": "input-hash"})
    monkeypatch.setattr(runner, "_command", execute)
    return root, calls


def test_exact_order_then_analysis_and_pending_backup_receipt(sealed):
    root, calls = sealed
    result = runner.run_test(root)
    modules = [call[0] for call in calls[1:]]
    assert modules == ["scripts.run_experiment"] + ["scripts.build_bank", "scripts.run_experiment"] * 3 + ["scripts.analyze_budget120"]
    assert calls[1][1].endswith("test_greedy.yaml")
    assert result["local_backup_verification"] == "pending"
    assert result["gpu_return"] == "manual_user"
    assert (root / "outputs/reproducibility/final_test_receipt.json").exists()


def test_invalid_seal_cannot_launch(sealed, monkeypatch):
    root, calls = sealed
    def reject(root):
        raise PermissionError("test sealed")
    monkeypatch.setattr(runner, "verify_lock", reject)
    with pytest.raises(PermissionError):
        runner.run_test(root)
    assert calls == []


def test_changed_registered_config_cannot_launch(sealed):
    root, calls = sealed
    (root / runner.FINAL).parent.joinpath("test_D.yaml").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        runner.run_test(root)
    assert calls == []


def test_failed_inference_never_runs_analysis_or_reports_complete(sealed, monkeypatch):
    root, calls = sealed
    def fail(root, command):
        calls.append(command)
        raise RuntimeError("inference failed")
    monkeypatch.setattr(runner, "_command", fail)
    with pytest.raises(RuntimeError):
        runner.run_test(root)
    assert not any(isinstance(call, list) and call[0] == "scripts.analyze_budget120" for call in calls)
    assert not (root / "outputs/reproducibility/final_test_receipt.json").exists()
