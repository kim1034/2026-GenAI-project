import json

import pytest

from scripts import run_budget_pipeline_check as runner
from scripts.run_experiment import sha256


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    folder = root / runner.FINAL
    folder.parent.mkdir(parents=True)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    selected = {c: {"run_config": f"configs/existing_{c}.yaml"} for c in "CDE"}
    write(root / "selection.json", {"runs": selected})
    path = folder.parent / "pipeline_check_analysis_plan.json"
    write(path, {"root": "../../../..", "split": "dev_select", "selection_file": "selection.json",
        "base_config": "base.yaml", "run_configs": {"greedy": "configs/runs/dev_select_greedy.yaml",
            **{c: selected[c]["run_config"] for c in "CDE"}}, "output_dir": "outputs/check"})
    final = {"base_config": "base.yaml", "common_alpha": .1, "selection": "selection.json",
        "inputs_sha256": {"selection.json": sha256(root / "selection.json")},
        "files_sha256": {path.relative_to(root).as_posix(): sha256(path)},
        "development_check": {"mode": "reuse_selected_development", "split": "dev_select",
            "independent_evaluation": False, "generate_dev_anal_responses": False,
            "analysis_plan": path.relative_to(root).as_posix()}}
    write(folder, final)
    calls = []
    monkeypatch.setattr(runner, "_require_outer_lock", lambda root: calls.append("lock"))
    def command(root, argv):
        calls.append(argv)
        write(root / "outputs/check/analysis.json", {"input_sha256": "verified-input"})
    monkeypatch.setattr(runner, "_command", command)
    monkeypatch.setattr(runner, "validate_rehearsal", lambda report, alpha, **kw: calls.append(kw))
    return root, calls, final


def test_only_cpu_analyzer_is_called_and_nonindependent_receipt_written(prepared):
    root, calls, _ = prepared
    result = runner.run_check(root)
    assert calls[0] == "lock"
    assert calls[1][0] == "scripts.analyze_budget120" and len(calls[1]) == 2
    assert calls[2] == {"reuse_selection": True}
    assert result["selection_data_reused"] and not result["independent_evaluation"]
    assert (root / "outputs/reproducibility/development_pipeline_check.json").exists()


def test_changed_selection_data_blocks_cpu_check(prepared):
    root, calls, _ = prepared
    (root / "selection.json").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        runner.run_check(root)
    assert calls == []


def test_nonselected_run_cannot_be_substituted(prepared):
    root, calls, final = prepared
    path = root / final["development_check"]["analysis_plan"]
    value = json.loads(path.read_text())
    value["run_configs"]["C"] = "configs/new_C.yaml"
    path.write_text(json.dumps(value))
    final["files_sha256"][path.relative_to(root).as_posix()] = sha256(path)
    (root / runner.FINAL).write_text(json.dumps(final))
    with pytest.raises(ValueError, match="reuse only"):
        runner.run_check(root)
    assert calls == []


def test_analysis_failure_cannot_write_pass_receipt(prepared, monkeypatch):
    root, _, _ = prepared
    def fail(*args):
        raise RuntimeError("analysis failed")
    monkeypatch.setattr(runner, "_command", fail)
    with pytest.raises(RuntimeError):
        runner.run_check(root)
    assert not (root / "outputs/reproducibility/development_pipeline_check.json").exists()
