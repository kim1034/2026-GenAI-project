"""Run the sealed four-condition test and final analysis under one stage lock."""

import json
from pathlib import Path

from scripts.budget_lock_preflight import BASE, FINAL, RECEIPT, _immutable, _path, _read, verify_lock
from scripts.finalize_budget_selection import _command, _require_outer_lock
from scripts.run_experiment import ROOT, sha256


def run_test(root):
    root = Path(root).resolve()
    seal = verify_lock(root)
    receipt = _read(root / RECEIPT)
    final_path = root / FINAL
    final = _read(final_path)
    expected_order = ("greedy", "C", "D", "E")
    registered = receipt["required_test_runs"]
    if (tuple(row["condition"] for row in registered) != expected_order
            or final["base_config"] != BASE
            or final["common_alpha"] != receipt["common_alpha"]):
        raise ValueError("Final test design differs from LOCK")
    commands = []
    for row in registered:
        condition, name = row["condition"], row["run_config"]
        if (final["run_configs"]["test"].get(condition) != name
                or sha256(_path(root, name)) != row["run_config_sha256"]
                or row["rows"] != (1 if condition == "greedy" else 16)
                or row["problem_count"] != 1319):
            raise ValueError("A registered test run changed")
        if condition != "greedy":
            commands.append(["-m", "scripts.build_bank", name])
        commands.append(["-m", "scripts.run_experiment", name])
    stage = _read(final_path.parent / "stage_test.json")
    analysis_path = final_path.parent / "test_analysis_plan.json"
    analysis = _read(analysis_path)
    if (stage["commands"] != commands or analysis["split"] != "test"
            or analysis["base_config"] != BASE or analysis["selection_file"] != final["selection"]
            or analysis["run_configs"] != final["run_configs"]["test"]
            or (analysis_path.parent / analysis["root"]).resolve() != root):
        raise ValueError("Final execution/analysis plan differs from LOCK")
    _require_outer_lock(root)
    for command in commands:
        _command(root, command[1:])
    # The analyzer verifies all four completed outputs before computing any statistic.
    _command(root, ["scripts.analyze_budget120", analysis_path.relative_to(root).as_posix()])
    output = _path(root, analysis["output_dir"])
    report = _read(output / "analysis.json")
    if (report["status"] != "complete" or report["scope"] != "confirmatory_test"
            or report["n"] != 1319 or report["common_alpha"] != receipt["common_alpha"]
            or report["inputs"]["LOCK_sha256"] != seal["LOCK_sha256"]):
        raise ValueError("Final analysis does not match the completed sealed test")
    files = {path.relative_to(root).as_posix(): sha256(path) for path in output.iterdir() if path.is_file()}
    result = {"schema_version": 1, "status": "test_and_analysis_complete",
              "protocol_amendment": "budget120-v1.4", "seal": seal,
              "analysis_artifacts_sha256": files, "input_sha256": report["input_sha256"],
              "gpu_return": "manual_user", "local_backup_verification": "pending"}
    _immutable(root / "outputs/reproducibility/final_test_receipt.json",
               (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


if __name__ == "__main__":
    print(json.dumps(run_test(ROOT), indent=2))
