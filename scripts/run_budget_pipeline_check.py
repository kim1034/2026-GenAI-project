"""Check final analysis on completed selection outputs without GPU inference."""

import json
from pathlib import Path

from scripts.budget_lock_preflight import FINAL, _immutable, _path, _read, validate_rehearsal
from scripts.finalize_budget_selection import _command, _require_outer_lock
from scripts.run_experiment import ROOT, sha256


def run_check(root):
    root = Path(root).resolve()
    final_path = root / FINAL
    final = _read(final_path)
    development = final["development_check"]
    if (development["mode"] != "reuse_selected_development" or development["split"] != "dev_select"
            or development["independent_evaluation"] is not False
            or development["generate_dev_anal_responses"] is not False):
        raise ValueError("Expected CPU-only reuse of completed selection data")
    for name, digest in {**final["inputs_sha256"], **final["files_sha256"]}.items():
        if sha256(_path(root, name)) != digest:
            raise ValueError("Frozen development evidence changed: " + name)
    plan_path = _path(root, development["analysis_plan"])
    plan = _read(plan_path)
    selection = _read(_path(root, final["selection"]))
    expected = {"greedy": "configs/runs/dev_select_greedy.yaml",
                **{c: selection["runs"][c]["run_config"] for c in ("C", "D", "E")}}
    if (plan["split"] != "dev_select" or plan["run_configs"] != expected
            or plan["selection_file"] != final["selection"] or plan["base_config"] != final["base_config"]
            or (plan_path.parent / plan["root"]).resolve() != root):
        raise ValueError("Pipeline check must reuse only final selected runs and original greedy labels")
    _require_outer_lock(root)
    _command(root, ["scripts.analyze_budget120", plan_path.relative_to(root).as_posix()])
    folder = _path(root, plan["output_dir"])
    report = _read(folder / "analysis.json")
    validate_rehearsal(report, final["common_alpha"], reuse_selection=True)
    result = {"schema_version": 1, "status": "passed_development_pipeline_check",
              "scope": "development_pipeline_check", "selection_data_reused": True,
              "independent_evaluation": False, "separate_CDE_rehearsal_omitted": True,
              "final_plan_sha256": sha256(final_path), "analysis_plan_sha256": sha256(plan_path),
              "analysis_input_sha256": report["input_sha256"],
              "artifacts_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in folder.iterdir() if p.is_file()}}
    _immutable(root / "outputs/reproducibility/development_pipeline_check.json",
               (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


if __name__ == "__main__":
    print(json.dumps(run_check(ROOT), indent=2))
