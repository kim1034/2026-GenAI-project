"""Verify development evidence before creating the V1.4 test LOCK.

This script never launches inference or reads test question/answer records.
--write-lock creates LOCK.md only; committing it and tagging lock-v1 are
explicit subsequent operations. --verify-lock checks the completed seal.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.analyze_budget120 import analyze_budget120
from scripts.prepare_budget_final_runs import prepare
from scripts.run_experiment import ROOT, run_id, sha256
from src.config import assert_test_locked, config_sha256, load_config, verify_data_source


FINAL = "configs/generated/budget120/final/final_plan.json"
BASE = "configs/generated/budget120/pilot/base.yaml"
RECEIPT = "outputs/lock_preflight.json"


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("LOCK evidence escapes the project")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def _tracked_evidence(root):
    # Git's line-oriented output quotes Korean filenames and escapes spaces/newlines.
    # NUL-separated output preserves the actual filenames for byte hashing.
    return {name: sha256(_path(root, name)) for name in _git(root, "ls-files", "-z").split("\0") if name}


def _immutable(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing LOCK evidence conflicts: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def validate_rehearsal(report, alpha, *, reuse_selection=False):
    if any(report.get(k) != v for k, v in {
        "schema_version": 1, "status": "complete", "protocol_amendment": "budget120-v1.4",
        "scope": "development_pipeline_check" if reuse_selection else "rehearsal",
        "confirmatory": False, "split": "dev_select" if reuse_selection else "dev_anal",
        "n": 200 if reuse_selection else 50,
        "common_alpha": alpha, "practical_delta": .02, "analysis_unavailable_reason": None}.items()):
        raise ValueError("Development rehearsal has not passed the registered analysis")
    if reuse_selection and (report.get("selection_data_reused") is not True
                            or report.get("independent_evaluation") is not False
                            or report["PR6"]["outcome"] is not None):
        raise ValueError("Reused selection data must not be reported as independent evaluation")
    expected = {"D-C": [10, 1], "D-E": [10, 2], "E-C": None}
    bootstrap = report["bootstrap"]
    if (bootstrap["keys"] != expected or bootstrap["attempts_per_registered_key"] != 10000
            or bootstrap["global_seed"] != 2026 or bootstrap["se_ddof"] != 1
            or bootstrap["ci"] != "percentile_95"):
        raise ValueError("Rehearsal bootstrap registry differs")
    for name, key in expected.items():
        comparison = report["comparisons"][name]
        if comparison["rng_key"] != key or comparison["confirmatory"]:
            raise ValueError("Rehearsal comparison scope/key mismatch")
        if key is None:
            if any(comparison[k] is not None for k in ("ci95", "se", "verdict")):
                raise ValueError("Unregistered E-C inference must remain absent")
        elif (comparison["attempted"] != 10000 or comparison["valid"] < 2
              or comparison["valid"] + comparison["skipped"] != 10000
              or comparison["ci95"] is None or comparison["se"] is None):
            raise ValueError("Registered bootstrap did not complete")
    if report["arithmetic_residual"] is None or abs(report["arithmetic_residual"]) > 1e-14:
        raise ValueError("D-C decomposition failed")
    for condition in ("C", "D", "E"):
        metrics = report["metrics"][condition]
        curve = [row for row in report["k_curve"] if row["condition"] == condition and row["K"] == 8]
        if (not metrics["label_classes_complete"] or len(metrics["auroc_subsets"]) != 2
                or len(curve) != 1 or abs(curve[0]["auroc"] - metrics["auroc"]) > 1e-14):
            raise ValueError("Two-subset K=8 estimand was not verified")


def validate_test_plan(root, final):
    for folder in ("outputs/manifests", "outputs/samples"):
        if any((root / folder).glob("test_*")):
            raise ValueError("Test output already exists before LOCK")
    configs = final["run_configs"]["test"]
    if set(configs) != {"greedy", "C", "D", "E"}:
        raise ValueError("Test requires exactly greedy/C/D/E")
    required, commands = [], []
    for condition in ("greedy", "C", "D", "E"):
        name = configs[condition]
        path = _path(root, name)
        run = yaml.safe_load(path.read_text(encoding="utf-8"))
        greedy = condition == "greedy"
        expected = {"base_config": BASE, "split": "test", "condition": condition,
                    "alpha": None if greedy else final["common_alpha"], "temperature": 0.0 if greedy else .5,
                    "basis": None if greedy else "B0", "seed_set": 0, "row_start": 0,
                    "n_rows": 1 if greedy else 16, "id_version": 2}
        if run != expected:
            raise ValueError("Test YAML differs from the selected frozen design")
        rid = run_id(run)
        for output in (root / "outputs/manifests" / f"{rid}.json",
                       root / "outputs/samples" / f"{rid}.jsonl", root / "outputs/samples" / rid):
            if output.exists():
                raise ValueError("Test output already exists before LOCK: " + rid)
        if not greedy:
            commands.append(["-m", "scripts.build_bank", name])
        commands.append(["-m", "scripts.run_experiment", name])
        required.append({"condition": condition, "run_id": rid, "run_config": name,
                         "run_config_sha256": sha256(path), "rows": expected["n_rows"], "problem_count": 1319})
    stage = _read(_path(root, FINAL).parent / "stage_test.json")
    if stage["commands"] != commands:
        raise ValueError("Test execution order differs from greedy/C/D/E")
    return required


def validate_pipeline_receipt(receipt, report, *, final_sha, plan_sha, artifact_sha):
    expected = {"schema_version": 1, "status": "passed_development_pipeline_check",
                "scope": "development_pipeline_check", "selection_data_reused": True,
                "independent_evaluation": False, "separate_CDE_rehearsal_omitted": True,
                "final_plan_sha256": final_sha, "analysis_plan_sha256": plan_sha,
                "analysis_input_sha256": report["input_sha256"], "artifacts_sha256": artifact_sha}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("CPU pipeline-check receipt differs from verified analysis evidence")


def preflight(root):
    root = Path(root).resolve()
    if (root / "LOCK.md").exists():
        raise ValueError("LOCK already exists; use --verify-lock")
    if _git(root, "status", "--porcelain"):
        raise ValueError("Commit all reviewed code/configuration before LOCK preflight")
    final_path = root / FINAL
    final = _read(final_path)
    prepare(root, root / final["selection"], final_path.parent)
    required = validate_test_plan(root, final)
    base = load_config(root / BASE)
    analysis_path = _path(root, base["stats"]["analysis_list"])
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    protocol_path = root / "configs/budget120_protocol.json"
    if sha256(protocol_path) != sha256((root / BASE).parent / "protocol.json"):
        raise ValueError("Current protocol differs from the frozen pilot protocol")
    if (analysis["primary"] != {"comparison": ["D", "C"], "bootstrap_key": [10, 1]}
            or analysis["bootstrap_repeats"] != 10000 or analysis["practical_delta"] != .02
            or analysis["descriptive"]["D_minus_E"]["bootstrap_key"] != [10, 2]
            or analysis["descriptive"]["E_minus_C"]["bootstrap_key"] is not None):
        raise ValueError("Snapshot analysis registry differs from the registered design")
    verify_data_source(base, root)  # Byte/blob checks only, including the sealed source.
    subprocess.run([sys.executable, "-m", "scripts.pilot_preflight", "--budget-v14"], cwd=root, check=True)
    development = final.get("development_check", {})
    reuse_selection = development.get("mode") == "reuse_selected_development"
    if reuse_selection:
        policy = _read(root / "configs/development_check_policy.json")
        if (policy["generate_dev_anal_responses"] is not False
                or policy["pipeline_check_split"] != "dev_select"
                or development["split"] != "dev_select"
                or development["independent_evaluation"] is not False):
            raise ValueError("Development pipeline-check amendment mismatch")
        check_plan = _path(root, development["analysis_plan"])
    else:
        if (root / "configs/development_check_policy.json").is_file():
            raise ValueError("Final plan does not honor the development-check amendment")
        check_plan = final_path.parent / "rehearsal_analysis_plan.json"
    report = analyze_budget120(check_plan)
    validate_rehearsal(report, final["common_alpha"], reuse_selection=reuse_selection)
    evidence = dict(final["inputs_sha256"])
    evidence.update(final["files_sha256"])
    for name in (FINAL, BASE, base["stats"]["analysis_list"], "configs/budget120_protocol.json",
                 "data/raw/SOURCE.json", "data/raw/SHA256SUMS", "data/processed/train.jsonl",
                 "data/processed/test.jsonl", "data/splits/SHA256SUMS", "reports/split_eol_normalization.json",
                 "reports/extraction_audit.json", "models/SHA256SUMS", "models/files.json",
                 "outputs/diagnostics/gpu_validation_v14.json", "outputs/diagnostics/dense_gue_n100.json",
                 "outputs/diagnostics/manipulation_dev_base.json", "outputs/diagnostics/runtime.json",
                 "outputs/diagnostics/gpu_probe.json", "calib/rho.json"):
        evidence[name] = sha256(_path(root, name))
    normalization = _read(root / "reports/split_eol_normalization.json")
    for entry in normalization["entries"]:
        current = root / "data/splits" / entry["file"]
        original = root / "outputs/reproducibility/split_eol_original" / (entry["file"] + "." + entry["original_byte_sha256"])
        if (sha256(current) != entry["canonical_lf_sha256"] or sha256(original) != entry["original_byte_sha256"]
                or current.read_text(encoding="ascii").splitlines() != original.read_text(encoding="ascii").splitlines()):
            raise ValueError("Original ID sequence preservation proof failed")
        evidence[current.relative_to(root).as_posix()] = sha256(current)
        evidence[original.relative_to(root).as_posix()] = sha256(original)
    analysis_dir = root / ("outputs/analysis/budget120/dev_select" if reuse_selection
                           else "outputs/analysis/budget120/dev_anal")
    for path in analysis_dir.iterdir():
        if path.is_file():
            evidence[path.relative_to(root).as_posix()] = sha256(path)
    if reuse_selection:
        cpu_receipt = root / "outputs/reproducibility/development_pipeline_check.json"
        validate_pipeline_receipt(_read(cpu_receipt), report, final_sha=sha256(final_path),
                                  plan_sha=sha256(check_plan), artifact_sha={
                                      p.relative_to(root).as_posix(): sha256(p)
                                      for p in analysis_dir.iterdir() if p.is_file()})
        evidence[cpu_receipt.relative_to(root).as_posix()] = sha256(cpu_receipt)
    validation = _read(root / "outputs/diagnostics/gpu_validation_v14.json")
    for check in validation["checks"]:
        if check.get("raw_file"):
            name = check["raw_file"]
            if sha256(_path(root, name)) != check["raw_file_sha256"]:
                raise ValueError("GPU repeat raw evidence changed")
            evidence[name] = check["raw_file_sha256"]
    # Capture every tracked source/config/doc byte at the reviewed commit.
    evidence.update(_tracked_evidence(root))
    if _git(root, "status", "--porcelain"):
        raise ValueError("Preflight unexpectedly changed tracked files")
    return {"schema_version": 1, "status": "ready_for_lock", "protocol_amendment": "budget120-v1.4",
            "reviewed_code_commit": _git(root, "rev-parse", "HEAD"), "common_alpha": final["common_alpha"],
            "base_config": BASE, "analysis_config": base["stats"]["analysis_list"],
            "config_sha256": config_sha256(base, analysis), "required_test_runs": required,
            "evidence_sha256": evidence, "automatic_time_limit_stop": False, "gpu_return": "manual_user",
            "reviewer_type": "AI", "human_signatures": "not_provided",
            "original_human_signoff_requirement_satisfied": False,
            "review_scope": "User-delegated AI execution and review; no claim of two-person human signoff",
            "development_check": {"scope": report["scope"], "split": report["split"], "n": report["n"],
                                  "independent_evaluation": not reuse_selection,
                                  "separate_CDE_rehearsal_omitted": reuse_selection,
                                  "dev_anal_greedy_and_gradients_previously_completed": True},
            "gates": {"G1": "computational_checks_1_and_16_rows", "G2": "required_CDE_checks_passed",
                      "G3": "selected_development_pipeline_reproduced" if reuse_selection else "rehearsal_analysis_reproduced",
                      "G4": "mechanical_LOCK_pending; human_signatures_not_provided"}}


def lock_document(receipt, receipt_sha):
    """Canonical LOCK bytes, reproducible from the pinned preflight receipt."""
    content = "# V1.4 experiment LOCK\n\n" + "\n".join([
        "config_sha256: " + receipt["config_sha256"],
        "preflight_sha256: " + receipt_sha,
        "reviewed_code_commit: " + receipt["reviewed_code_commit"],
        "common_alpha: " + str(receipt["common_alpha"]),
        "reviewer_type: AI", "human_signatures: not_provided", "gpu_return: manual_user",
        "automatic_time_limit_stop: false", "",
        "V1.4 overrides dormant original boost/budget/32-row fields. Only greedy/C/D/E test runs below are authorized.",
        "D-C is confirmatory; D-E descriptive; E-C point only. No re-selection after rehearsal.",
        "Development check: " + json.dumps(receipt["development_check"], sort_keys=True),
        "When selection data are reused, their performance is selection-biased and is not independent validation.",
        "Original two-person human extraction/LOCK signatures were not provided; AI review is not human signoff.",
        "", "```json", json.dumps(receipt["required_test_runs"], indent=2), "```", ""])
    return content.encode("utf-8")


def verify_lock(root):
    root = Path(root).resolve()
    receipt_path = root / RECEIPT
    receipt = _read(receipt_path)
    content = (root / "LOCK.md").read_text(encoding="utf-8")
    if (receipt["status"] != "ready_for_lock" or "preflight_sha256: " + sha256(receipt_path) not in content):
        raise ValueError("LOCK does not pin the verified preflight receipt")
    for name, expected in receipt["evidence_sha256"].items():
        if sha256(_path(root, name)) != expected:
            raise ValueError("Locked evidence changed: " + name)
    if (root / "LOCK.md").read_bytes() != lock_document(receipt, sha256(receipt_path)):
        raise ValueError("LOCK content differs from the canonical preflight evidence")
    assert_test_locked(load_config(root / receipt["base_config"]), root / "LOCK.md")
    return {"status": "verified_test_seal", "lock_commit": _git(root, "rev-parse", "HEAD"),
            "LOCK_sha256": sha256(root / "LOCK.md"), "preflight_sha256": sha256(receipt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write-lock", action="store_true")
    modes.add_argument("--verify-lock", action="store_true")
    args = parser.parse_args()
    if args.verify_lock:
        result = verify_lock(ROOT)
        _immutable(ROOT / "outputs/lock_verified.json", (json.dumps(result, indent=2) + "\n").encode())
    else:
        result = preflight(ROOT)
        raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _immutable(ROOT / RECEIPT, raw)
        if args.write_lock:
            _immutable(ROOT / "LOCK.md", lock_document(result, sha256(ROOT / RECEIPT)))
    print(json.dumps({key: result[key] for key in ("status",) if key in result}))


if __name__ == "__main__":
    main()
