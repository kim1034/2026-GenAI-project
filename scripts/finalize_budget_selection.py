"""Finalize the fixed v1.4 A+C3 selection, optional C edge and common-alpha D/E.

Default mode verifies/prepares only. --execute permits missing adaptive runs;
--run-initial explicitly permits the initial four runs and implies --execute.
GPU subprocesses require an already-held outer run_stage lock; this script
never starts a nested stage or acquires that lock for its own execution.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.make_pilot_grid import _different_paths, _fingerprint
from scripts.project_power import _load_run
from scripts.run_experiment import ROOT, _validate_run, run_id, sha256
from src.bank import bank_path
from src.config import load_config
from src.result_io import summarize_run
from src.select_alpha import accept_edge, edge_candidate, selection_decision


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Selection evidence must stay inside the project root")
    relative = path.relative_to(root).as_posix()
    if (relative.startswith("data/raw/") or relative.startswith("data/processed/test")
            or relative.startswith("data/splits/test")
            or any(part.startswith("test_") for part in path.relative_to(root).parts)):
        raise ValueError("Test/raw evidence is outside development selection scope")
    return path


def _rel(root, path):
    return path.relative_to(root).as_posix()


def _immutable(path, value, *, raw=False):
    data = value if raw else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable selection evidence conflicts with inputs/content: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def _pin(root, hashes, path):
    key, digest = _rel(root, path), sha256(path)
    if key in hashes and hashes[key] != digest:
        raise ValueError(f"Pinned selection input changed: {key}")
    hashes[key] = digest


def _check_previous_pins(root):
    for name in ("budget120_selection_initial.json", "budget120_selection_edge.json", "budget120_selection.json"):
        path = root / "outputs/analysis" / name
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            for relative, expected in previous["inputs_sha256"].items():
                if sha256(_path(root, relative)) != expected:
                    raise ValueError(f"Existing selection conflicts with changed pinned input: {relative}")


def _require_outer_lock(root):
    import fcntl
    with (root / "outputs/logs/stage.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle, fcntl.LOCK_UN)
    raise RuntimeError("GPU execution requires the existing outer run_stage lock")


def _command(root, argv):
    _require_outer_lock(root)
    subprocess.run([sys.executable, "-u", "-m", *argv], cwd=root, check=True)


def _other_bank_builders(root):
    proc = Path("/proc")
    if not proc.exists():
        return []
    found = []
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            command = (path / "cmdline").read_bytes()
            if ((b"scripts.build_bank\x00" in command or b"scripts.build_development_banks\x00" in command)
                    and (path / "cwd").resolve() == root):
                found.append(int(path.name))
        except (OSError, PermissionError):
            continue
    return found


def _ensure_bank(root, config_path, run, workers):
    if run["condition"] == "A":
        return
    path = bank_path(root, run["condition"], "B0", 0, "dev_select", 0, 16)
    waiting = False
    while not (path.is_file() and path.with_suffix(".json").is_file()):
        builders = _other_bank_builders(root)
        if builders:
            if not waiting:
                print(json.dumps({"status": "waiting_for_existing_bank_builder", "condition": run["condition"], "pids": builders}), flush=True)
                waiting = True
            time.sleep(5)
            continue
        if (path.exists() or path.with_suffix(".json").exists() or path.with_suffix(".partial.npy").exists()
                or path.with_suffix(".state.json").exists()):
            raise RuntimeError("Incomplete bank evidence requires explicit recovery; no second writer will be started")
        _command(root, ["scripts.build_bank", _rel(root, config_path), "--workers", str(workers)])
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise RuntimeError("Bank builder did not complete the required bank")


def _run_config(root, path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifest_path(root, run):
    return root / "outputs/manifests" / f"{run_id(run)}.json"


def _ensure_run(root, path, *, execute, workers, force=False):
    run = _run_config(root, path)
    manifest = _manifest_path(root, run)
    complete = manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete"
    if not complete or force:
        if not execute:
            return False
        _ensure_bank(root, path, run, workers)
        _command(root, ["scripts.run_experiment", _rel(root, path)])
    return True


def _load(root, path, condition, alpha, *, greedy=None, reference=None):
    run = _run_config(root, path)
    manifest = _manifest_path(root, run)
    item, grouped, data = _load_run(root, root, root, path, manifest, condition, alpha,
                                   0.0 if condition == "greedy" else .5, baseline=None)
    if condition == "greedy":
        return item, grouped, data
    if reference is not None:
        differences = _different_paths(reference["execution_signature"], item["execution_signature"])
        if differences:
            raise ValueError("Selected condition configuration/runtime differs from A: " + ", ".join(differences))
    if greedy is None:
        raise ValueError("Verified greedy labels/prompts are required")
    gold = {row["id"]: row["gold_number"] for row in data}
    summary, _, _ = summarize_run(grouped, greedy, gold)
    if summary["label_classes_complete"] is not True or summary["auroc"] is None:
        raise ValueError("Selection requires both greedy label classes")
    summary.update(alpha=alpha, condition=condition, split="dev_select", status="complete", run_id=item["run_id"])
    return item, summary


def _pin_run(root, hashes, path, item):
    run = _run_config(root, path)
    base_path = _path(root, run.get("base_config", "configs/base.yaml"))
    base = load_config(base_path)
    for target in (path, _manifest_path(root, run), root / "outputs/samples" / f'{item["run_id"]}.jsonl', base_path,
                   _path(root, base["stats"]["analysis_list"]), _path(root, base["noise"]["rho_file"]),
                   _path(root, base["prompt"]["system_file"]), _path(root, base["prompt"]["fewshot_file"]),
                   root / "data/splits/dev_select.ids", root / "data/processed/train.jsonl"):
        _pin(root, hashes, target)


def _reference(root, path, item):
    return {"run_config": _rel(root, path), "run_config_sha256": sha256(path),
            "manifest": _rel(root, _manifest_path(root, _run_config(root, path))),
            "manifest_sha256": item["manifest_sha256"], "output_sha256": item["output_sha256"]}


def _adaptive_config(root, plan_path, template, condition, alpha):
    run = {**template, "condition": condition, "alpha": alpha, "basis": "B0", "id_version": 2}
    _validate_run(run, load_config(_path(root, run["base_config"])))
    path = plan_path.parent / "adaptive" / f"{run_id(run)}.yaml"
    _immutable(path, yaml.safe_dump(run, sort_keys=False).encode("utf-8"), raw=True)
    return path


def finalize(root, selection_plan, *, execute=False, run_initial=False, bank_workers=8,
             greedy_config="configs/runs/dev_select_greedy.yaml"):
    root = Path(root).resolve()
    plan_path = _path(root, selection_plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("protocol_amendment") != "budget120-v1.4" or plan.get("split") != "dev_select"
            or plan.get("selected_indices_zero_based") != [0, 2, 5] or len(plan.get("original_grid", [])) != 6
            or plan.get("original_grid") != sorted(set(plan["original_grid"]))
            or plan.get("selected_grid") != [plan["original_grid"][i] for i in [0, 2, 5]]
            or len(plan.get("c_run_configs", [])) != 3 or type(bank_workers) is not int or bank_workers < 1):
        raise ValueError("Unsupported or incomplete fixed three-point selection plan")
    if run_initial and not execute:
        raise ValueError("run_initial requires explicit execution authorization")
    _check_previous_pins(root)  # Before even a hash-skipping initial GPU subprocess.
    hashes = dict(plan["inputs_sha256"])
    for name, expected in hashes.items():
        if sha256(_path(root, name)) != expected:
            raise ValueError(f"Pinned selection input changed: {name}")
    for key, hash_key in (("base_config", "base_config_sha256"), ("grid_decision", "grid_decision_sha256"),
                          ("pilot_plan", "pilot_plan_sha256")):
        if sha256(_path(root, plan[key])) != plan[hash_key]:
            raise ValueError(f"Frozen selection {key} SHA changed")
    grid_info = json.loads(_path(root, plan["grid_decision"]).read_text())["grid"]
    if (grid_info.get("grid") != plan["original_grid"] or grid_info.get("status") != "ok"
            or grid_info.get("launch_allowed") is not True
            or plan.get("edge_extrapolation") != {"source": "original_sorted_six_point_grid",
                "lower_ratio": plan["original_grid"][1] / plan["original_grid"][0],
                "upper_ratio": plan["original_grid"][-1] / plan["original_grid"][-2]}):
        raise ValueError("Selection subset/edge rule differs from the verified original six-point grid")
    _pin(root, hashes, plan_path)
    base = load_config(_path(root, plan["base_config"]))
    if (base.get("protocol_amendment") != "budget120-v1.4" or base["decode"]["k_generate_dev"] != 16
            or base["selection"]["max_acc_drop"] != .05 or base["selection"]["max_invalid_increase"] != .05
            or base["selection"]["tie_tolerance"] != .005):
        raise ValueError("Frozen v1.4 selection rules changed")
    paths = [_path(root, name) for name in [plan["a_run_config"], *plan["c_run_configs"]]]
    if len(set(paths)) != 4 or set(plan["run_configs_sha256"]) != {_rel(root, path) for path in paths}:
        raise ValueError("Initial selection must contain exactly four distinct pinned configurations")
    for i, path in enumerate(paths):
        if sha256(path) != plan["run_configs_sha256"][_rel(root, path)]:
            raise ValueError("Initial run configuration SHA changed")
        run = _run_config(root, path)
        expected = {"base_config": plan["base_config"], "condition": "A" if i == 0 else "C",
                    "alpha": None if i == 0 else plan["selected_grid"][i - 1], "split": "dev_select",
                    "temperature": .5, "n_rows": 16, "seed_set": 0, "row_start": 0, "id_version": 2,
                    "basis": None if i == 0 else "B0"}
        if any(run.get(k) != v for k, v in expected.items()):
            raise ValueError("Initial run does not match the frozen selection plan")
    if run_initial:
        for path in paths:
            _ensure_run(root, path, execute=True, workers=bank_workers, force=True)
    greedy_path = _path(root, greedy_config)
    gm, greedy, _ = _load(root, greedy_path, "greedy", None)
    _pin_run(root, hashes, greedy_path, gm)
    a, baseline = _load(root, paths[0], "A", None, greedy=greedy)
    if a["execution_signature"]["base"]["model"] != gm["execution_signature"]["base"]["model"]:
        raise ValueError("Selection model differs from the greedy label source")
    _pin_run(root, hashes, paths[0], a)
    entries = []
    for path, alpha in zip(paths[1:], plan["selected_grid"]):
        item, summary = _load(root, path, "C", alpha, greedy=greedy, reference=a)
        _pin_run(root, hashes, path, item)
        entries.append((path, item, summary))
    decision = selection_decision([entry[2] for entry in entries], baseline, base["selection"])
    chosen = next(entry for entry in entries if entry[2]["alpha"] == decision["selected_alpha"])
    candidate = edge_candidate(plan["original_grid"], decision["selected_alpha"], kind="relative")
    initial = {"schema_version": 1, "status": "complete", "protocol_amendment": "budget120-v1.4", "split": "dev_select",
               "source_six_point_grid": plan["original_grid"], "selected_three_point_grid": plan["selected_grid"],
               "baseline_A": baseline, **decision, "edge_candidate": candidate, "inputs_sha256": dict(hashes),
               "input_fingerprint_sha256": _fingerprint(hashes)}
    initial_path = root / "outputs/analysis/budget120_selection_initial.json"
    _immutable(initial_path, initial)
    _pin(root, hashes, initial_path)
    edge = {"schema_version": 1, "status": "complete", "edge_used": candidate is not None, "candidate_alpha": candidate,
            "reason": "not_original_grid_endpoint", "accepted": False, "initial_decision_sha256": sha256(initial_path)}
    if candidate is not None:
        edge_path = _adaptive_config(root, plan_path, _run_config(root, paths[1]), "C", candidate)
        if not _ensure_run(root, edge_path, execute=execute, workers=bank_workers):
            return {"status": "pending_runs", "initial_decision": _rel(root, initial_path), "pending_run_configs": [_rel(root, edge_path)]}
        item, summary = _load(root, edge_path, "C", candidate, greedy=greedy, reference=a)
        _pin_run(root, hashes, edge_path, item)
        accepted = accept_edge(chosen[2], summary, baseline_accuracy=baseline["maj_acc"], baseline_invalid=baseline["invalid"])
        edge.update(accepted=accepted, reason="accepted_strict_improvement" if accepted else "rejected_inadmissible_or_gain_not_strictly_above_0.005",
                    metrics=summary, run=_reference(root, edge_path, item))
        if accepted:
            chosen = edge_path, item, summary
    edge["inputs_sha256"] = dict(hashes)
    edge_evidence = root / "outputs/analysis/budget120_selection_edge.json"
    _immutable(edge_evidence, edge)
    _pin(root, hashes, edge_evidence)
    common = chosen[2]["alpha"]
    adaptive = {name: _adaptive_config(root, plan_path, _run_config(root, paths[1]), name, common) for name in ("D", "E")}
    runs = {"A": _reference(root, paths[0], a), "C": _reference(root, chosen[0], chosen[1])}
    for condition, path in adaptive.items():
        if not _ensure_run(root, path, execute=execute, workers=bank_workers):
            return {"status": "pending_runs", "common_alpha": common, "pending_run_configs": [_rel(root, p) for p in adaptive.values()]}
        item, _ = _load(root, path, condition, common, greedy=greedy, reference=a)
        _pin_run(root, hashes, path, item)
        runs[condition] = _reference(root, path, item)
    result = {"schema_version": 1, "status": "complete", "protocol_amendment": "budget120-v1.4", "split": "dev_select",
              "common_alpha": common, "T_best": .5, "Ap_is_A": True, "edge_checks_complete": True,
              "source_six_point_grid": plan["original_grid"], "selected_three_point_grid": plan["selected_grid"],
              "selected_metrics": chosen[2], "feasible_count": decision["feasible_count"],
              "feasible_count_scope": "initial_three_point_grid", "selection_reason": decision["reason"],
              "edge_used": candidate is not None, "edge_reason": edge["reason"], "runs": runs,
              "inputs_sha256": hashes, "input_fingerprint_sha256": _fingerprint(hashes)}
    _immutable(root / "outputs/analysis/budget120_selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-plan", default="configs/generated/budget120/selection/selection_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-initial", action="store_true")
    parser.add_argument("--bank-workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(finalize(ROOT, args.selection_plan, execute=args.execute or args.run_initial,
                              run_initial=args.run_initial, bank_workers=args.bank_workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
