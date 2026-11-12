"""Prepare frozen final run configs after verified C/D/E selection.

This command reads development evidence only and never launches inference.
The test stage starts with the runner's existing LOCK guard before model load.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.project_power import _load_run
from scripts.run_experiment import ROOT, _validate_run, run_id, sha256
from src.config import config_sha256, load_config


BASE_CONFIG = "configs/generated/budget120/pilot/base.yaml"


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _source_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Selection inputs must stay inside the project")
    rel = path.relative_to(root).as_posix()
    if (rel.startswith("outputs/samples/test_")
            or rel.startswith("data/processed/test.") or rel.startswith("data/raw/test.")):
        raise ValueError("Final preparation cannot read test questions, answers, or outputs")
    return path


def prepare(root, selection, destination):
    root, selection_path, destination = (Path(path).resolve() for path in (root, selection, destination))
    allowed = root / "configs/generated/budget120/final"
    if destination != allowed and not destination.is_relative_to(allowed):
        raise ValueError("Final snapshots require a dedicated budget120/final directory")
    selection_path = _source_path(root, selection_path)
    decision = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {"schema_version": 1, "status": "complete", "protocol_amendment": "budget120-v1.4",
                "split": "dev_select", "edge_checks_complete": True}
    if (any(decision.get(key) != value for key, value in required.items())
            or type(decision.get("schema_version")) is not int
            or decision.get("edge_checks_complete") is not True
            or set(decision.get("runs", {})) != {"A", "C", "D", "E"}):
        raise ValueError("Complete budget120 A/C/D/E selection with resolved edges is required")
    alpha = decision.get("common_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Selection requires a positive finite common alpha")
    base_path = root / BASE_CONFIG
    base = load_config(base_path)
    if (base.get("protocol_amendment") != "budget120-v1.4"
            or any(base["decode"][key] != 16 for key in ("k_generate_dev", "k_generate_test", "k_generate_rehearsal"))
            or base["data"]["splits"] != {"dev_base": 50, "dev_select": 200, "dev_anal": 50, "test": 1319}):
        raise ValueError("The unchanged budget120 base must preserve 16 rows and original split sizes")
    input_hashes = {}

    def record(path, expected=None):
        path = _source_path(root, path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Selection source SHA changed: {path.relative_to(root)}")
        input_hashes[path.relative_to(root).as_posix()] = digest
        return path

    if not isinstance(decision.get("inputs_sha256"), dict) or not decision["inputs_sha256"]:
        raise ValueError("Selection must preserve its input SHA map")
    for path, expected in decision["inputs_sha256"].items():
        record(path, expected)
    record(selection_path)
    record(base_path)
    analysis_path = record(base["stats"]["analysis_list"])
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    protocol_path = record(base_path.parent / "protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_amendment") != "budget120-v1.4":
        raise ValueError("Frozen protocol amendment does not match selection")
    policy_path = root / "configs/development_check_policy.json"
    policy = None
    if policy_path.exists():
        record(policy_path)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reuse_selected = policy is not None and policy.get("generate_dev_anal_responses") is False
    if reuse_selected and (policy.get("base_protocol") != "budget120-v1.4"
                          or policy.get("pipeline_check_split") != "dev_select"
                          or policy.get("reuse_only_final_selected_CDE_and_existing_greedy") is not True):
        raise ValueError("Development reuse policy must identify the selected dev_select evidence")
    source_runs, baseline = {}, None
    for condition in ("C", "D", "E"):
        reference = decision["runs"][condition]
        yaml_path = record(reference["run_config"], reference["run_config_sha256"])
        manifest_path = record(reference["manifest"], reference["manifest_sha256"])
        run = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if run.get("base_config") != BASE_CONFIG or run.get("id_version") != 2:
            raise ValueError("Selected runs must reuse the exact frozen pilot base and v2 IDs")
        provenance, grouped, _ = _load_run(root, root, root, yaml_path, manifest_path,
                                           condition, alpha, .5)
        if provenance["output_sha256"] != reference["output_sha256"]:
            raise ValueError("Selection raw output SHA differs from completed manifest")
        if baseline is not None:
            previous, previous_grouped = baseline
            if provenance["execution_signature"] != previous["execution_signature"] or list(grouped) != list(previous_grouped):
                raise ValueError("Selected C/D/E configuration, IDs, or runtime differ")
            if any(item["prompt_sha256"] != previous_grouped[pid][0]["prompt_sha256"]
                   for pid, items in grouped.items() for item in items):
                raise ValueError("Selected C/D/E problem prompts differ")
        else:
            baseline = provenance, grouped
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record(root / "outputs/samples" / f"{provenance['run_id']}.jsonl", reference["output_sha256"])
        record(manifest["bank_file"], provenance["bank_sha256"])
        record(_source_path(root, manifest["bank_file"]).with_suffix(".json"), provenance["bank_manifest_sha256"])
        source_runs[condition] = reference
    for value in (base["noise"]["rho_file"], base["prompt"]["fewshot_file"], base["prompt"]["system_file"],
                  "data/splits/dev_select.ids", "data/processed/train.jsonl"):
        record(value)
    check_split = "dev_select" if reuse_selected else "dev_anal"
    rehearsal_greedy = f"configs/runs/{check_split}_greedy.yaml"
    record(rehearsal_greedy)
    greedy_run = yaml.safe_load((root / rehearsal_greedy).read_text(encoding="utf-8"))
    if any(greedy_run.get(key) != value for key, value in {
            "split": check_split, "condition": "greedy", "n_rows": 1, "temperature": 0,
            "alpha": None, "seed_set": 0}.items()):
        raise ValueError(f"The existing {check_split} greedy label configuration is required")
    run_splits = [("test", ("greedy", "C", "D", "E"))]
    if not reuse_selected:
        run_splits.insert(0, ("dev_anal", ("C", "D", "E")))
    contents, names, configs = {}, set(), {split: {} for split, _ in run_splits}
    for split, conditions in run_splits:
        for condition in conditions:
            greedy = condition == "greedy"
            run = {"base_config": BASE_CONFIG, "split": split, "condition": condition,
                   "alpha": None if greedy else alpha, "temperature": 0.0 if greedy else .5,
                   "basis": None if greedy else "B0", "seed_set": 0, "row_start": 0,
                   "n_rows": 1 if greedy else 16, "id_version": 2}
            _validate_run(run, base)
            name = run_id(run)
            if name in names:
                raise ValueError("Final run IDs collide")
            names.add(name)
            path = destination / f"{name}.yaml"
            contents[path] = yaml.safe_dump(run, sort_keys=False).encode("utf-8")
            configs[split][condition] = path.relative_to(root).as_posix()
            manifest_path = root / "outputs/manifests" / f"{name}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (manifest.get("run_config_sha256") != hashlib.sha256(contents[path]).hexdigest()
                        or manifest.get("config_sha256") != config_sha256(base, analysis)):
                    raise ValueError(f"Existing final run ID has conflicting configuration: {name}")
            elif any(path.exists() for path in (root / "outputs/samples" / name,
                                                root / "outputs/samples" / f"{name}.jsonl")):
                raise ValueError(f"Existing final run ID has output without a manifest: {name}")
    expected_count = 4 if reuse_selected else 7
    if len(names) != expected_count:
        raise ValueError(f"Exactly {expected_count} final configurations are required")
    test = [["-m", "scripts.run_experiment", configs["test"]["greedy"]]]
    for condition in ("C", "D", "E"):
        test.extend([["-m", "scripts.build_bank", configs["test"][condition]],
                     ["-m", "scripts.run_experiment", configs["test"][condition]]])
    stages = [("test", test)]
    if not reuse_selected:
        rehearsal = [["-m", "scripts.build_bank", configs["dev_anal"][condition]] for condition in ("C", "D", "E")]
        rehearsal.extend(["-m", "scripts.run_experiment", configs["dev_anal"][condition]] for condition in ("C", "D", "E"))
        stages.insert(0, ("rehearsal", rehearsal))
    for name, commands in stages:
        contents[destination / f"stage_{name}.json"] = _json({
            "name": f"budget120_{name}", "preparation_only": True, "require_initial_probe": True,
            "requires_test_lock": name == "test", "commands": commands,
            "test_access_guard": "scripts.run_experiment -> src.config.assert_test_locked before model load" if name == "test" else None})
    analysis_plans = [("pipeline_check", "dev_select"), ("test", "test")] if reuse_selected else [("rehearsal", "dev_anal"), ("test", "test")]
    for name, split in analysis_plans:
        analysis_runs = {"greedy": configs["test"]["greedy"] if split == "test" else rehearsal_greedy,
                         **{condition: (source_runs[condition]["run_config"] if split == "dev_select"
                                        else configs[split][condition]) for condition in ("C", "D", "E")}}
        contents[destination / f"{name}_analysis_plan.json"] = _json({
            "root": "/".join([".."] * len(destination.relative_to(root).parts)), "split": split,
            "selection_file": selection_path.relative_to(root).as_posix(), "base_config": BASE_CONFIG,
            "run_configs": analysis_runs, "output_dir": f"outputs/analysis/budget120/{split}"})
    final = {"schema_version": 1, "protocol_amendment": "budget120-v1.4", "preparation_only": True,
             "selection": selection_path.relative_to(root).as_posix(), "selection_sha256": sha256(selection_path),
             "base_config": BASE_CONFIG, "base_config_sha256": sha256(base_path), "common_alpha": alpha,
             "protocol": protocol_path.relative_to(root).as_posix(), "protocol_sha256": sha256(protocol_path),
             "source_runs": source_runs, "run_configs": configs, "inputs_sha256": input_hashes,
             "files_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                              for path, data in contents.items()},
             "test_analysis_requires_all": ["C", "D", "E"]}
    if policy is not None:
        final["development_check_policy"] = policy_path.relative_to(root).as_posix()
        final["development_check_policy_sha256"] = sha256(policy_path)
    if reuse_selected:
        final["development_check"] = {
            "mode": "reuse_selected_development", "split": "dev_select",
            "analysis_plan": (destination / "pipeline_check_analysis_plan.json").relative_to(root).as_posix(),
            "independent_evaluation": False, "generate_dev_anal_responses": False}
    contents[destination / "final_plan.json"] = _json(final)
    if reuse_selected:
        stale = [destination / "stage_rehearsal.json", destination / "rehearsal_analysis_plan.json",
                 *destination.glob("dev_anal_*.yaml")]
        if any(path.exists() for path in stale):
            raise ValueError("Conflicting existing rehearsal files under development reuse policy")
    for path, data in contents.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Conflicting existing final snapshot: {path}")
    for path, data in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return {"run_count": len(names), "final_plan": (destination / "final_plan.json").relative_to(root).as_posix(),
            "files_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                             for path, data in contents.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/analysis/budget120_selection.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "configs/generated/budget120/final")
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.selection, args.destination), indent=2))


if __name__ == "__main__":
    main()
