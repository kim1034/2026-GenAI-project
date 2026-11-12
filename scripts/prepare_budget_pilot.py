"""Freeze the amended A+C8 pilot and command queue; never execute model work."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from scripts.run_experiment import ROOT, run_id, sha256
from src.config import config_sha256, load_config


def _yaml(value):
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def prepare(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    legacy = root / "configs/generated/pilot"
    if (not destination.is_relative_to(root / "configs/generated/budget120")
            or destination == root / "configs/generated/budget120"
            or destination.is_relative_to(legacy)):
        raise ValueError("Budget pilot snapshots require a dedicated configs/generated/budget120 directory")
    base_path = root / "configs/base.yaml"
    base = load_config(base_path)
    rho_path = root / base["noise"]["rho_file"]
    analysis_path = root / base["stats"]["analysis_list"]
    protocol_path = root / "configs/budget120_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_amendment"] != "budget120-v1.4":
        raise ValueError("Unexpected budget amendment")
    if base["data"]["splits"] != {"dev_base": 50, "dev_select": 200, "dev_anal": 50, "test": 1319}:
        raise ValueError("The original data split sizes must remain fixed")
    if len(base["noise"]["pilot_grid_C"]) != 8:
        raise ValueError("The original eight-point C pilot is required")
    original_analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    if original_analysis["primary"]["comparison"] != ["D", "C"]:
        raise ValueError("The original D-C primary comparison is required")
    # Preserve registered random keys. No E-C key is invented by this amendment.
    analysis = copy.deepcopy(original_analysis)
    analysis["protocol_amendment"] = protocol["protocol_amendment"]
    analysis["secondary"] = {}
    analysis["descriptive"] = {
        "D_minus_E": copy.deepcopy(original_analysis["secondary"]["D_minus_E"]),
        "E_minus_C": {"conditions": ["E", "C"], "bootstrap_key": None},
    }
    analysis["original_secondary_registry"] = copy.deepcopy(original_analysis["secondary"])
    analysis["claim_original_holm_family_complete"] = False
    rel = destination.relative_to(root).as_posix()
    snapshot = copy.deepcopy(base)
    snapshot["protocol_amendment"] = protocol["protocol_amendment"]
    for key in ("k_generate_test", "k_generate_rehearsal", "k_generate_dev"):
        snapshot["decode"][key] = 16
    snapshot["noise"]["rho_file"] = f"{rel}/rho.json"
    snapshot["stats"]["analysis_list"] = f"{rel}/analysis.yaml"
    # Explicit active analysis scope; retain all underlying noise definitions.
    snapshot["stats"]["holm_family"] = []
    snapshot["stats"]["secondary_correction"] = None
    snapshot["stats"]["verdict_report"] = ["D-C"]
    snapshot["stats"]["descriptive_only"] = ["D-E", "E-C"]
    inputs = [base_path, rho_path, analysis_path, protocol_path]
    inputs.extend(root / base["prompt"][key] for key in ("system_file", "fewshot_file"))
    inputs.extend(root / base["data"]["splits_dir"] / f"{split}.ids" for split in base["data"]["splits"])
    input_hashes = {}
    for path in inputs:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Input lies outside project: {path}")
        input_hashes[resolved.relative_to(root).as_posix()] = sha256(resolved)
    contents = {
        destination / "base.yaml": _yaml(snapshot),
        destination / "analysis.yaml": _yaml(analysis),
        destination / "rho.json": rho_path.read_bytes(),
        destination / "protocol.json": protocol_path.read_bytes(),
    }
    configs, names = [], set()
    for condition, magnitudes in [("A", [None]), ("C", base["noise"]["pilot_grid_C"])]:
        for alpha in magnitudes:
            run = {"base_config": f"{rel}/base.yaml", "split": "dev_base", "condition": condition,
                   "alpha": alpha, "temperature": .5, "basis": None if condition == "A" else "B0",
                   "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
            name = run_id(run)
            if name in names:
                raise ValueError("Budget pilot contains colliding run IDs")
            names.add(name)
            path = destination / f"{name}.yaml"
            contents[path] = _yaml(run)
            configs.append(path.relative_to(root).as_posix())
            manifest_path = root / "outputs/manifests" / f"{name}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (manifest.get("run_config_sha256") != hashlib.sha256(contents[path]).hexdigest()
                        or manifest.get("config_sha256") != config_sha256(snapshot, analysis)):
                    raise ValueError(f"Existing output has conflicting configuration: {name}")
    if len(names) != 9:
        raise ValueError("Expected exactly nine A+C pilot configurations")
    plan = {"root": "/".join([".."] * len(destination.relative_to(root).parts)),
            "a_run_config": configs[0],
            "conditions": {"C": {"expected_alphas": base["noise"]["pilot_grid_C"],
                                   "run_configs": configs[1:],
                                   "output": "outputs/analysis/budget120_pilot_C_grid.json"}},
            "protocol_amendment": protocol["protocol_amendment"],
            "preparation_only": True, "inputs_sha256": input_hashes}
    contents[destination / "grid_plan.json"] = _json(plan)
    commands = [["-m", "scripts.build_bank", configs[1]]]
    commands.extend(["-m", "scripts.run_experiment", path] for path in configs)
    commands.append(["-m", "scripts.make_pilot_grid", "--plan", f"{rel}/grid_plan.json"])
    contents[destination / "stage_plan.json"] = _json({
        "name": "budget120_pilot", "require_initial_probe": True,
        "preparation_only": True, "commands": commands, "inputs_sha256": input_hashes})
    # Preflight the entire batch before creating any file, including on resume.
    for path, data in contents.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Conflicting existing budget pilot snapshot: {path}")
    for path, data in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return {"run_count": len(names), "grid_plan": f"{rel}/grid_plan.json",
            "stage_plan": f"{rel}/stage_plan.json",
            "files_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                             for path, data in contents.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "configs/generated/budget120/pilot")
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.destination), indent=2))


if __name__ == "__main__":
    main()
