"""Prepare the fixed A+C3 dev_select queue from fully revalidated pilot evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.make_pilot_grid import _fingerprint, make_pilot_grids
from scripts.run_experiment import ROOT, _validate_run, run_id, sha256
from src.config import config_sha256, load_config


INDICES = [0, 2, 5]
BASE_CONFIG = "configs/generated/budget120/pilot/base.yaml"


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _revalidate_pilot(root, grid_path, plan_path):
    """Re-read all nine completed runs with the existing CPU evidence validator."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan_path.parent / plan["root"]).resolve() != root:
        raise ValueError("Pilot plan points to a different project root")
    if set(plan["conditions"]) != {"C"} or len(plan["conditions"]["C"]["run_configs"]) != 8:
        raise ValueError("The full A+C8 pilot plan is required")
    if (root / plan["conditions"]["C"]["output"]).resolve() != grid_path:
        raise ValueError("Grid decision does not match the pilot plan output")
    decision = json.loads(grid_path.read_text(encoding="utf-8"))
    if decision.get("plan_sha256") != sha256(plan_path):
        raise ValueError("Pilot decision plan SHA does not match")
    fingerprint_input = {key: value for key, value in decision.items() if key not in ("input_sha256", "grid")}
    if decision.get("input_sha256") != _fingerprint(fingerprint_input):
        raise ValueError("Pilot decision input fingerprint does not match")
    verified = make_pilot_grids(plan_path)
    if verified.get(str(grid_path)) != decision:
        raise ValueError("Pilot decision differs from revalidated raw/config/runtime evidence")
    return decision, plan


def prepare(root, grid_decision, pilot_plan, destination):
    root = Path(root).resolve()
    grid_path, plan_path, destination = (Path(path).resolve() for path in (grid_decision, pilot_plan, destination))
    allowed = root / "configs/generated/budget120/selection"
    if destination != allowed and not destination.is_relative_to(allowed):
        raise ValueError("Selection snapshots require a dedicated budget120/selection directory")
    if not grid_path.is_relative_to(root) or not plan_path.is_relative_to(root):
        raise ValueError("Pilot inputs must be inside the project root")
    decision, pilot = _revalidate_pilot(root, grid_path, plan_path)
    grid_info = decision["grid"]
    original = grid_info["grid"]
    if (grid_info.get("launch_allowed") is not True or grid_info.get("status") != "ok"
            or len(original) != 6 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                        or not math.isfinite(x) or x <= 0 for x in original)
            or original != sorted(set(original))):
        raise ValueError("A complete, sorted, distinct, launch-allowed six-point pilot grid is required")
    selected = [original[index] for index in INDICES]
    base_path = root / BASE_CONFIG
    base = load_config(base_path)
    if (base.get("protocol_amendment") != "budget120-v1.4"
            or base["decode"]["k_generate_dev"] != 16 or base["data"]["splits"]["dev_select"] != 200):
        raise ValueError("The frozen budget120 pilot base with dev_select 200x16 is required")
    evidence = [decision["inputs"]["A"], *decision["inputs"]["pilot_runs"]]
    if len(evidence) != 9 or any(item.get("base_yaml_sha256") != sha256(base_path) for item in evidence):
        raise ValueError("All nine pilot runs must identify the unchanged frozen base SHA")
    analysis_path = root / base["stats"]["analysis_list"]
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    inputs = [grid_path, plan_path, base_path, analysis_path, root / base["noise"]["rho_file"]]
    inputs.extend(root / base["prompt"][key] for key in ("system_file", "fewshot_file"))
    ids_path = root / base["data"]["splits_dir"] / "dev_select.ids"
    ids = ids_path.read_text(encoding="ascii").splitlines()
    if len(ids) != 200 or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("The original 200 unique dev_select IDs are required")
    ids_rel = ids_path.relative_to(root).as_posix()
    if pilot.get("inputs_sha256", {}).get(ids_rel) != sha256(ids_path):
        raise ValueError("dev_select IDs differ from the frozen pilot plan")
    inputs.append(ids_path)
    input_hashes = {}
    for path in inputs:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Selection inputs must be inside the project root")
        input_hashes[path.relative_to(root).as_posix()] = sha256(path)
    contents, configs, names = {}, [], set()
    for condition, magnitudes in [("A", [None]), ("C", selected)]:
        for alpha in magnitudes:
            run = {"base_config": BASE_CONFIG, "split": "dev_select", "condition": condition,
                   "alpha": alpha, "temperature": .5, "basis": None if condition == "A" else "B0",
                   "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
            _validate_run(run, base)
            name = run_id(run)
            if name in names:
                raise ValueError("Selection contains colliding run IDs")
            names.add(name)
            path = destination / f"{name}.yaml"
            contents[path] = yaml.safe_dump(run, sort_keys=False).encode("utf-8")
            configs.append(path.relative_to(root).as_posix())
            manifest_path = root / "outputs/manifests" / f"{name}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (manifest.get("run_config_sha256") != hashlib.sha256(contents[path]).hexdigest()
                        or manifest.get("config_sha256") != config_sha256(base, analysis)):
                    raise ValueError(f"Existing selection output has conflicting configuration: {name}")
    selection = {
        "protocol_amendment": "budget120-v1.4", "preparation_only": True,
        "split": "dev_select", "base_config": BASE_CONFIG, "base_config_sha256": sha256(base_path),
        "grid_decision": grid_path.relative_to(root).as_posix(), "grid_decision_sha256": sha256(grid_path),
        "pilot_plan": plan_path.relative_to(root).as_posix(), "pilot_plan_sha256": sha256(plan_path),
        "original_grid": original, "selected_grid": selected, "selected_indices_zero_based": INDICES,
        "edge_extrapolation": {"source": "original_sorted_six_point_grid",
                               "lower_ratio": original[1] / original[0],
                               "upper_ratio": original[-1] / original[-2]},
        "a_run_config": configs[0], "c_run_configs": configs[1:], "inputs_sha256": input_hashes,
        "run_configs_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                               for path, data in contents.items()},
    }
    contents[destination / "selection_plan.json"] = _json(selection)
    commands = [["-m", "scripts.build_bank", configs[1]]]
    commands.extend(["-m", "scripts.run_experiment", path] for path in configs)
    contents[destination / "stage_plan.json"] = _json({
        "name": "budget120_selection", "preparation_only": True, "require_initial_probe": True,
        "commands": commands, "inputs_sha256": input_hashes})
    for path, data in contents.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Conflicting existing selection snapshot: {path}")
    for path, data in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return {"run_count": len(names), "selection_plan": str((destination / "selection_plan.json").relative_to(root)),
            "stage_plan": str((destination / "stage_plan.json").relative_to(root)),
            "files_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                             for path, data in contents.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-decision", type=Path, default=ROOT / "outputs/analysis/budget120_pilot_C_grid.json")
    parser.add_argument("--pilot-plan", type=Path, default=ROOT / "configs/generated/budget120/pilot/grid_plan.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "configs/generated/budget120/selection")
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.grid_decision, args.pilot_plan, args.destination), indent=2))


if __name__ == "__main__":
    main()
