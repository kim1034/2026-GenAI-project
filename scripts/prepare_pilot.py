"""Prepare immutable configurations for the existing 15-run pilot; never launch GPU work."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from scripts.run_experiment import ROOT, run_id, sha256
from src.config import load_config


def prepare(root, destination):
    root, destination = root.resolve(), destination.resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Pilot snapshots must be inside a dedicated project directory")
    base_path = root / "configs/base.yaml"
    base = load_config(base_path)
    sparse_path = root / "calib/sparse_decision.json"
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    if sparse.get("status") != "complete" or sparse.get("p") not in base["noise"]["sparse_rule"]["candidates"]:
        raise ValueError("A completed, prespecified sparsity decision is required")
    if sparse.get("base_config_sha256") != sha256(base_path) or not sparse.get("inputs_sha256"):
        raise ValueError("Sparsity decision does not identify current base settings and inputs")
    for name, expected in sparse["inputs_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Sparsity decision input changed: {name}")
    rho_path = root / base["noise"]["rho_file"]
    analysis_path = root / base["stats"]["analysis_list"]
    rel = destination.relative_to(root).as_posix()
    snapshot = copy.deepcopy(base)
    snapshot["noise"]["conditions"]["S"]["p"] = sparse["p"]
    snapshot["noise"]["rho_file"] = f"{rel}/rho.json"
    snapshot["stats"]["analysis_list"] = f"{rel}/analysis.yaml"
    contents = {
        destination / "base.yaml": yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True).encode("utf-8"),
        destination / "rho.json": rho_path.read_bytes(), destination / "analysis.yaml": analysis_path.read_bytes(),
        destination / "sparse_decision.json": sparse_path.read_bytes(),
    }
    configs = {"C": [], "S": []}
    names, a_path = set(), None
    for condition, magnitudes in [("A", [None]), ("C", base["noise"]["pilot_grid_C"]), ("S", base["noise"]["pilot_grid_S"])]:
        for alpha in magnitudes:
            run = {"base_config": f"{rel}/base.yaml", "split": "dev_base", "condition": condition,
                   "alpha": alpha, "temperature": .5, "basis": None if condition == "A" else "B0",
                   "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
            name = run_id(run)
            if name in names:
                raise ValueError("Pilot contains colliding run IDs")
            names.add(name)
            path = destination / f"{name}.yaml"
            contents[path] = yaml.safe_dump(run, sort_keys=False).encode("utf-8")
            if condition == "A":
                a_path = path.relative_to(root).as_posix()
            else:
                configs[condition].append(path.relative_to(root).as_posix())
    if len(names) != 15:
        raise ValueError("Expected exactly 15 prespecified pilot configurations")
    plan = {"root": "/".join([".."] * len(destination.relative_to(root).parts)), "a_run_config": a_path,
            "sparse_decision": {"path": f"{rel}/sparse_decision.json", "sha256": sha256(sparse_path)},
            "conditions": {c: {"expected_alphas": base["noise"][f"pilot_grid_{c}"], "run_configs": configs[c],
                                "output": f"outputs/analysis/pilot_{c}_grid.json"} for c in ("C", "S")},
            "preparation_only": True,
            "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in (base_path, rho_path, analysis_path, sparse_path)}}
    contents[destination / "grid_plan.json"] = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Verify the entire batch before writing any new file.
    for path, data in contents.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Conflicting existing pilot snapshot: {path}")
    for path, data in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return {"run_count": len(names), "grid_plan": str((destination / "grid_plan.json").relative_to(root)),
            "files_sha256": {str(path.relative_to(root)): hashlib.sha256(data).hexdigest() for path, data in contents.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "configs/generated/pilot")
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.destination), indent=2))


if __name__ == "__main__":
    main()
