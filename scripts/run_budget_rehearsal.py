"""Execute the pinned development rehearsal under the existing stage lock."""

import argparse
import json
from pathlib import Path

import yaml

from scripts.finalize_budget_selection import _command, _require_outer_lock
from scripts.prepare_budget_final_runs import prepare
from scripts.run_experiment import ROOT, sha256


def run_rehearsal(root, final_plan):
    root = Path(root).resolve()
    path = (root / final_plan).resolve()
    if not path.is_relative_to(root / "configs/generated/budget120/final"):
        raise ValueError("Rehearsal requires the dedicated final plan")
    plan = json.loads(path.read_text(encoding="utf-8"))
    # Revalidate completed selection evidence and immutable generated snapshots.
    prepare(root, root / plan["selection"], path.parent)
    for name, expected in plan["files_sha256"].items():
        item = (root / name).resolve()
        if not item.is_relative_to(root) or sha256(item) != expected:
            raise ValueError("Final snapshot hash changed: " + name)
    configs = plan["run_configs"]["dev_anal"]
    if set(configs) != {"C", "D", "E"}:
        raise ValueError("Rehearsal requires exactly C/D/E")
    for condition, name in configs.items():
        config_path = (root / name).resolve()
        if not config_path.is_relative_to(path.parent):
            raise ValueError("Rehearsal configuration is outside final snapshots")
        run = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = {"condition": condition, "split": "dev_anal", "n_rows": 16,
                    "alpha": plan["common_alpha"], "temperature": .5, "basis": "B0",
                    "seed_set": 0, "row_start": 0, "id_version": 2,
                    "base_config": plan["base_config"]}
        if run != expected:
            raise ValueError("Rehearsal run differs from the frozen selection")
    commands = [["-m", "scripts.build_bank", configs[c]] for c in ("C", "D", "E")]
    commands += [["-m", "scripts.run_experiment", configs[c]] for c in ("C", "D", "E")]
    stage = json.loads((path.parent / "stage_rehearsal.json").read_text(encoding="utf-8"))
    analysis_path = path.parent / "rehearsal_analysis_plan.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (stage["commands"] != commands or analysis["split"] != "dev_anal"
            or analysis["selection_file"] != plan["selection"]
            or analysis["base_config"] != plan["base_config"]
            or any(analysis["run_configs"].get(c) != configs[c] for c in configs)
            or analysis["run_configs"].get("greedy") != "configs/runs/dev_anal_greedy.yaml"
            or (analysis_path.parent / analysis["root"]).resolve() != root):
        raise ValueError("Rehearsal execution/analysis plan mismatch")
    _require_outer_lock(root)
    for command in commands:
        _command(root, command[1:])
    _command(root, ["scripts.analyze_budget120", analysis_path.relative_to(root).as_posix()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-plan", default="configs/generated/budget120/final/final_plan.json")
    args = parser.parse_args()
    run_rehearsal(ROOT, args.final_plan)


if __name__ == "__main__":
    main()
