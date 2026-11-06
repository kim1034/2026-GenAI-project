"""Recalculate the A5000 budget after measuring actual per-problem run times."""

import argparse
from pathlib import Path

import yaml


def calculate(config: dict, *, rows16_seconds: float, rows32_seconds: float) -> dict[str, float]:
    budget = config["budget"]
    runs = budget["problem_runs"]
    if rows16_seconds <= 0 or rows32_seconds <= 0:
        raise ValueError("measured run times must be positive")
    dev16 = sum(runs[key] for key in ("pilot_dev16", "select_dev16", "grid_edge_dev16_max"))
    dev32 = runs["rehearsal_dev32"]
    development = (dev16 * rows16_seconds + dev32 * rows32_seconds) / 3600
    required_test = runs["required_test32"] * rows32_seconds / 3600
    boost = runs["boost_test32"] * rows32_seconds / 3600
    optional = runs["optional_test32"] * rows32_seconds / 3600
    required = development + required_test + budget["other_gpu_hours"]
    planned = required + boost + optional
    return {
        "development_hours": development,
        "required_hours": required,
        "boost_hours": boost,
        "optional_hours": optional,
        "planned_hours": planned,
        "reserve_hours": budget["hours_total"] - planned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--rows16-seconds", type=float)
    parser.add_argument("--rows32-seconds", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    measured = config["budget"]["t_run_s"]
    assumed = config["budget"]["assumed_t_run_s"]
    rows16 = args.rows16_seconds or measured["rows16"] or assumed["rows16"]
    rows32 = args.rows32_seconds or measured["rows32"] or assumed["rows32"]
    result = calculate(config, rows16_seconds=rows16, rows32_seconds=rows32)
    print(f"16-row {rows16:.2f}s; 32-row {rows32:.2f}s")
    for name, value in result.items():
        print(f"{name}: {value:.1f}")


if __name__ == "__main__":
    main()
