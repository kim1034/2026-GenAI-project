"""Development-only noise magnitude selection and common-alpha diagnostic."""

from collections.abc import Sequence
from decimal import Decimal
import math


def edge_candidate(grid, selected, *, kind, already_used=False):
    if kind not in ("relative", "B", "temperature"):
        raise ValueError("unknown edge kind")
    values = sorted(float(value) for value in grid)
    if not values or len(values) != len(set(values)) or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("grid requires distinct finite positive values")
    if selected not in values:
        raise ValueError("selected value is not in the completed grid")
    if already_used or values[0] < selected < values[-1]:
        return None
    if len(values) < 2:
        raise ValueError("edge direction is undefined for a one-point grid")
    lower = selected == values[0]
    if kind == "B":
        return .005 if lower else .13
    if kind == "temperature":
        return .3 if lower else 1.9
    candidate = values[0] ** 2 / values[1] if lower else values[-1] ** 2 / values[-2]
    if not math.isfinite(candidate) or candidate <= 0:
        raise ValueError("invalid extrapolated magnitude")
    return candidate


def accept_edge(previous, candidate, *, baseline_accuracy, baseline_invalid,
                max_accuracy_drop=.05, max_invalid_increase=.05, minimum_gain=.005):
    def dec(value):
        if not math.isfinite(value):
            raise ValueError("nonfinite selection metric")
        return Decimal(str(value))
    for row in (previous, candidate):
        for key in ("auroc", "maj_acc", "invalid"):
            if not 0 <= dec(row[key]) <= 1:
                raise ValueError("selection metric outside [0,1]")
    return (dec(candidate["maj_acc"]) >= dec(baseline_accuracy) - dec(max_accuracy_drop)
            and dec(candidate["invalid"]) <= dec(baseline_invalid) + dec(max_invalid_increase)
            and dec(candidate["auroc"]) - dec(previous["auroc"]) > dec(minimum_gain))


def select_alpha(
    rows: Sequence[dict],
    *,
    baseline_accuracy: float,
    baseline_invalid: float,
    max_accuracy_drop: float,
    max_invalid_increase: float,
    tie_tolerance: float,
) -> tuple[float, str]:
    if not rows:
        raise ValueError("selection rows are required")
    feasible = [
        row for row in rows
        if row["maj_acc"] >= baseline_accuracy - max_accuracy_drop
        and row["invalid"] <= baseline_invalid + max_invalid_increase
    ]
    if not feasible:
        return min(float(row["alpha"]) for row in rows), "no_feasible"
    best = max(float(row["auroc"]) for row in feasible)
    chosen = min(float(row["alpha"]) for row in feasible if row["auroc"] >= best - tie_tolerance)
    return chosen, "ok"


def common_alpha_exception(
    rows: Sequence[dict],
    *,
    common_alpha: float,
    selected_alpha: float,
    minimum_gap_steps: int,
    minimum_auroc_gain: float,
) -> bool:
    ordered = sorted(rows, key=lambda row: row["alpha"])
    positions = {float(row["alpha"]): index for index, row in enumerate(ordered)}
    if common_alpha not in positions or selected_alpha not in positions:
        raise ValueError("both alphas must be present in the tested grid")
    common = ordered[positions[common_alpha]]
    selected = ordered[positions[selected_alpha]]
    return (
        abs(positions[selected_alpha] - positions[common_alpha]) >= minimum_gap_steps
        and selected["auroc"] - common["auroc"] > minimum_auroc_gain
    )
