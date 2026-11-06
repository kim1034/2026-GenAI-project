"""Development-only noise magnitude selection and common-alpha diagnostic."""

from collections.abc import Sequence


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
