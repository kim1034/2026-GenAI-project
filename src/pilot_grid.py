"""Translate the fixed development pilot rule into an auditable selection grid."""

import math
from decimal import Decimal
import numpy as np


def pilot_grid(rows, baseline_unique8_first_subset, expected_alphas):
    expected = [float(x) for x in expected_alphas]
    if not expected or len(set(expected)) != len(expected) or any(not math.isfinite(x) or x <= 0 for x in expected):
        raise ValueError("Expected pilot magnitudes must be unique, finite and positive")
    if not math.isfinite(baseline_unique8_first_subset) or not 1 <= baseline_unique8_first_subset <= 8:
        raise ValueError("Baseline distinct-answer count must be in [1, 8]")
    ordered = sorted(rows, key=lambda row: row["alpha"])
    alphas = [float(row["alpha"]) for row in ordered]
    if alphas != sorted(expected):
        raise ValueError("Pilot rows must match the entire expected grid without duplicates")
    for row in ordered:
        unique, invalid = row["mean_unique8_first_subset"], row["invalid"]
        if not math.isfinite(unique) or not 1 <= unique <= 8 or not math.isfinite(invalid) or not 0 <= invalid <= 1:
            raise ValueError("Invalid pilot summary")
    threshold = Decimal(str(baseline_unique8_first_subset)) + Decimal("0.2")
    lower = [float(row["alpha"]) for row in ordered if Decimal(str(row["mean_unique8_first_subset"])) >= threshold]
    upper = [float(row["alpha"]) for row in ordered if row["invalid"] <= 0.2]
    lo, hi = min(lower) if lower else None, max(upper) if upper else None
    result = {"alpha_lo": lo, "alpha_hi": hi, "grid": [], "status": "blocked_no_hi", "launch_allowed": False}
    if hi is None:
        return result
    if lo is None or lo > hi:
        grid = [a for a in alphas if a <= hi][-6:]
        status, rule = ("ok" if len(grid) == 6 else "underfull_fallback"), "largest_six_pilot_values_at_or_below_hi"
    else:
        grid = sorted({float(f"{value:.3g}") for value in np.geomspace(lo, hi, 6)})
        status = "degenerate" if lo == hi else "ok" if len(grid) == 6 else "rounded_duplicates"
        rule = "six_log_spaced_values_three_significant_digits"
    result.update(grid=grid, status=status, rule=rule, launch_allowed=status == "ok")
    return result
