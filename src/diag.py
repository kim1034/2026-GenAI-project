"""Small pure functions for gradient-based effective dimension diagnostics."""

import numpy as np


def effective_dimension(gradient: np.ndarray, *, center: bool = True) -> float:
    values = np.asarray(gradient, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("gradient must be a finite vector")
    if center:
        values = values - values.mean()
    energy = np.sum(values**2)
    fourth = np.sum(values**4)
    if fourth == 0:
        raise ValueError("effective dimension is undefined for a zero gradient")
    return float(energy**2 / fourth)


def sparse_decision(deff, *, dimension=8960):
    values = np.asarray(deff, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 1) or np.any(values > dimension * (1 + 1e-10)):
        raise ValueError("effective dimensions must be finite and in [1, dimension]")
    median = float(np.median(values))
    target = float(np.clip(1 / median, .001, .02))
    candidates = np.array([.001, .002, .005, .01, .02])
    selected = float(candidates[np.argmin(np.abs(np.log(candidates) - np.log(target)))])
    q1, q3 = np.quantile(np.log10(values), [.25, .75], method="linear")
    heterogeneity = float(q3 - q1)
    return {"status": "complete", "p": selected, "median_deff_g_m": median, "inverse_median_clipped": target,
            "H": heterogeneity, "PR2b": "S<C" if heterogeneity >= .3 else "practical_equivalence",
            "support_count": round(selected * dimension), "problems": len(values),
            "quantile_method": "linear", "diagnostic_fallback": False}
