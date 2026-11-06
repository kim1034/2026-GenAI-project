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
