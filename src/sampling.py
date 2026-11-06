"""CPU float64 inverse-CDF sampling with common random numbers."""

import numpy as np


def sample_from_logits(logits: np.ndarray, uniform: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    u = np.asarray(uniform, dtype=np.float64)
    if values.ndim != 2 or u.shape != (values.shape[0],):
        raise ValueError("expected [rows, vocabulary] logits and one uniform value per row")
    if not np.all(np.isfinite(values)) or not np.all((0 <= u) & (u < 1)):
        raise ValueError("logits must be finite and uniforms must lie in [0, 1)")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0:
        return np.argmax(values, axis=1).astype(np.int64)
    cdf = np.array(values, copy=True)
    np.divide(cdf, temperature, out=cdf)
    cdf -= cdf.max(axis=1, keepdims=True)
    np.exp(cdf, out=cdf)
    np.cumsum(cdf, axis=1, out=cdf)
    thresholds = u * cdf[:, -1]
    return np.array(
        [min(np.searchsorted(cdf[row], thresholds[row], side="right"), values.shape[1] - 1)
         for row in range(values.shape[0])],
        dtype=np.int64,
    )
