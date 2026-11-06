"""Statistics for the prespecified mean-of-subset AUROC estimand."""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from src.rng import rng_for


def _inputs(labels: Sequence[int], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    x = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or x.ndim not in (1, 2) or x.shape[0] != len(y):
        raise ValueError("scores must align with a one-dimensional label vector")
    if not np.all(np.isfinite(x)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("finite scores and both label classes are required")
    return y, x[:, None] if x.ndim == 1 else x


def mean_auroc(labels: Sequence[int], scores: np.ndarray) -> float:
    y, x = _inputs(labels, scores)
    return float(np.mean([roc_auc_score(y, x[:, j]) for j in range(x.shape[1])]))


def paired_bootstrap(
    labels: Sequence[int],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    repeats: int,
    analysis_no: int,
    global_seed: int = 2026,
) -> dict[str, object]:
    y, a = _inputs(labels, scores_a)
    _, b = _inputs(labels, scores_b)
    if a.shape != b.shape or repeats < 2:
        raise ValueError("paired scores must have the same shape and repeats must exceed one")
    point = mean_auroc(y, a) - mean_auroc(y, b)
    rng = rng_for(10, analysis_no, global_seed=global_seed)
    differences = []
    skipped = 0
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        if y[indices].min() == y[indices].max():
            skipped += 1
            continue
        differences.append(mean_auroc(y[indices], a[indices]) - mean_auroc(y[indices], b[indices]))
    if len(differences) < 2:
        raise ValueError("too few bootstrap draws contain both classes")
    values = np.asarray(differences)
    return {
        "delta": float(point),
        "ci95": tuple(float(v) for v in np.percentile(values, [2.5, 97.5])),
        "se": float(values.std(ddof=1)),
        "skipped": skipped,
        "repeats": repeats,
    }


def _placement(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    positive = scores[y == 1]
    negative = scores[y == 0]
    # One value per problem: average placement across response subsets.
    pos_placement = np.zeros(len(positive), dtype=np.float64)
    neg_placement = np.zeros(len(negative), dtype=np.float64)
    for subset in range(scores.shape[1]):
        comparisons = positive[:, subset, None] - negative[None, :, subset]
        kernel = (comparisons > 0).astype(np.float64) + 0.5 * (comparisons == 0)
        pos_placement += kernel.mean(axis=1)
        neg_placement += kernel.mean(axis=0)
    pos_placement /= scores.shape[1]
    neg_placement /= scores.shape[1]
    return pos_placement, neg_placement, float(pos_placement.mean())


def paired_delong(labels: Sequence[int], scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    y, a = _inputs(labels, scores_a)
    _, b = _inputs(labels, scores_b)
    if a.shape != b.shape:
        raise ValueError("paired scores must have the same shape")
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        raise ValueError("DeLong variance requires at least two examples per class")
    a_pos, a_neg, auc_a = _placement(a, y)
    b_pos, b_neg, auc_b = _placement(b, y)
    pos_diff = a_pos - b_pos
    neg_diff = a_neg - b_neg
    variance = pos_diff.var(ddof=1) / len(pos_diff) + neg_diff.var(ddof=1) / len(neg_diff)
    se = float(np.sqrt(max(variance, 0.0)))
    delta = auc_a - auc_b
    p = float(2 * norm.sf(abs(delta) / se)) if se > 0 else (0.0 if delta else 1.0)
    return {"delta": delta, "se": se, "p_two_sided": p}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    if not ordered or any(not 0 <= p <= 1 for _, p in ordered):
        raise ValueError("one or more p-values in [0, 1] are required")
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p))
        adjusted[name] = running
    return adjusted


def verdict(ci95: tuple[float, float], *, practical_delta: float) -> int:
    low, high = ci95
    d = practical_delta
    if not (low <= high and d > 0):
        raise ValueError("ordered confidence interval and positive practical delta required")
    if low >= d:
        return 1
    if 0 < low < d:
        return 2 if high < d else 3
    if high <= -d:
        return 4
    if -d < high < 0:
        return 5 if low > -d else 6
    if -d < low <= 0 <= high < d:
        return 7
    return 8


def false_alarm_interpolate(fa: Sequence[float], metric: Sequence[float], at_fa: float) -> float:
    """Linear interpolation after monotone isotonic smoothing of a development curve."""
    from sklearn.isotonic import IsotonicRegression

    x = np.asarray(fa, dtype=np.float64)
    y = np.asarray(metric, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not np.all(np.isfinite(x + y)):
        raise ValueError("aligned finite one-dimensional curves are required")
    if not x.min() <= at_fa <= x.max():
        raise ValueError("target false-alarm rate lies outside the observed curve")
    fitted = IsotonicRegression(increasing=True, out_of_bounds="raise").fit_transform(x, y)
    order = np.argsort(x)
    return float(np.interp(at_fa, x[order], fitted[order]))


def trend_contrast(labels: Sequence[int], scores_by_condition: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> float:
    if set(weights) - set(scores_by_condition):
        raise ValueError("a trend weight refers to a missing condition")
    if not np.isclose(sum(weights.values()), 0.0):
        raise ValueError("trend weights must sum to zero")
    return float(sum(weight * mean_auroc(labels, scores_by_condition[name]) for name, weight in weights.items()))


def project_test_se(bootstrap_subset_deltas: np.ndarray, *, n_dev: int, n_test: int, target_subsets: int) -> float:
    """Project bootstrap subset covariance from development to test sample size."""
    draws = np.asarray(bootstrap_subset_deltas, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] < 2:
        raise ValueError("at least two draws and two response subsets are required")
    if n_dev < 2 or n_test < 2 or target_subsets < 1:
        raise ValueError("positive sample sizes and subset count are required")
    covariance = np.cov(draws, rowvar=False, ddof=1)
    count = covariance.shape[0]
    within = float(np.trace(covariance) / count)
    between = float((covariance.sum() - np.trace(covariance)) / (count * (count - 1)))
    variance = max(0.0, between + (within - between) / target_subsets)
    return float(np.sqrt(variance * n_dev / n_test))


def equivalence_probability(se: float, *, practical_delta: float) -> float:
    if se <= 0 or practical_delta <= 0:
        raise ValueError("standard error and practical delta must be positive")
    boundary = min(1.96, practical_delta / se - 1.96)
    return float(max(0.0, 2 * norm.cdf(boundary) - 1))
