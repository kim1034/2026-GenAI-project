import numpy as np
import pytest

from src.stats import project_test_se, projection_details


def test_negative_projection_is_reported_without_clipping():
    draws = np.array([[-1, 1], [0, 0], [1, -1]], dtype=float)
    result = projection_details(draws, n_dev=200, n_test=1319, target_subsets=4)
    assert result["V1"] == pytest.approx(1)
    assert result["Cv"] == pytest.approx(-1)
    assert result["raw_variance"] == pytest.approx(-.5)
    assert result["status"] == "invalid_projection"
    assert result["se"] is None
    assert result["negative_covariance_warning"]
    with pytest.raises(ValueError, match="negative projected"):
        project_test_se(draws, n_dev=200, n_test=1319, target_subsets=4)


def test_negative_covariance_does_not_invalidate_positive_raw_projection():
    x = np.array([-1, -1, 1, 1], dtype=float)
    z = np.array([-1, 1, -1, 1], dtype=float)
    draws = np.stack([x, -.1 * x + z], axis=1)
    result = projection_details(draws, n_dev=200, n_test=1319, target_subsets=4)
    assert result["Cv"] < 0
    assert result["status"] == "ok"
    assert result["se"] > 0


def test_nonfinite_draws_and_fractional_subset_counts_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        projection_details(np.array([[1, 2], [3, np.nan]]), n_dev=200, n_test=1319, target_subsets=4)
    with pytest.raises(ValueError, match="integers"):
        projection_details(np.ones((3, 2)), n_dev=200, n_test=1319, target_subsets=2.5)
