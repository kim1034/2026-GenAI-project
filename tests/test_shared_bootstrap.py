import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.stats import mean_auroc, shared_subset_bootstrap


def test_all_comparisons_share_indices_and_invalid_attempts_are_preserved():
    y = np.array([0, 0, 1, 1])
    a = np.array([[0, 1], [1, 2], [2, 0], [3, 3]])
    b = a[:, ::-1]
    c = a.copy()
    result = shared_subset_bootstrap(y, {"A": a, "B": b, "C": c}, [("A", "B"), ("A", "C")], repeats=30)
    assert result["attempted"] == result["valid"] + result["skipped"] == 30
    assert result["skipped"] > 0
    assert np.isnan(result["subset_deltas"][~result["valid_mask"]]).all()
    assert np.all(result["subset_deltas"][result["valid_mask"], 1] == 0)
    for attempt in np.flatnonzero(result["valid_mask"]):
        idx = result["indices"][attempt]
        expected = [roc_auc_score(y[idx], a[idx, j]) - roc_auc_score(y[idx], b[idx, j]) for j in range(2)]
        assert result["subset_deltas"][attempt, 0] == pytest.approx(expected)
    again = shared_subset_bootstrap(y, {"A": a, "B": b, "C": c}, [("A", "B"), ("A", "C")], repeats=30)
    np.testing.assert_equal(result["subset_deltas"], again["subset_deltas"])
    np.testing.assert_equal(result["indices"], again["indices"])


def test_nonbinary_labels_cannot_be_silently_cast_to_binary():
    with pytest.raises(ValueError, match="label classes"):
        mean_auroc([0, 1.5, 0, 1.5], np.arange(4))
