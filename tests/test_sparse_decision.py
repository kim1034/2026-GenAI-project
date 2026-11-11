import numpy as np
import pytest

from src.diag import sparse_decision


def test_sparsity_uses_inverse_median_and_prespecified_candidates():
    result = sparse_decision(np.full(300, 200.0))
    assert result["p"] == .005
    assert result["H"] == 0
    assert result["PR2b"] == "practical_equivalence"
    assert sparse_decision(np.full(300, 8000.0))["p"] == .001
    assert sparse_decision(np.full(300, 1.0))["p"] == .02


def test_heterogeneity_comes_from_log10_iqr_not_raw_iqr():
    result = sparse_decision([10] * 150 + [1000] * 150)
    assert result["H"] == 2
    assert result["PR2b"] == "S<C"
    with pytest.raises(ValueError):
        sparse_decision([1, np.nan])
