import numpy as np
import pytest

from src.gradient_diag import summarize_gradient


def test_centered_sensitivity_is_offset_invariant():
    gradient = np.array([1.0, -1.0] * 10)
    direct = summarize_gradient(gradient)
    shifted = summarize_gradient(gradient + 7.0)
    assert direct["deff"] == pytest.approx(20)
    assert direct["top10"] == pytest.approx(0.5)
    assert direct["norm"] == pytest.approx(np.sqrt(20))
    for key in ("deff", "top10", "norm"):
        assert shifted[key] == pytest.approx(direct[key])
    assert shifted["mean"] == pytest.approx(7)


def test_common_mode_only_has_no_identifiable_effective_dimension():
    with pytest.raises(ValueError, match="zero gradient"):
        summarize_gradient(np.ones(20))
