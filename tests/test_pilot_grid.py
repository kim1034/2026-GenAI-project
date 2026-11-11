import pytest

from src.pilot_grid import pilot_grid


def rows(unique, invalid):
    return [{"alpha": a, "mean_unique8_first_subset": u, "invalid": i}
            for a, u, i in zip([.0125, .025, .05, .1, .2, .4, .8, 1.6], unique, invalid)]


def test_pilot_uses_distinct_count_lower_bound_and_invalid_upper_bound():
    result = pilot_grid(rows([1, 1, 1.3, 1.5, 2, 3, 5, 8], [0, 0, 0, 0, 0, .1, .2, .3]), 1,
                        [.0125, .025, .05, .1, .2, .4, .8, 1.6])
    assert result["alpha_lo"] == .05
    assert result["alpha_hi"] == .8
    assert result["grid"] == [.05, .0871, .152, .264, .459, .8]
    assert result["launch_allowed"]


def test_fallback_and_undefined_cases_do_not_invent_new_magnitudes():
    expected = [.0125, .025, .05, .1, .2, .4, .8, 1.6]
    result = pilot_grid(rows([1] * 8, [0] * 8), 1, expected)
    assert result["grid"] == expected[-6:]
    assert result["launch_allowed"]
    result = pilot_grid(rows([1] * 8, [0, 0] + [.3] * 6), 1, expected)
    assert result["status"] == "underfull_fallback"
    assert not result["launch_allowed"]
    assert pilot_grid(rows([1] * 8, [.3] * 8), 1, expected)["status"] == "blocked_no_hi"
    with pytest.raises(ValueError, match="entire expected"):
        pilot_grid(rows([1] * 8, [0] * 8)[:-1], 1, expected)


def test_exact_point_two_increase_is_inclusive():
    expected = [.0125, .025, .05, .1, .2, .4, .8, 1.6]
    result = pilot_grid(rows([2.1, 2.1, 2.3, 3, 3, 3, 3, 3], [0] * 8), 2.1, expected)
    assert result["alpha_lo"] == .05
