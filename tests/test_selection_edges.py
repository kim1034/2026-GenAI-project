import pytest

from src.select_alpha import accept_edge, edge_candidate


def test_one_edge_uses_actual_grid_ratio_or_prescribed_absolute_value():
    assert edge_candidate([.05, .0871, .152], .152, kind="relative") == pytest.approx(.152**2/.0871)
    assert edge_candidate([.05, .0871, .152], .0871, kind="relative") is None
    assert edge_candidate([.01, .03, .05], .01, kind="B") == .005
    assert edge_candidate([.5, .6, 1.7], 1.7, kind="temperature") == 1.9
    assert edge_candidate([.05, .1], .1, kind="relative", already_used=True) is None
    with pytest.raises(ValueError, match="one-point"):
        edge_candidate([.05], .05, kind="relative")


def test_edge_requires_strict_gain_and_inclusive_feasibility():
    previous = {"auroc": .7, "maj_acc": .8, "invalid": 0}
    candidate = {"auroc": .705, "maj_acc": .75, "invalid": .05}
    assert not accept_edge(previous, candidate, baseline_accuracy=.8, baseline_invalid=0)
    candidate["auroc"] = .706
    assert accept_edge(previous, candidate, baseline_accuracy=.8, baseline_invalid=0)
    candidate["invalid"] = .051
    assert not accept_edge(previous, candidate, baseline_accuracy=.8, baseline_invalid=0)
