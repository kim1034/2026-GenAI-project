import pytest

from src.select_alpha import selection_decision


BASELINE = {"split": "dev_select", "condition": "A", "maj_acc": .8, "invalid": 0, "run_id": "A"}
RULE = {"max_acc_drop": .05, "max_invalid_increase": .05, "tie_tolerance": .005}


def test_selection_uses_inclusive_feasibility_and_smallest_tied_magnitude():
    rows = [{"alpha": .1, "auroc": .7, "maj_acc": .75, "invalid": .05},
            {"alpha": .2, "auroc": .705, "maj_acc": .8, "invalid": 0}]
    result = selection_decision(rows, BASELINE, RULE)
    assert result["selected_alpha"] == .1
    assert result["feasible_count"] == 2
    assert result["reason"] == "ok"
    with pytest.raises(ValueError, match="condition A"):
        selection_decision(rows, {**BASELINE, "condition": "greedy"}, RULE)


@pytest.mark.parametrize("replacement", [{"auroc": None}, {"auroc": float("nan")},
                                        {"status": "running"}, {"label_classes_complete": False}])
def test_selection_rejects_incomplete_or_undefined_metrics(replacement):
    row = {"alpha": .1, "auroc": .7, "maj_acc": .8, "invalid": 0, **replacement}
    with pytest.raises(ValueError):
        selection_decision([row], BASELINE, RULE)
