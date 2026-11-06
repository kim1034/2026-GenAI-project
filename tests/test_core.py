import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.diag import effective_dimension
from src.config import load_config
from src.data import load_split
from src.extract import INVALID, extract_lenient, extract_main, is_correct
from src.rng import rng_for
from src.score import answer_entropy, flip_fraction, grouped_entropy
from src.select_alpha import common_alpha_exception, select_alpha
from src.stats import (
    equivalence_probability,
    holm_adjust,
    mean_auroc,
    paired_bootstrap,
    paired_delong,
    project_test_se,
    verdict,
)
from scripts.budget import calculate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 18.", "18"),
        ("The answer is $1,200.", "1200"),
        ("the answer is: 7.50 dollars", "7.5"),
        ("The answer is -3.", "-3"),
        ("The answer is 3/4.", "0.75"),
        ("The answer is 18. Wait, the answer is 20.", "20"),
        ("The answer is 18.00", "18"),
        ("THE ANSWER IS -0", "0"),
        ("The answer is 1/0", INVALID),
        ("No final anchor, but 17 appears", INVALID),
        ("", INVALID),
    ],
)
def test_main_extractor(text, expected):
    assert extract_main(text) == expected


def test_lenient_and_exact_answer():
    assert extract_lenient("No anchor, so she has 18 apples.") == "18"
    assert is_correct("18", "18.00")
    assert not is_correct(INVALID, "0")


def test_invalid_answers_are_separate_in_main_entropy():
    answers = ["18"] * 6 + [INVALID] * 2
    assert answer_entropy(answers) == pytest.approx(0.7356219398)
    assert answer_entropy(answers, invalid="merge") == pytest.approx(0.5623351446)
    assert answer_entropy(answers, invalid="drop") == 0
    assert answer_entropy([INVALID] * 8) == pytest.approx(math.log(8))
    assert grouped_entropy(["18"] * 8 + [INVALID] * 8) == pytest.approx([0, math.log(8)])
    assert flip_fraction(["18", INVALID, "20"], "18") == pytest.approx(2 / 3)
    assert flip_fraction(["18", "18"], INVALID) == 1


def test_rng_keys_are_repeatable_and_independent():
    assert np.array_equal(rng_for(1, 0).integers(0, 1000, 20), rng_for(1, 0).integers(0, 1000, 20))
    assert not np.array_equal(rng_for(1, 0).integers(0, 1000, 20), rng_for(1, 1).integers(0, 1000, 20))


def test_mean_of_auroc_and_paired_inference():
    generator = np.random.default_rng(42)
    y = np.repeat([0, 1], 50)
    a = generator.normal(size=(100, 4)) + y[:, None] * 0.5
    b = a * 0.6 + generator.normal(size=(100, 4)) * 0.4
    expected = np.mean([roc_auc_score(y, a[:, j]) for j in range(4)])
    assert mean_auroc(y, a) == pytest.approx(expected)
    delong = paired_delong(y, a, b)
    bootstrap = paired_bootstrap(y, a, b, repeats=200, analysis_no=1)
    assert delong["delta"] == pytest.approx(bootstrap["delta"])
    assert delong["se"] == pytest.approx(bootstrap["se"], rel=0.35)
    assert paired_bootstrap(y, a, b, repeats=200, analysis_no=1) == bootstrap


def test_holm_and_verdict_boundaries():
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )
    assert verdict((-0.01, 0.01), practical_delta=0.02) == 7
    assert verdict((-0.03, 0.01), practical_delta=0.02) == 8
    assert verdict((0.02, 0.04), practical_delta=0.02) == 1


def test_common_alpha_selection_and_exception():
    rows = [
        {"alpha": 0.1, "auroc": 0.70, "maj_acc": 0.8, "invalid": 0.02},
        {"alpha": 0.2, "auroc": 0.704, "maj_acc": 0.8, "invalid": 0.02},
        {"alpha": 0.4, "auroc": 0.72, "maj_acc": 0.8, "invalid": 0.02},
    ]
    assert select_alpha(rows, baseline_accuracy=0.8, baseline_invalid=0.02,
                        max_accuracy_drop=0.05, max_invalid_increase=0.05,
                        tie_tolerance=0.005) == (0.4, "ok")
    assert common_alpha_exception(rows, common_alpha=0.1, selected_alpha=0.4,
                                  minimum_gap_steps=2, minimum_auroc_gain=0.01)


def test_effective_dimension_limits():
    assert effective_dimension(np.array([1, 0, 0, 0]), center=False) == 1
    assert effective_dimension(np.ones(4), center=False) == 4
    with pytest.raises(ValueError):
        effective_dimension(np.ones(4), center=True)


def test_local_development_data_and_sealed_test():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/base.yaml")
    rows = load_split(root, "dev_base", config)
    assert len(rows) == 50
    assert all(row["id"].startswith("train-") and "gold_number" in row for row in rows)
    with pytest.raises(PermissionError):
        load_split(root, "test", config)


def test_budget_reproduces_document_assumption():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/base.yaml")
    result = calculate(config, rows16_seconds=8, rows32_seconds=12)
    assert result["required_hours"] == pytest.approx(62.7, abs=0.05)
    assert result["planned_hours"] == pytest.approx(97.9, abs=0.05)


def test_power_projection_matches_observed_two_subset_variance():
    draws = np.array([[0.2, 0.1], [0.0, 0.2], [0.3, 0.2], [0.1, 0.0]])
    expected = draws.mean(axis=1).std(ddof=1) * math.sqrt(200 / 1319)
    assert project_test_se(draws, n_dev=200, n_test=1319, target_subsets=2) == pytest.approx(expected)
    assert 0.8 < equivalence_probability(0.006, practical_delta=0.02) < 0.9
