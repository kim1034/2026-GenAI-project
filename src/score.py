"""Answer entropy and answer-change measurements."""

import math
from collections import Counter
from collections.abc import Sequence


INVALID = "INVALID"


def _prepared(answers: Sequence[str], invalid: str) -> list[str]:
    if invalid == "singleton":
        return [f"INVALID#{i}" if answer == INVALID else answer for i, answer in enumerate(answers)]
    if invalid == "merge":
        return list(answers)
    if invalid == "drop":
        return [answer for answer in answers if answer != INVALID]
    raise ValueError(f"unknown invalid-answer policy: {invalid}")


def answer_entropy(answers: Sequence[str], *, invalid: str = "singleton") -> float:
    if not answers:
        raise ValueError("at least one answer is required")
    valid = _prepared(answers, invalid)
    if not valid:
        return math.log(len(answers))
    counts = Counter(valid)
    n = len(valid)
    return -sum((count / n) * math.log(count / n) for count in counts.values())


def grouped_entropy(answers: Sequence[str], *, group_size: int = 8) -> list[float]:
    if group_size < 2 or not answers or len(answers) % group_size:
        raise ValueError("answers must form complete groups of size at least two")
    return [answer_entropy(answers[i : i + group_size]) for i in range(0, len(answers), group_size)]


def majority(answers: Sequence[str]) -> str:
    valid = [answer for answer in answers if answer != INVALID]
    if not valid:
        return INVALID
    counts = Counter(valid)
    top = max(counts.values())
    return next(answer for answer in valid if counts[answer] == top)


def flip_fraction(answers: Sequence[str], greedy: str) -> float:
    if not answers:
        raise ValueError("at least one answer is required")
    return sum(answer == INVALID or greedy == INVALID or answer != greedy for answer in answers) / len(answers)


def selectivity(flip_fractions: Sequence[float], labels: Sequence[int], *, clip: float = 1e-4) -> dict[str, float]:
    values = list(flip_fractions)
    y = list(labels)
    if len(values) != len(y) or not y or set(y) != {0, 1}:
        raise ValueError("both label classes and aligned flip fractions are required")
    fa = sum(value for value, label in zip(values, y) if label == 0) / y.count(0)
    hit = sum(value for value, label in zip(values, y) if label == 1) / y.count(1)
    def logit(value: float) -> float:
        p = min(1 - clip, max(clip, value))
        return math.log(p / (1 - p))
    return {"fa": fa, "hit": hit, "sel": logit(hit) - logit(fa)}
