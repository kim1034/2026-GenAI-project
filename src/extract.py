"""Extract numeric GSM8K answers from model responses."""

import re
from decimal import Decimal, InvalidOperation, DivisionByZero


ANCHOR = re.compile(r"the answer is", re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")
INVALID = "INVALID"


def _normalize(token: str) -> str:
    token = token.replace(",", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            value = Decimal(numerator.strip()) / Decimal(denominator.strip())
        else:
            value = Decimal(token)
    except (InvalidOperation, DivisionByZero, ZeroDivisionError):
        return INVALID
    if not value.is_finite():
        return INVALID
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def extract_main(text: str) -> str:
    anchors = list(ANCHOR.finditer(text))
    if not anchors:
        return INVALID
    match = NUMBER.search(text[anchors[-1].end() :])
    return _normalize(match.group()) if match else INVALID


def extract_lenient(text: str) -> str:
    main = extract_main(text)
    if main != INVALID:
        return main
    matches = list(NUMBER.finditer(text))
    return _normalize(matches[-1].group()) if matches else INVALID


def is_correct(answer: str, gold_number: str) -> bool:
    return answer != INVALID and Decimal(answer) == Decimal(gold_number)
