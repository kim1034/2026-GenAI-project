"""Check the locally stored GSM8K source files without printing test examples."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = {"train": 7473, "test": 1319}


def main() -> None:
    checksum_lines = []
    for split, expected_count in EXPECTED.items():
        path = ROOT / f"gsm8k_{split}.jsonl"
        raw_bytes = path.read_bytes()
        rows = raw_bytes.decode("utf-8").split("\n")
        if rows[-1] == "":
            rows.pop()
        assert len(rows) == expected_count, (split, len(rows), expected_count)

        comma_count = 0
        decimal_count = 0
        for index, line in enumerate(rows):
            row = json.loads(line)
            assert set(row) == {"question", "answer"}, (split, index)
            assert isinstance(row["question"], str) and row["question"], (split, index)
            assert isinstance(row["answer"], str) and "####" in row["answer"], (split, index)
            gold_raw = row["answer"].rsplit("####", 1)[1].strip()
            comma_count += "," in gold_raw
            gold = Decimal(gold_raw.replace(",", ""))
            decimal_count += gold != gold.to_integral_value()

        digest = hashlib.sha256(raw_bytes).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}")
        print(f"{split}: {len(rows)} rows, 0 parse failures, {comma_count} comma answers, {decimal_count} noninteger answers")
        print(f"  SHA-256: {digest}")

    (ROOT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
