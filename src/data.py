"""Read local GSM8K splits with test access guarded by LOCK.md."""

import json
from pathlib import Path

from src.config import assert_test_locked


DEVELOPMENT_SPLITS = {"dev_base", "dev_select", "dev_anal"}


def load_split(root: Path, split: str, config: dict) -> list[dict]:
    if split not in DEVELOPMENT_SPLITS | {"test"}:
        raise ValueError(f"unknown split: {split}")
    if split == "test":
        assert_test_locked(config, root / "LOCK.md")
    ids_path = root / "data" / "splits" / f"{split}.ids"
    ids = ids_path.read_text(encoding="ascii").splitlines()
    if len(ids) != config["data"]["splits"][split] or len(ids) != len(set(ids)):
        raise ValueError("split IDs have an unexpected count or duplicates")
    source = "test" if split == "test" else "train"
    rows_path = root / "data" / "processed" / f"{source}.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").split("\n") if line]
    by_id = {row["id"]: row for row in rows}
    try:
        return [by_id[problem_id] for problem_id in ids]
    except KeyError as error:
        raise ValueError(f"split contains an unknown problem ID: {error}") from error
