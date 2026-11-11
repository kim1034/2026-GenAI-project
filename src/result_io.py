"""Validate complete experiment outputs before scoring them."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.extract import INVALID, is_correct
from src.score import answer_entropy, flip_fraction, majority, selectivity
from src.stats import mean_auroc


def load_completed(root: Path, manifest: dict, expected_ids: list[str]) -> dict[str, list[dict]]:
    if manifest.get("status") != "complete":
        raise ValueError("Only completed runs may be scored")
    path = root / "outputs/samples" / f'{manifest["run_id"]}.jsonl'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["output_sha256"]:
        raise ValueError("Output hash does not match manifest")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    n = manifest["n_rows"]
    if len(rows) != n * len(expected_ids) or manifest["sample_count"] != len(rows):
        raise ValueError("Wrong response count")
    grouped = {key: [] for key in expected_ids}
    for row in rows:
        if row["id"] not in grouped or row["run_id"] != manifest["run_id"]:
            raise ValueError("Unexpected problem or run ID")
        grouped[row["id"]].append(row)
    expected_k = list(range(manifest.get("row_start", 0), manifest.get("row_start", 0) + n))
    for items in grouped.values():
        items.sort(key=lambda x: x["k"])
        if [item["k"] for item in items] != expected_k:
            raise ValueError("Missing or duplicate response indices")
        if len({item["prompt_sha256"] for item in items}) != 1:
            raise ValueError("Prompt changed inside one problem")
    return grouped


def summarize_run(grouped, greedy, gold_by_id):
    ids = list(grouped)
    scores, labels, flips, majority_correct = [], [], [], []
    invalid = truncated = count = 0
    for problem_id in ids:
        items = grouped[problem_id]
        baseline = greedy[problem_id][0]
        if any(item["prompt_sha256"] != baseline["prompt_sha256"] for item in items):
            raise ValueError("Condition prompt differs from greedy prompt")
        answers = [item["answer"] for item in items]
        if len(answers) % 8:
            raise ValueError("Entropy estimation requires complete groups of eight")
        scores.append([answer_entropy(answers[i:i + 8]) for i in range(0, len(answers), 8)])
        majority_correct.extend(is_correct(majority(answers[i:i + 8]), gold_by_id[problem_id])
                                for i in range(0, len(answers), 8))
        labels.append(int(not is_correct(baseline["answer"], gold_by_id[problem_id])))
        flips.append(flip_fraction(answers[:32], baseline["answer"]))
        invalid += sum(answer == INVALID for answer in answers)
        truncated += sum(item["finish_reason"] == "length" for item in items)
        count += len(items)
    score_array = np.asarray(scores)
    summary = {"problems": len(ids), "responses": count, "maj_acc": float(np.mean(majority_correct)),
               "invalid": invalid / count, "truncated": truncated / count,
               "positive_labels": sum(labels), "auroc": None, "label_classes_complete": len(set(labels)) == 2}
    if len(set(labels)) == 2:
        summary.update(auroc=mean_auroc(labels, score_array), **selectivity(flips, labels))
    return summary, np.asarray(labels), score_array
