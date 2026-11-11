import hashlib
import json

import pytest

from src.result_io import load_completed, summarize_run


def test_duplicate_response_is_rejected_even_with_valid_file_hash(tmp_path):
    folder = tmp_path / "outputs/samples"
    folder.mkdir(parents=True)
    row = {"id": "train-00001", "run_id": "dev", "k": 0, "prompt_sha256": "same"}
    raw = (json.dumps(row) + "\n") * 2
    (folder / "dev.jsonl").write_bytes(raw.encode("utf-8"))
    manifest = {"status": "complete", "run_id": "dev", "n_rows": 2, "sample_count": 2,
                "output_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    with pytest.raises(ValueError, match="duplicate"):
        load_completed(tmp_path, manifest, ["train-00001"])


def test_summary_uses_mean_of_subset_auroc_and_checks_prompt():
    def items(answers):
        return [{"answer": answer, "prompt_sha256": "same", "finish_reason": "eos"} for answer in answers]
    greedy = {"a": items(["1"]), "b": items(["2"])}
    # Subset AUCs are 1 and 0.5. Averaging the per-problem entropies instead would yield 1.
    grouped = {"a": items(["1"] * 16), "b": items(["2", "3"] * 4 + ["2"] * 8)}
    summary, y, scores = summarize_run(grouped, greedy, {"a": "1", "b": "1"})
    assert summary["auroc"] == pytest.approx(0.75)
    assert list(y) == [0, 1]
    grouped["b"][0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt"):
        summarize_run(grouped, greedy, {"a": "1", "b": "1"})
