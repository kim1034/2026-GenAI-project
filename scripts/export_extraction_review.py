"""Export complete development greedy responses for a documented extraction audit."""

import argparse
import json
from pathlib import Path

from scripts.run_experiment import ROOT, json_write
from src.config import load_config
from src.data import load_split
from src.result_io import load_completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    base = load_config(ROOT / "configs/base.yaml")
    rows = load_split(ROOT, "dev_base", base)
    run_id = "dev_base_greedy_aNA_T0.00_NA_s0_k1"
    path = args.snapshot_root / "outputs/manifests" / f"{run_id}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    grouped = load_completed(args.snapshot_root, manifest, [row["id"] for row in rows])
    review = [{"id": row["id"], "question": row["question"], "gold_number": row["gold_number"],
               "generated_text": grouped[row["id"]][0]["text"], "extracted_answer": grouped[row["id"]][0]["answer"]}
              for row in rows]
    json_write(ROOT / "outputs/diagnostics/extraction_review_dev_base.json",
               {"scope": "development_greedy_only", "source_output_sha256": manifest["output_sha256"],
                "review_status": "awaiting_review", "problems": review})
    print(f"Exported {len(review)} complete responses; extraction review remains pending")


if __name__ == "__main__":
    main()
