"""Generate a development-only progress table from verified complete runs."""

import json
import argparse
from pathlib import Path

from scripts.run_experiment import ROOT, json_write
from src.config import load_config
from src.data import load_split
from src.extract import INVALID, is_correct
from src.result_io import load_completed, summarize_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=ROOT)
    args = parser.parse_args()
    snapshot = args.snapshot_root.resolve()
    config = load_config(ROOT / "configs/base.yaml")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (snapshot / "outputs/manifests").glob("dev_*.json")]
    results = []
    for split in ("dev_base", "dev_select", "dev_anal"):
        data = load_split(ROOT, split, config)
        ids = [row["id"] for row in data]
        gold = {row["id"]: row["gold_number"] for row in data}
        completed = [m for m in manifests if m["split"] == split and m["status"] == "complete"]
        greedy_manifests = [m for m in completed if m["condition"] == "greedy"]
        if not greedy_manifests:
            continue
        if len(greedy_manifests) != 1:
            raise ValueError("Ambiguous greedy reference")
        gm = greedy_manifests[0]
        frozen_ids = snapshot / f"data/splits/{split}.ids"
        if snapshot != ROOT and (not frozen_ids.is_file() or frozen_ids.read_bytes() != (ROOT / f"data/splits/{split}.ids").read_bytes()):
            raise RuntimeError("Backup split IDs differ from the local frozen split")
        greedy = load_completed(snapshot, gm, ids)
        greedy_summary = {"run_id": gm["run_id"], "split": split, "condition": "greedy", "problems": len(ids),
                          "accuracy": sum(is_correct(greedy[i][0]["answer"], gold[i]) for i in ids) / len(ids),
                          "invalid": sum(greedy[i][0]["answer"] == INVALID for i in ids) / len(ids),
                          "truncated": sum(greedy[i][0]["finish_reason"] == "length" for i in ids) / len(ids)}
        results.append(greedy_summary)
        for manifest in completed:
            if manifest["condition"] == "greedy":
                continue
            grouped = load_completed(snapshot, manifest, ids)
            summary, _, _ = summarize_run(grouped, greedy, gold)
            results.append({"run_id": manifest["run_id"], "split": split,
                            "condition": manifest["condition"], **summary})
    output = ROOT / "outputs/analysis/development.json"
    json_write(output, {"scope": "development_only", "complete_runs": results})
    lines = ["# 개발 실험 자동 집계", "", "완료 manifest와 원시 출력 SHA-256을 확인한 실행만 집계한다. test 결과는 읽지 않는다.", "",
             "| 분할 | 조건 | 문제 수 | 정확도/8응답 다수결 정확도 | 무효율 | 잘림율 | 평균 AUROC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in results:
        auc = "—" if item.get("auroc") is None else f'{item["auroc"]:.4f}'
        accuracy = item.get("accuracy", item.get("maj_acc"))
        lines.append(f'| {item["split"]} | {item["condition"]} | {item["problems"]} | {accuracy:.4f} | {item["invalid"]:.4f} | {item["truncated"]:.4f} | {auc} |')
    (ROOT / "reports/development_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
