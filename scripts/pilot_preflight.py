"""Require the completed development checks before starting the existing pilot."""

import json
import argparse

from scripts.run_experiment import ROOT, sha256
from src.result_io import load_completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-v14", action="store_true")
    args = parser.parse_args()
    def read(name):
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    validation = read("outputs/diagnostics/gpu_validation_v14.json" if args.budget_v14 else "outputs/diagnostics/gpu_validation.json")
    if args.budget_v14 and (validation.get("protocol_amendment") != "budget120-v1.4" or validation.get("scope_rows") != [1, 16]):
        raise RuntimeError("GPU validation scope differs from the amended protocol")
    if validation["status"] != "passed_computational_checks" or not all(validation["hooks"].values()):
        raise RuntimeError("GPU computational validation is incomplete")
    if not read("outputs/diagnostics/dense_gue_n100.json")["passed"]:
        raise RuntimeError("Dense GUE comparison did not pass")
    manipulation = read("outputs/diagnostics/manipulation_dev_base.json")
    required_banks = {"C", "D", "E"} if args.budget_v14 else {"B", "C", "D", "E", "T5", "S"}
    if not manipulation["passed"] or not required_banks <= {b["condition"] for b in manipulation["banks"]}:
        raise RuntimeError("Required development banks have not all passed")
    runtime = read("outputs/diagnostics/runtime.json")
    if runtime["status"] != "complete" or not runtime["zero_hook_bitwise_equal"]:
        raise RuntimeError("Runtime or zero-hook validation incomplete")
    probe = read("outputs/diagnostics/gpu_probe.json")
    if probe["last_logits_max_abs_error"] >= 1e-4:
        raise RuntimeError("Last-logit validation failed")
    wrong = total = 0
    for split in ("dev_base", "dev_select", "dev_anal"):
        ids = (ROOT / f"data/splits/{split}.ids").read_text().splitlines()
        name = f"{split}_greedy_aNA_T0.00_NA_s0_k1"
        manifest = read(f"outputs/manifests/{name}.json")
        results = load_completed(ROOT, manifest, ids)
        rows = [results[key][0] for key in ids]
        wrong += sum(not item["correct"] for item in rows)
        total += len(rows)
        if split == "dev_base":
            audit = read("reports/extraction_audit.json")
            if audit["reviewed"] != 50 or audit["mismatches"] or audit["ambiguous"] or audit["source_output_sha256"] != manifest["output_sha256"]:
                raise RuntimeError("Extraction review evidence differs from baseline")
            if sum(item["answer"] == "INVALID" for item in rows) / len(rows) > .1:
                raise RuntimeError("Development base invalid-answer threshold exceeded")
    if total != 300 or not .15 <= wrong / total <= .85:
        raise RuntimeError("Development label prevalence requires review")
    if not args.budget_v14:
        decision = read("calib/sparse_decision.json")
        if decision["status"] != "complete" or decision["problems"] != 300:
            raise RuntimeError("Sparsity decision incomplete")
    print("Development preflight passed; existing pilot may start")


if __name__ == "__main__":
    main()
