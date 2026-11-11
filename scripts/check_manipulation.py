"""Check generated float32 noise banks against the prespecified G2 thresholds."""

import argparse
import json

import numpy as np
from scipy.stats import kurtosis, ks_2samp

from scripts.run_experiment import ROOT, json_write
from src.bank import load_bank


def gap_ratio(vectors):
    ordered = np.sort(vectors.astype(np.float64), axis=1)
    d = ordered.shape[1]
    gaps = np.diff(ordered[:, d // 4:3 * d // 4], axis=1)
    a, b = gaps[:, :-1], gaps[:, 1:]
    ratio = np.divide(np.minimum(a, b), np.maximum(a, b), out=np.zeros_like(a), where=np.maximum(a, b) > 0)
    return float(ratio.mean())


def check_bank(path):
    manifest = json.loads(path.read_text())
    bank = load_bank(path.with_suffix(".npy"), path, manifest["ids"], row_start=manifest["row_start"], n_rows=manifest["n_rows"])
    condition = manifest["condition"]
    flat = bank.reshape(-1, bank.shape[-1])
    if len(flat) < 100:
        raise ValueError("G2 requires at least 100 vectors")
    sample = np.asarray(flat[:100], dtype=np.float64)
    check = {"manifest": str(path.relative_to(ROOT)), "condition": condition, "vectors": len(flat),
             "checks": {}, "mean": float(sample.mean()), "kurtosis_median": float(np.median(kurtosis(sample, axis=1, fisher=True))),
             "tail6": float(np.mean(np.sqrt(bank.shape[-1]) * np.abs(sample) > 6))}
    tests = check["checks"]
    if condition in ("B", "Bc"):
        minimum, maximum = min(float(part.min()) for part in bank), max(float(part.max()) for part in bank)
        check.update(minimum=minimum, maximum=maximum)
        tests["range"] = (0 <= minimum and maximum < 1) if condition == "B" else (-0.5 <= minimum and maximum < 0.5)
        tests["mean"] = abs(check["mean"] - (0.5 if condition == "B" else 0)) <= 0.01
    else:
        norm_error = mean_error = 0.0
        for part in bank:
            values = np.asarray(part, dtype=np.float64)
            norm_error = max(norm_error, float(np.abs(np.linalg.norm(values, axis=1) - 1).max()))
            mean_error = max(mean_error, float(np.abs(values.mean(axis=1)).max()))
        check.update(norm_max_error=norm_error, mean_max_abs=mean_error)
        tests.update(norm=norm_error < 1e-5, centered_mean=mean_error < 1e-6)
    if condition not in ("S", "B"):
        check["gap_ratio"] = gap_ratio(sample)
        bounds = (0.59, 0.61) if condition == "D" else (0.52, 0.54) if condition == "O" else (0.37, 0.40)
        tests["gap_ratio"] = bounds[0] <= check["gap_ratio"] <= bounds[1]
    bounds = {"C": (-0.15, 0.15), "D": (-1.05, -0.90), "E": (-1.05, -0.90), "O": (-1.05, -0.90),
              "Bc": (-1.22, -1.17), "L": (2.7, 3.2), "T5": (3.3, 4.8), "S_H": (-0.05, 0.05), "T5_H": (-0.05, 0.05)}
    if condition == "S":
        approx = 3 * manifest["dimension"] / (round(manifest["dimension"] * manifest["sparse_p"]) + 2) - 3
        bounds["S"] = (0.8 * approx, 1.05 * approx)
        tests["support_counts_verified"] = bool(manifest.get("support_count_sha256"))
    if condition in bounds:
        low, high = bounds[condition]
        tests["kurtosis"] = low <= check["kurtosis_median"] <= high
    tail_bounds = {"C": (0, 2e-6), "L": (1.5e-4, 2.7e-4), "T5": (4.5e-4, 6.9e-4)}
    if condition in tail_bounds:
        low, high = tail_bounds[condition]
        tests["tail6"] = low <= check["tail6"] <= high
    check["passed"] = all(tests.values())
    return check, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev_base", choices=["dev_base", "dev_select", "dev_anal", "test"])
    args = parser.parse_args()
    checks, samples = [], {}
    for path in sorted((ROOT / "bank").glob(f"*/*/s0/{args.split}_rows*.json")):
        if path.name.endswith(".state.json"):
            continue
        check, sample = check_bank(path)
        checks.append(check)
        samples[(check["condition"], path.stem)] = sample
    if not checks:
        raise RuntimeError("No completed banks found")
    marginal = []
    for (condition, stem), sample in samples.items():
        if condition == "D" and ("E", stem) in samples:
            statistic = float(ks_2samp(sample.ravel(), samples[("E", stem)].ravel()).statistic)
            marginal.append({"bank": stem, "ks_D_E": statistic, "passed": statistic < 0.02})
    report = {"split": args.split, "banks": checks, "marginal_checks": marginal,
              "passed": all(item["passed"] for item in checks + marginal),
              "scope": "generated_banks_only; does not include dense-GUE equivalence or hook tests"}
    json_write(ROOT / f"outputs/diagnostics/manipulation_{args.split}.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("G2 bank check failed; preserve evidence and inspect before any inference")


if __name__ == "__main__":
    main()
