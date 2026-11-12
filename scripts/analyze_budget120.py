"""Offline V1.4 analysis; all four final runs must verify before any statistics.

Usage: python -m scripts.analyze_budget120 plan.json
Plan: root, split (dev_select|dev_anal|test), selection_file, base_config,
run_configs {greedy,C,D,E}, output_dir. Paths are relative to root, except root
itself, which is relative to the plan. Never launches a model or GPU job.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.make_pilot_grid import _different_paths, _runtime, _signature
from scripts.run_experiment import _validate_run, run_id, sha256
from src.config import assert_test_locked, config_sha256, load_config
from src.data import load_split
from src.result_io import load_completed, summarize_run
from src.score import answer_entropy
from src.stats import shared_subset_bootstrap, verdict


REVISION = "budget120-v1.4"
CONDITIONS = ("C", "D", "E")
ATTEMPTS = 10000
PR6 = {1: "반증", 2: "부분확인", 3: "반증", 4: "반증", 5: "부분확인", 6: "반증", 7: "확인", 8: "유보"}


def _read(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input/output path escapes ROOT")
    return path


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _load_verified(root, run_path, condition, split, alpha, base, ids):
    """Validate provenance and raw completeness without calculating a statistic."""
    run = _read(run_path)
    actual_base_path = _path(root, run.get("base_config", "configs/base.yaml"))
    actual_base = load_config(actual_base_path)
    if condition == "greedy":
        for section in ("model", "prompt", "data", "global_seed"):
            if actual_base[section] != base[section]:
                raise ValueError("Greedy label model/prompt/data/seed differs from final base")
        for key, value in base["decode"].items():
            if not key.startswith("k_generate_") and actual_base["decode"].get(key) != value:
                raise ValueError("Greedy decode settings changed")
    elif actual_base != base:
        raise ValueError("Run base differs from analysis base_config")
    base = actual_base
    _validate_run(run, base)
    rows = 1 if condition == "greedy" else 16
    required = {"split": split, "condition": condition, "alpha": alpha,
                "temperature": 0.0 if condition == "greedy" else .5,
                "seed_set": 0, "n_rows": rows}
    if (any(run.get(key) != value for key, value in required.items())
            or run.get("row_start", 0) != 0
            or run.get("basis") != ("B0" if condition in CONDITIONS else None)):
        raise ValueError("Run violates fixed split/condition/alpha/rows/T/B0/s0")
    name = run_id(run)
    manifest_path = _path(root, f"outputs/manifests/{name}.json")
    manifest = _read(manifest_path)
    analysis_path = _path(root, base["stats"]["analysis_list"])
    analysis = _read(analysis_path)
    expected = {"status": "complete", "run_id": name, "condition": condition, "split": split,
                "n_rows": rows, "row_start": 0, "problem_count": len(ids),
                "sample_count": len(ids) * rows, "run_config_sha256": sha256(run_path),
                "config_sha256": config_sha256(base, analysis), "model_id": base["model"]["id"],
                "model_revision": base["model"]["revision"],
                "fewshot_sha256": sha256(_path(root, base["prompt"]["fewshot_file"])),
                "system_sha256": sha256(_path(root, base["prompt"]["system_file"]))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest disagrees with complete run/config/source identity")
    snapshots = {"run_parameters": run, "base_config_snapshot": base, "alpha": alpha,
                 "temperature": required["temperature"], "id_version": run.get("id_version", 1)}
    if any(key in manifest and manifest[key] != value for key, value in snapshots.items()):
        raise ValueError("Manifest parameter snapshot changed")
    # Legacy development greedy is an immutable label reference. Its original
    # runtime metadata is retained, never upgraded or claimed equivalent.
    runtime = None if condition == "greedy" and split in ("dev_select", "dev_anal") else _runtime(manifest, base)
    if runtime is not None:
        for source, digest in runtime["source_sha256"].items():
            if sha256(_path(root, source)) != digest:
                raise ValueError("Execution source changed: " + source)
    grouped = load_completed(root, manifest, ids)
    for items in grouped.values():
        for item in items:
            fields = {"cond": condition, "alpha": alpha, "temperature": required["temperature"],
                      "basis": run.get("basis") or "NA", "seed_set": 0,
                      "n_rows": rows, "subset": item["k"] // 8}
            if any(item.get(key) != value for key, value in fields.items()):
                raise ValueError("Raw response parameters changed")
            if not isinstance(item.get("answer"), str) or item.get("finish_reason") not in ("stop", "length", "eos"):
                raise ValueError("Raw answer/finish_reason missing or invalid")
    evidence = {"run_config": str(run_path.relative_to(root)), "run_config_sha256": sha256(run_path),
                "manifest": str(manifest_path.relative_to(root)), "manifest_sha256": sha256(manifest_path),
                "output_sha256": manifest["output_sha256"], "base_config_sha256": sha256(actual_base_path),
                "analysis_sha256": sha256(analysis_path), "runtime": runtime,
                "runtime_comparison": "legacy label reference; not claimed equivalent" if runtime is None else "validated",
                "signature": _signature(base, analysis, root, sha256(root / f"data/splits/{split}.ids"))}
    if condition in CONDITIONS:
        rho_path = _path(root, base["noise"]["rho_file"])
        if manifest.get("rho_sha256") != sha256(rho_path):
            raise ValueError("Execution-time rho hash missing or changed")
        if "rho_snapshot" in manifest and manifest["rho_snapshot"] != _read(rho_path):
            raise ValueError("Execution-time rho snapshot changed")
        bank_path = _path(root, manifest["bank_file"])
        bank_manifest = bank_path.with_suffix(".json")
        bank = _read(bank_manifest)
        wanted = {"condition": condition, "split": split, "basis": "B0", "seed_set": 0,
                  "row_start": 0, "n_rows": 16, "dimension": base["noise"]["d"],
                  "global_seed": 2026, "ids": ids, "dtype": "float32", "sparse_p": None}
        if (any(bank.get(key) != value for key, value in wanted.items())
                or bank.get("file_sha256") != manifest.get("bank_sha256")
                or sha256(bank_path) != manifest.get("bank_sha256")):
            raise ValueError("Noise bank identity/hash changed")
        evidence.update(rho_sha256=sha256(rho_path), bank_sha256=sha256(bank_path),
                        bank_manifest_sha256=sha256(bank_manifest))
    elif manifest.get("bank_file") is not None or manifest.get("bank_sha256") is not None:
        raise ValueError("Unperturbed run has a noise bank")
    return evidence, grouped


def _verify_selection(root, path, base):
    selection = _read(path)
    if (selection.get("schema_version") != 1 or selection.get("status") != "complete"
            or selection.get("protocol_amendment") != REVISION or selection.get("split") != "dev_select"
            or set(selection.get("runs", {})) != {"A", "C", "D", "E"}):
        raise ValueError("A completed V1.4 dev_select decision is required")
    alpha = selection.get("common_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("A finite positive frozen common_alpha is required")
    inputs = selection.get("inputs_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Selection inputs_sha256 must record input path/hash pairs")
    for source, digest in inputs.items():
        if sha256(_path(root, source)) != digest:
            raise ValueError("Selection input hash changed")
    ids_path = root / "data/splits/dev_select.ids"
    ids = ids_path.read_text(encoding="ascii").splitlines()
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("Selection requires the unchanged 200 dev_select IDs")
    reference_runtime = None
    for condition in ("A", "C", "D", "E"):
        spec = selection["runs"][condition]
        run_path = _path(root, spec["run_config"])
        # Selection may have its own immutable base snapshot.
        selected_base = load_config(_path(root, _read(run_path).get("base_config", "configs/base.yaml")))
        evidence, _ = _load_verified(root, run_path, condition, "dev_select",
                                     None if condition == "A" else alpha, selected_base, ids)
        for key in ("run_config_sha256", "manifest_sha256", "output_sha256"):
            if spec.get(key) != evidence[key]:
                raise ValueError("Selected run hash changed")
        if _path(root, spec["manifest"]) != _path(root, evidence["manifest"]):
            raise ValueError("Selected manifest identity changed")
        if reference_runtime is not None and _different_paths(reference_runtime, evidence["runtime"]):
            raise ValueError("Selection execution runtimes differ")
        reference_runtime = evidence["runtime"]
        for section in ("model", "prompt"):
            if selected_base[section] != base[section]:
                raise ValueError("Selected model/prompt differs from final base")
    return selection, reference_runtime


def verify_inputs(plan_path):
    """The sole I/O gate: returns only after selection and all target runs pass."""
    plan_path = Path(plan_path).resolve()
    plan = _read(plan_path)
    root = (plan_path.parent / plan["root"]).resolve()
    split = plan["split"]
    if split not in ("dev_select", "dev_anal", "test") or set(plan["run_configs"]) != {"greedy", *CONDITIONS}:
        raise ValueError("Analysis requires dev_select/dev_anal/test and exactly greedy/C/D/E")
    output = _path(root, plan["output_dir"])
    if output == root or not any(output.is_relative_to(root / folder) and output != root / folder
                                 for folder in ("outputs", "reports")):
        raise ValueError("output_dir must be a dedicated ROOT/outputs or ROOT/reports subdirectory")
    base_path = _path(root, plan["base_config"])
    base = load_config(base_path)
    if (base["global_seed"] != 2026 or base["decode"]["k_main"] != 8
            or base["data"]["splits"] != {"dev_base": 50, "dev_select": 200, "dev_anal": 50, "test": 1319}
            or any(base["decode"][key] != 16 for key in ("k_generate_dev", "k_generate_test", "k_generate_rehearsal"))):
        raise ValueError("Base must preserve IDs/counts and V1.4 16-response design")
    protocol_path = root / "configs/budget120_protocol.json"
    protocol = _read(protocol_path)
    if (protocol.get("protocol_amendment") != REVISION
            or protocol["analysis"]["confirmatory"] != ["D-C"]
            or protocol["analysis"]["E_minus_C_bootstrap_key"] is not None
            or protocol["analysis"]["practical_delta"] != .02):
        raise ValueError("Unrecognized protocol amendment")
    if split == "test":
        assert_test_locked(base, root / "LOCK.md")
    policy_path = root / "configs/development_check_policy.json"
    if split == "dev_select":
        if not policy_path.is_file():
            raise ValueError("dev_select pipeline check requires development_check_policy.json")
        policy = _read(policy_path)
        if (policy.get("generate_dev_anal_responses") is not False
                or policy.get("pipeline_check_split") != "dev_select"):
            raise ValueError("Development check policy does not authorize selected-data reuse")
    selection_path = _path(root, plan["selection_file"])
    selection, selection_runtime = _verify_selection(root, selection_path, base)
    if split == "dev_select":
        for condition in CONDITIONS:
            if _path(root, plan["run_configs"][condition]) != _path(root, selection["runs"][condition]["run_config"]):
                raise ValueError("Pipeline check must reuse the exact final selected C/D/E run configs")
        if _path(root, plan["run_configs"]["greedy"]) != root / "configs/runs/dev_select_greedy.yaml":
            raise ValueError("Pipeline check must reuse configs/runs/dev_select_greedy.yaml")
    data = load_split(root, split, base)  # Includes the existing test seal, before test answers are read.
    ids = [row["id"] for row in data]
    expected_n = {"test": 1319, "dev_anal": 50, "dev_select": 200}[split]
    if len(ids) != expected_n:
        raise ValueError("Unexpected final split count")
    runs, provenance = {}, {}
    for condition in ("greedy", *CONDITIONS):
        evidence, grouped = _load_verified(root, _path(root, plan["run_configs"][condition]), condition,
                                           split, None if condition == "greedy" else selection["common_alpha"], base, ids)
        if evidence["runtime"] is not None and _different_paths(selection_runtime, evidence["runtime"]):
            raise ValueError("Final execution runtime/source differs from selection")
        if condition != "greedy":
            for pid in ids:
                if any(item["prompt_sha256"] != runs["greedy"][pid][0]["prompt_sha256"] for item in grouped[pid]):
                    raise ValueError("Condition prompt differs from greedy")
        runs[condition], provenance[condition] = grouped, evidence
    code_root = Path(__file__).resolve().parents[1]
    sources = [code_root / name for name in ("scripts/analyze_budget120.py", "scripts/run_experiment.py",
               "scripts/make_pilot_grid.py", "src/result_io.py", "src/score.py", "src/stats.py",
               "src/config.py", "src/data.py", "src/extract.py", "src/rng.py")]
    inputs = {"plan_sha256": sha256(plan_path), "selection_sha256": sha256(selection_path),
              "base_config_sha256": sha256(base_path), "protocol_sha256": sha256(protocol_path),
              "development_check_policy_sha256": sha256(policy_path) if policy_path.is_file() else None,
              "ids_sha256": sha256(root / f"data/splits/{split}.ids"),
              "processed_sha256": sha256(root / f"data/processed/{'test' if split == 'test' else 'train'}.jsonl"),
              "LOCK_sha256": sha256(root / "LOCK.md") if (root / "LOCK.md").is_file() else None,
              "analysis_code_sha256": {path.name: sha256(path) for path in sources}, "runs": provenance}
    return {"root": root, "output": output, "split": split, "alpha": selection["common_alpha"],
            "data": data, "runs": runs, "inputs": inputs}


def compute_statistics(verified):
    """Pure analysis of fully verified evidence; never reads a dataset or run file."""
    data, runs = verified["data"], verified["runs"]
    ids = [row["id"] for row in data]
    gold = {row["id"]: row["gold_number"] for row in data}
    metrics, scores, curves, problems = {}, {}, [], []
    y = None
    for condition in CONDITIONS:
        summary, labels, score = summarize_run(runs[condition], runs["greedy"], gold)
        if score.shape != (len(ids), 2):
            raise ValueError("Final scores require two disjoint groups of eight")
        if y is not None and not np.array_equal(y, labels):
            raise ValueError("Conditions have different greedy labels")
        y = labels
        complete = summary["label_classes_complete"]
        summary.update(auroc_subsets=[float(roc_auc_score(y, score[:, j])) for j in range(2)] if complete else None,
                       auprc_subsets=[float(average_precision_score(y, score[:, j])) for j in range(2)] if complete else None,
                       reason=None if complete else "greedy labels contain only one class")
        summary["auprc"] = float(np.mean(summary["auprc_subsets"])) if complete else None
        for key in ("fa", "hit", "sel"):
            summary.setdefault(key, None)
        metrics[condition], scores[condition] = summary, score
        for k in (2, 4, 8, 16):
            matrix = np.array([[answer_entropy([item["answer"] for item in runs[condition][pid][start:start + k]])
                                for start in range(0, 16, k)] for pid in ids])
            values = [float(roc_auc_score(y, matrix[:, j])) for j in range(16 // k)] if complete else None
            value = float(np.mean(values)) if complete else None
            if k == 8 and complete and not np.isclose(value, summary["auroc"], atol=1e-14):
                raise ValueError("K=8 differs from primary estimand")
            curves.append({"condition": condition, "K": k, "subsets": 16 // k, "auroc": value,
                           "auroc_subsets": values, "reason": summary["reason"]})
        for i, pid in enumerate(ids):
            problems.append({"id": pid, "condition": condition, "y": int(y[i]),
                             "entropy_0_7": float(score[i, 0]), "entropy_8_15": float(score[i, 1])})
    comparisons, arrays = {}, {"labels": y, "ordered_ids": np.array(ids),
                               "condition_names": np.array(CONDITIONS),
                               "condition_scores": np.stack([scores[name] for name in CONDITIONS])}
    complete = set(y.tolist()) == {0, 1}
    for left, right, key in (("D", "C", 1), ("D", "E", 2), ("E", "C", None)):
        name = f"{left}-{right}"
        entry = {"delta": metrics[left]["auroc"] - metrics[right]["auroc"] if complete else None,
                 "ci95": None, "se": None, "verdict": None, "rng_key": [10, key] if key else None,
                 "attempted": 0, "valid": 0, "skipped": 0,
                 "confirmatory": name == "D-C" and verified["split"] == "test",
                 "reason": None if complete else "greedy labels contain only one class"}
        if key is None:
            entry["reason"] = "no registered E-C bootstrap key; point estimate only"
        elif complete:
            boot = shared_subset_bootstrap(y, scores, [(left, right)], repeats=ATTEMPTS,
                                           analysis_no=key, global_seed=2026)
            draws = boot["subset_deltas"][:, 0, :].mean(axis=1)
            valid_draws = draws[boot["valid_mask"]]
            entry.update(ci95=np.percentile(valid_draws, [2.5, 97.5]).tolist(), se=float(valid_draws.std(ddof=1)),
                         attempted=boot["attempted"], valid=boot["valid"], skipped=boot["skipped"])
            for field in ("indices", "valid_mask", "subset_deltas", "point_subset_deltas"):
                arrays[f"{name}_{field}"] = boot[field]
            arrays[f"{name}_mean_deltas"] = draws
            if not np.isclose(entry["delta"], boot["point_subset_deltas"].mean(), atol=1e-14):
                raise ValueError("Bootstrap point differs from original full-sample delta")
            if name == "D-C":
                entry["verdict"] = verdict(tuple(entry["ci95"]), practical_delta=.02)
        comparisons[name] = entry
    arithmetic = None if not complete else float(comparisons["D-C"]["delta"] - comparisons["D-E"]["delta"] - comparisons["E-C"]["delta"])
    if arithmetic is not None and abs(arithmetic) > 1e-14:
        raise ValueError("D-C = D-E + E-C arithmetic check failed")
    confirmatory = verified["split"] == "test"
    selection_data_reused = verified["split"] == "dev_select"
    scope = "development_pipeline_check" if selection_data_reused else "confirmatory_test" if confirmatory else "rehearsal"
    v = comparisons["D-C"]["verdict"]
    report = {"schema_version": 1, "status": "complete", "protocol_amendment": REVISION,
              "scope": scope, "confirmatory": confirmatory,
              "selection_data_reused": selection_data_reused, "independent_evaluation": not selection_data_reused,
              "split": verified["split"], "common_alpha": verified["alpha"], "n": len(ids),
              "practical_delta": .02,
              "inputs": verified["inputs"], "input_sha256": _fingerprint(verified["inputs"]),
              "estimand": "mean of two subset AUROCs; not AUROC of mean entropy",
              "metrics": metrics, "comparisons": comparisons, "k_curve": curves,
              "bootstrap": {"attempts_per_registered_key": ATTEMPTS, "global_seed": 2026,
                            "resample_unit": "paired problem", "single_class_attempt": "skip without redraw",
                            "se_ddof": 1, "ci": "percentile_95", "keys": {"D-C": [10, 1], "D-E": [10, 2], "E-C": None}},
              "arithmetic_residual": arithmetic,
              "PR6": {"verdict": v, "outcome": PR6[v] if confirmatory and v else None,
                      "reason": None if confirmatory else ("selection data reused for pipeline check; no independent evaluation or confirmatory conclusion"
                                                            if selection_data_reused else "rehearsal calculation check; no confirmatory conclusion")},
              "not_performed": ["original PR1 confirmation", "PR2a", "PR2b", "PR2c", "PR3_T5", "PR3_S", "PR4", "PR5",
                                "S", "B", "Ap", "T5", "P2", "boost", "test gradients", "original Holm family", "p values"],
              "analysis_unavailable_reason": None if complete else "greedy labels contain only one class"}
    return report, problems, arrays


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                     for key, value in row.items()} for row in rows)
    return stream.getvalue().encode("utf-8")


def _npz(arrays):
    # Fixed ZIP metadata permits byte-identical reruns across wall-clock times.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())
    return stream.getvalue()


def _figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for name in CONDITIONS:
        rows = [row for row in report["k_curve"] if row["condition"] == name]
        axes[0].plot([row["K"] for row in rows], [np.nan if row["auroc"] is None else row["auroc"] for row in rows], "o-", label=name)
    axes[0].set(xlabel="Responses per disjoint subset (K)", ylabel="Mean subset AUROC", xticks=[2, 4, 8, 16])
    axes[0].legend(frameon=False)
    for i, (name, entry) in enumerate(report["comparisons"].items()):
        if entry["delta"] is not None:
            axes[1].plot(entry["delta"], i, "o", color="black")
            if entry["ci95"] is not None:
                axes[1].hlines(i, *entry["ci95"], color="black")
    axes[1].axvline(0, color="gray", linewidth=.8)
    axes[1].set(xlabel="AUROC difference (95% percentile CI)", yticks=[0, 1, 2], yticklabels=["D-C", "D-E", "E-C (point only)"], ylim=(-.5, 2.5))
    fig.suptitle("V1.4: " + report["scope"])
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=180, metadata={"Software": "budget120-v1.4"})
    plt.close(fig)
    return stream.getvalue()


def render_outputs(report, problems, arrays):
    """Tables, markdown and figure derive exclusively from analysis JSON values."""
    npz = _npz(arrays)
    report["npz_sha256"] = hashlib.sha256(npz).hexdigest()
    lines = ["# 실험 결과 — V1.4", "", f"Scope: {report['scope']}; n={report['n']}; common alpha={report['common_alpha']}.",
             "", "AUROC and AUPRC average the two disjoint eight-response subset estimates.",
             "", "| Condition | AUROC | AUPRC | Majority accuracy | Invalid | Truncated |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in report["metrics"].items():
        lines.append(f"| {name} | {row['auroc']} | {row['auprc']} | {row['maj_acc']} | {row['invalid']} | {row['truncated']} |")
    lines += ["", "| Comparison | Delta | CI95 | SE | Verdict |", "| --- | ---: | --- | ---: | --- |"]
    for name, row in report["comparisons"].items():
        lines.append(f"| {name} | {row['delta']} | {row['ci95']} | {row['se']} | {row['verdict']} |")
    lines += ["", "D-E is descriptive. E-C has a point estimate only because no bootstrap key was registered.",
              "", f"PR6: {report['PR6']}", "", "Not performed: " + ", ".join(report["not_performed"]), ""]
    return {"analysis.json": (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            "problem_scores.csv": _csv(problems), "bootstrap.npz": npz,
            "metrics.csv": _csv([{"condition": name, **row} for name, row in report["metrics"].items()]),
            "comparisons.csv": _csv([{"comparison": name, **row} for name, row in report["comparisons"].items()]),
            "k_curve.csv": _csv(report["k_curve"]), "실험결과.md": "\n".join(lines).encode("utf-8"),
            "figure2.png": _figure(report)}


def analyze_budget120(plan_path):
    verified = verify_inputs(plan_path)
    report, problems, arrays = compute_statistics(verified)
    outputs = render_outputs(report, problems, arrays)
    output = verified["output"]
    for name, raw in outputs.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Analysis output conflicts with verified inputs/content: " + str(path))
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    report = analyze_budget120(args.plan)
    print(json.dumps({key: report[key] for key in ("status", "scope", "n", "input_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
