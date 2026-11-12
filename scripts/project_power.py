"""Project the fixed V8 development covariance without opening test outputs.

Usage: python -m scripts.project_power config.json (YAML is also accepted).
Config: root, selection_file, baseline_greedy_manifest,
baseline_greedy_run_config, manifests:{C,D,E,S,A,Ap?}, output_prefix.
Optional config_root/source_root locate immutable YAML/data/prompt snapshots.
Root directories are relative to this config; evidence/output paths to root;
run YAML paths to config_root. See validate_selection for the version 1 protocol.
No actual power decision is possible before all final selection runs complete.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from scripts.make_pilot_grid import _different_paths, _fingerprint, _path, _runtime, _signature
from scripts.run_experiment import _validate_run, run_id, sha256
from src.config import config_sha256, load_config
from src.data import load_split
from src.extract import is_correct
from src.result_io import load_completed
from src.score import grouped_entropy
from src.stats import equivalence_probability, projection_details, shared_subset_bootstrap


CONDITIONS = ("C", "D", "E", "S", "A", "Ap")
COMPARISONS = (("D", "C"), ("D", "E"), ("S", "C"), ("C", "Ap"))
N_DEV, N_TEST, ATTEMPTS = 200, 1319, 2000


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_selection(path, root, config_root):
    """Explicit protocol until the selection CLI produces this artifact.

    schema_version=1, status=complete, scope=development_selection,
    split=dev_select, common_alpha, alpha_S, T_best, Ap_is_A,
    edge_checks_complete=true, runs={C,D,E,S,A,Ap?}. Each run contains
    run_config/run_config_sha256, manifest/manifest_sha256,
    edge_check_complete=true and evidence:[{path,sha256}]. Evidence must be
    completed JSON with schema_version=1, status=complete, split=dev_select
    and scope=development_selection_evidence. Final C/D/E runs use the common C-selected
    magnitude; D's own optimum never substitutes for that magnitude.
    """
    selection = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version": 1, "status": "complete", "scope": "development_selection",
                "split": "dev_select", "edge_checks_complete": True}
    if (any(selection.get(k) != v for k, v in required.items())
            or type(selection.get("schema_version")) is not int
            or selection.get("edge_checks_complete") is not True):
        raise ValueError("Selection protocol is unsupported, incomplete, or edge checks are unresolved")
    for key in ("common_alpha", "alpha_S", "T_best"):
        value = selection.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Selection requires a positive finite {key}")
    alias = selection["T_best"] == 0.5
    if selection.get("Ap_is_A") is not alias:
        raise ValueError("Ap_is_A must agree exactly with T_best=0.5")
    names = set(CONDITIONS) - ({"Ap"} if alias else set())
    if set(selection.get("runs", {})) != names:
        raise ValueError("Selection must identify exactly the final required runs")
    for name, reference in selection["runs"].items():
        if reference.get("edge_check_complete") is not True or not reference.get("evidence"):
            raise ValueError("Final run lacks completed selection evidence/edge checks")
        yaml_path = _path(config_root, reference["run_config"])
        manifest_path = _path(root, reference["manifest"])
        if (sha256(yaml_path) != reference["run_config_sha256"]
                or sha256(manifest_path) != reference["manifest_sha256"]):
            raise ValueError("Final selected run file hash changed")
        for item in reference["evidence"]:
            evidence_path = _path(root, item["path"])
            if sha256(evidence_path) != item["sha256"]:
                raise ValueError("Selection evidence hash changed")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (type(evidence.get("schema_version")) is not int or evidence.get("schema_version") != 1
                    or evidence.get("status") != "complete"
                    or evidence.get("split") != "dev_select"
                    or evidence.get("scope") != "development_selection_evidence"):
                raise ValueError("Selection evidence must be completed development evidence")
    return selection


def _load_run(root, config_root, source_root, yaml_path, manifest_path, condition, alpha, temperature,
              *, baseline=None):
    run = _read(yaml_path)
    base = load_config(_path(config_root, run.get("base_config", "configs/base.yaml")))
    _validate_run(run, base)
    rows = 1 if condition == "greedy" else 16
    required = {"split": "dev_select", "condition": condition, "alpha": alpha, "temperature": temperature,
                "seed_set": 0, "n_rows": rows}
    if (any(run.get(k) != v for k, v in required.items()) or run.get("row_start", 0) != 0
            or run.get("basis") != (None if condition in ("A", "Ap", "greedy") else "B0")
            or base["decode"]["k_main"] != 8 or base["global_seed"] != 2026
            or base["data"]["splits"]["dev_select"] != N_DEV or base["data"]["splits"]["test"] != N_TEST):
        raise ValueError("Final run violates fixed dev_select condition/alpha/temperature/row settings")
    data = load_split(source_root, "dev_select", base)  # Never load the sealed test split.
    ids = [row["id"] for row in data]
    if len(ids) != N_DEV:
        raise ValueError("Power projection requires exactly 200 development IDs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_path = _path(config_root, run.get("base_config", "configs/base.yaml"))
    analysis_path = _path(config_root, base["stats"]["analysis_list"])
    expected = {"run_id": run_id(run), "condition": condition, "split": "dev_select", "n_rows": rows,
                "row_start": 0, "problem_count": N_DEV, "run_config_sha256": sha256(yaml_path),
                "config_sha256": config_sha256(base, _read(analysis_path)),
                "model_id": base["model"]["id"], "model_revision": base["model"]["revision"],
                "fewshot_sha256": sha256(_path(source_root, base["prompt"]["fewshot_file"])),
                "system_sha256": sha256(_path(source_root, base["prompt"]["system_file"]))}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Manifest disagrees with actual run YAML/base/analysis/prompt sources")
    optional = {"run_parameters": run, "base_config_snapshot": base, "alpha": alpha,
                "temperature": temperature, "id_version": run.get("id_version", 1)}
    if any(key in manifest and manifest[key] != value for key, value in optional.items()):
        raise ValueError("Manifest parameter snapshots disagree with selected YAML")
    grouped = load_completed(root, manifest, ids)
    for items in grouped.values():
        for item in items:
            fields = {"cond": condition, "alpha": alpha, "temperature": temperature, "n_rows": rows,
                      "basis": run.get("basis") or "NA", "seed_set": 0, "subset": item["k"] // 8}
            if any(k not in item or item[k] != v for k, v in fields.items()):
                raise ValueError("Raw output parameters disagree with final selected run")
    signature = _signature(base, _read(analysis_path), source_root,
                           sha256(source_root / "data/splits/dev_select.ids"))
    if condition != "greedy":
        signature["runtime"] = _runtime(manifest, base)
    provenance = {"run_id": run_id(run), "run_config": str(yaml_path), "run_config_sha256": sha256(yaml_path),
                  "condition": condition, "alpha": alpha, "temperature": temperature,
                  "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                  "output_sha256": manifest["output_sha256"], "base_yaml_sha256": sha256(base_path),
                  "config_sha256": expected["config_sha256"], "analysis_yaml_sha256": sha256(analysis_path),
                  "execution_signature": signature, "execution_invocations": manifest.get("execution_invocations"),
                  "sparse_p": base["noise"]["conditions"]["S"].get("p")}
    if condition in ("C", "D", "E", "S"):
        rho_path = _path(source_root, base["noise"]["rho_file"])
        if manifest.get("rho_sha256") != sha256(rho_path):
            raise ValueError("Final run lacks a matching execution-time rho hash")
        if "rho_snapshot" in manifest and manifest["rho_snapshot"] != _read(rho_path):
            raise ValueError("Final run rho snapshot changed")
        provenance["rho_sha256"] = manifest["rho_sha256"]
        bank_path = _path(root, manifest["bank_file"])
        bank = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
        wanted = {"condition": condition, "split": "dev_select", "basis": "B0", "seed_set": 0,
                  "row_start": 0, "n_rows": 16, "dimension": base["noise"]["d"], "global_seed": 2026,
                  "ids": ids, "dtype": "float32", "sparse_p": provenance["sparse_p"] if condition == "S" else None}
        if (any(bank.get(k) != v for k, v in wanted.items())
                or bank.get("file_sha256") != manifest.get("bank_sha256")
                or sha256(bank_path) != manifest.get("bank_sha256")):
            raise ValueError("Final noise bank identity/hash changed")
        if condition == "S" and (provenance["sparse_p"] is None or not 0 < provenance["sparse_p"] <= 1):
            raise ValueError("S requires an already fixed p")
        provenance.update(bank_sha256=manifest["bank_sha256"], bank_manifest_sha256=sha256(bank_path.with_suffix(".json")))
    if baseline is not None:
        previous, reference = baseline
        previous_signature = {k: v for k, v in previous["execution_signature"].items() if k != "runtime"}
        current_signature = {k: v for k, v in signature.items() if k != "runtime"}
        differences = _different_paths(previous_signature, current_signature)
        if differences or list(reference) != ids:
            raise ValueError("Condition differs from greedy configuration/IDs: " + ", ".join(differences))
        if any(item["prompt_sha256"] != reference[pid][0]["prompt_sha256"]
               for pid, items in grouped.items() for item in items):
            raise ValueError("Condition prompt differs from greedy")
    return provenance, grouped, data


def projection_report(bootstrap):
    """Use the reviewed statistics core, retaining negative/zero variance states."""
    results = {}
    for c, (left, right) in enumerate(COMPARISONS):
        draws = bootstrap["subset_deltas"][bootstrap["valid_mask"], c]
        covariance = np.cov(draws, rowvar=False, ddof=1)
        projections = {}
        for subsets in (2, 4, 8):
            detail = projection_details(draws, n_dev=N_DEV, n_test=N_TEST, target_subsets=subsets)
            detail.update(mde_80=2.80 * detail["se"] if detail["status"] == "ok" else None,
                          verdict7_probability_at_true_delta_zero=(equivalence_probability(detail["se"], practical_delta=.02)
                                                                    if detail["status"] == "ok" else None))
            projections[f"J{subsets}"] = detail
        actual_j2 = float(draws.mean(axis=1).var(ddof=1))
        if not np.isclose(projections["J2"]["raw_variance"], actual_j2, atol=1e-14, rtol=1e-10):
            raise ValueError("J2 projection does not match paired bootstrap mean variance")
        point = bootstrap["point_subset_deltas"][c]
        results[f"{left}-{right}"] = {"point_subset_deltas": point.tolist(), "mean_delta": float(point.mean()),
                                     "covariance_ddof1": covariance.tolist(), "covariance_ddof": 1,
                                     "V1": projections["J2"]["V1"], "Cv": projections["J2"]["Cv"],
                                     "residual_component": projections["J2"]["V1"] - projections["J2"]["Cv"],
                                     "bootstrap_mean_variance": actual_j2, "projections": projections,
                                     "J8_execution_scope": ("C/D/E boost scenario" if (left, right) in COMPARISONS[:2]
                                                            else "hypothetical only; no 64-row execution authorization")}
    trigger = results["D-C"]["projections"]["J4"]
    return results, {"comparison": "D-C", "target_subsets": 4, "threshold": .0072, "operator": ">",
                     "recommended": trigger["se"] > .0072 if trigger["status"] == "ok" else None,
                     "diagnostic_status": trigger["status"], "se": trigger["se"]}


def _markdown(report):
    lines = ["# Development power projection", "", "200 development problems; two disjoint groups of eight responses.",
             "One shared, unstratified bootstrap index per attempt; 2000 attempts, no replacement for skipped draws.",
             "", "| Comparison | J | Development mean delta | SE | 80% MDE | P(verdict 7 at true delta 0) | Status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    def display(value):
        return "—" if value is None else f"{value:.8g}"
    for name, comparison in report["comparisons"].items():
        for j, projection in comparison["projections"].items():
            lines.append(f"| {name} | {j[1:]} | {display(comparison['mean_delta'])} | {display(projection['se'])} | {display(projection['mde_80'])} | "
                         f"{display(projection['verdict7_probability_at_true_delta_zero'])} | {projection['status']} |")
    lines.extend(["", f"Boost statistical recommendation: {report['boost_recommendation']['recommended']}.",
                  "Budget status: pending_runtime. Final boost and test row count remain unset.",
                  "", *report["assumptions_and_warnings"], ""])
    return "\n".join(lines)


def project_power(config_path):
    config_path = Path(config_path).resolve()
    config = _read(config_path)
    root = _path(config_path.parent, config["root"])
    config_root = _path(config_path.parent, config.get("config_root", str(root)))
    source_root = _path(config_path.parent, config.get("source_root", str(root)))
    selection_path = _path(root, config["selection_file"])
    selection = validate_selection(selection_path, root, config_root)
    budget = None
    if config.get("budget_decision_file"):
        path = _path(root, config["budget_decision_file"])
        budget = json.loads(path.read_text(encoding="utf-8"))
        if budget.get("status") != "pending_runtime":
            raise ValueError("Completed budget decisions are unsupported until a reviewed V9.4 protocol exists")
        budget = {"path": str(path), "sha256": sha256(path), "used_for_final_decision": False}
    greedy, grouped_greedy, data = _load_run(root, config_root, source_root,
        _path(config_root, config["baseline_greedy_run_config"]), _path(root, config["baseline_greedy_manifest"]),
        "greedy", None, 0.0)
    ids = [row["id"] for row in data]
    gold = {row["id"]: row["gold_number"] for row in data}
    labels = np.array([int(not is_correct(grouped_greedy[pid][0]["answer"], gold[pid])) for pid in ids], dtype=np.int8)
    if set(labels) != {0, 1}:
        raise ValueError("Power projection requires both greedy label classes")
    names = set(selection["runs"])
    if set(config["manifests"]) != names:
        raise ValueError("Manifest list must contain exactly the final selected runs")
    provenance, scores, runtime, nonnull_p = {}, {}, None, set()
    for name in CONDITIONS:
        if name == "Ap" and selection["Ap_is_A"]:
            provenance[name] = {"alias_of": "A", "run_id": provenance["A"]["run_id"]}
            scores[name] = scores["A"]
            continue
        reference = selection["runs"][name]
        manifest_path = _path(root, config["manifests"][name])
        if manifest_path != _path(root, reference["manifest"]):
            raise ValueError("Input manifest is not the final selected run")
        alpha = selection["common_alpha"] if name in ("C", "D", "E") else selection["alpha_S"] if name == "S" else None
        temperature = selection["T_best"] if name == "Ap" else 0.5
        item, grouped, _ = _load_run(root, config_root, source_root, _path(config_root, reference["run_config"]),
                                    manifest_path, name, alpha, temperature, baseline=(greedy, grouped_greedy))
        current_runtime = item["execution_signature"]["runtime"]
        if runtime is not None:
            differences = _different_paths(runtime, current_runtime)
            if differences:
                raise ValueError("Selected execution runtimes differ: " + ", ".join(differences))
        runtime = current_runtime
        nonnull_p.update({item["sparse_p"]} - {None})
        if len(nonnull_p) > 1:
            raise ValueError("Selected sparse p differs across snapshots")
        provenance[name] = item
        scores[name] = np.array([grouped_entropy([row["answer"] for row in grouped[pid]], group_size=8) for pid in ids])
        if scores[name].shape != (N_DEV, 2):
            raise ValueError("Power scores must have shape 200 x 2")
    bootstrap = shared_subset_bootstrap(labels, scores, COMPARISONS, repeats=ATTEMPTS, analysis_no=50, global_seed=2026)
    if (bootstrap["indices"].shape != (ATTEMPTS, N_DEV) or bootstrap["valid_mask"].shape != (ATTEMPTS,)
            or bootstrap["subset_deltas"].shape != (ATTEMPTS, len(COMPARISONS), 2)
            or bootstrap["point_subset_deltas"].shape != (len(COMPARISONS), 2)
            or bootstrap["attempted"] != ATTEMPTS
            or bootstrap["valid"] != int(bootstrap["valid_mask"].sum())
            or bootstrap["skipped"] != ATTEMPTS - bootstrap["valid"]
            or bootstrap["valid"] < 2):
        raise ValueError("Bootstrap output does not preserve the fixed attempt/condition/subset design")
    comparisons, boost = projection_report(bootstrap)
    aucs = np.array([[roc_auc_score(labels, scores[name][:, j]) for j in range(2)] for name in CONDITIONS])
    stream = io.BytesIO()
    np.savez_compressed(stream, bootstrap_indices=bootstrap["indices"], valid_mask=bootstrap["valid_mask"],
                        bootstrap_subset_deltas=bootstrap["subset_deltas"], attempt_numbers=np.arange(ATTEMPTS),
                        point_subset_deltas=bootstrap["point_subset_deltas"], condition_names=np.array(CONDITIONS),
                        condition_scores=np.stack([scores[name] for name in CONDITIONS]), labels=labels,
                        ordered_ids=np.array(ids), comparison_names=np.array(["-".join(pair) for pair in COMPARISONS]),
                        point_subset_aurocs=aucs)
    npz_raw = stream.getvalue()
    npz_hash = hashlib.sha256(npz_raw).hexdigest()
    inputs = {"config_sha256": sha256(config_path), "selection_sha256": sha256(selection_path), "greedy": greedy,
              "selection_file": str(selection_path),
              "selected_runs": provenance, "ids_sha256": sha256(source_root / "data/splits/dev_select.ids"),
              "processed_train_sha256": sha256(source_root / "data/processed/train.jsonl"), "budget_note": budget}
    report = {"schema_version": 1, "scope": "development_power_projection", "status": "complete",
              "input_sha256": _fingerprint(inputs), "inputs": inputs, "n_dev": N_DEV, "n_test_setting": N_TEST,
              "selected_parameters": {key: selection[key] for key in ("common_alpha", "alpha_S", "T_best", "Ap_is_A")},
              "label_counts": {"positive": int(labels.sum()), "negative": int(N_DEV - labels.sum())},
              "comparison_order": ["-".join(pair) for pair in COMPARISONS], "condition_order": list(CONDITIONS),
              "score_shape": [N_DEV, 2], "bootstrap": {"attempted": bootstrap["attempted"], "valid": bootstrap["valid"],
              "skipped": bootstrap["skipped"], "rng_key": [10, 50], "global_seed": 2026,
              "axes_subset_deltas": ["attempt", "comparison", "subset"], "axes_condition_scores": ["condition", "problem", "subset"]},
              "point_subset_aurocs": {name: aucs[i].tolist() for i, name in enumerate(CONDITIONS)},
              "comparisons": comparisons, "boost_recommendation": boost, "budget_status": "pending_runtime",
              "boost_final": None, "test_rows_final": None, "npz_sha256": npz_hash,
              "assumptions_and_warnings": [
                  "Covariance is extrapolated from two development subsets; negative Cv is retained without clipping.",
                  "MDE=2.80*SE; verdict-7 probability assumes true delta zero and a normal approximation.",
                  "Selected development results do not constitute confirmatory test power or test evidence.",
                  "J8 for S-C and C-Ap is hypothetical; those comparisons retain common first 32 rows under a C/D/E boost.",
                  "No reviewed V9.4 runtime/budget decision was supplied to this projection; assumed 8/12 seconds are not measurements.",
                  "Test raw outputs and test answers were not read."]}
    for name, comparison in comparisons.items():
        if comparison["Cv"] < 0:
            report["assumptions_and_warnings"].append(f"{name}: negative covariance retained; inspect projection statuses.")
    prefix = _path(root, config["output_prefix"])
    paths = [Path(str(prefix) + ".npz"), Path(str(prefix) + ".json"),
             _path(root, config.get("report_file", "reports/power.md"))]
    if len(set(paths)) != 3:
        raise ValueError("Power NPZ, JSON and report paths must be distinct")
    outputs = dict(zip(paths, (npz_raw,
                   (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
                   _markdown(report).encode("utf-8"))))
    for path, raw in outputs.items():
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Power output conflicts with verified inputs or contents: {path}")
    for path, raw in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(project_power(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
