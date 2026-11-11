"""Build fixed pilot grids from complete, verified CPU-readable evidence only.

Run with ``python -m scripts.make_pilot_grid --plan plan.json``. The explicit
JSON plan has ``root``, ``a_run_config`` and ``conditions`` (C and/or S). Each
condition has ``expected_alphas``, ``run_configs`` and ``output``. Optional
``config_root`` locates immutable base/analysis YAML snapshots; ``source_root``
locates immutable dev_base.ids and prompt sources. Both default to root.
Root paths are relative to the plan; run YAML paths are relative to config_root;
output paths are relative to root. Omit S while its p or runs are unavailable.
Including S requires ``sparse_decision: {path: ..., sha256: ...}`` identifying
a completed diagnostic JSON with ``status: complete`` and its fixed ``p``.
All named inputs must validate before any new decision file is written.
Execution-time runtime metadata (and C/S rho hashes) are required; older runs
without this evidence are rejected. Source fingerprint or GPU UUID differences
block automatic equivalence and require review, even on otherwise similar GPUs.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.run_experiment import _validate_run, run_id, sha256
from src.config import config_sha256, load_config
from src.extract import INVALID
from src.pilot_grid import pilot_grid
from src.result_io import load_completed


def _path(root, value):
    return (root / value).resolve()


def _fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _different_paths(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        return [path for key in sorted(set(first) | set(second))
                for path in (_different_paths(first[key], second[key], f"{prefix}.{key}".strip("."))
                             if key in first and key in second else [f"{prefix}.{key}".strip(".")])]
    return [] if first == second else [prefix]


def _signature(base, analysis, source_root, ids_hash):
    normalized = copy.deepcopy(base)
    normalized.pop("budget", None)
    hardware = normalized.get("hardware", {})
    hardware.pop("gpu_name", None)
    if not hardware:
        normalized.pop("hardware", None)
    normalized["noise"]["conditions"]["S"]["p"] = None
    # Only filenames with checked identical contents may vary between snapshots.
    for section, field in (("prompt", "fewshot_file"), ("prompt", "system_file"), ("noise", "rho_file")):
        normalized[section][field] = {"sha256": sha256(_path(source_root, base[section][field]))}
    normalized["stats"]["analysis_list"] = {"content": analysis}
    return {"base": normalized, "ids_sha256": ids_hash}


def _runtime(manifest, base):
    invocations = manifest.get("execution_invocations", [])
    if not invocations or invocations[0].get("existing_problem_files") != 0:
        raise ValueError("Insufficient execution evidence: initial invocation metadata missing")
    required = {"python", "torch", "transformers", "cuda", "gpu_name", "gpu_total_memory",
                "deterministic_algorithms", "tf32_matmul", "tf32_cudnn", "cpu_threads", "cpu_interop_threads",
                "cublas_workspace_config", "model_dtype", "model_revision_loaded", "tokenizer_revision_loaded",
                "sampling", "prefill_optimization_O1", "source_sha256"}
    reference = None
    for invocation in invocations:
        observed = invocation.get("environment", {})
        if invocation.get("scope") != "this_invocation_only" or not required <= set(observed):
            raise ValueError("Insufficient execution evidence: runtime fields missing")
        fixed = {"deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
                 "model_dtype": "torch.float32", "model_revision_loaded": base["model"]["revision"],
                 "tokenizer_revision_loaded": base["model"]["revision"],
                 "sampling": "cpu_float64_inverse_cdf", "prefill_optimization_O1": False}
        if (any(observed[key] != value for key, value in fixed.items())
                or not observed["source_sha256"]
                or any(not observed[key] for key in ("python", "torch", "transformers", "cuda", "gpu_name"))
                or any(not isinstance(observed[key], int) or observed[key] <= 0
                       for key in ("gpu_total_memory", "cpu_threads", "cpu_interop_threads"))
                or observed["cublas_workspace_config"] not in (":4096:8", ":16:8")):
            raise ValueError("Runtime does not match fixed pilot execution requirements")
        # Hostname is provenance, not an execution setting. Source differences
        # remain explicit and block automatic equivalence; never infer benignness.
        core = {key: value for key, value in observed.items() if key != "hostname"}
        if reference is not None:
            differences = _different_paths(reference, core)
            if differences:
                raise ValueError("Runtime changed between invocations: " + ", ".join(differences))
        reference = core
    return reference


def _bank_evidence(root, manifest, base, condition, ids):
    if condition == "A":
        if manifest.get("bank_file") is not None or manifest.get("bank_sha256") is not None:
            raise ValueError("A must not use a noise bank")
        return {}
    bank_path = _path(root, manifest["bank_file"])
    bank_manifest_path = bank_path.with_suffix(".json")
    bank = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    expected = {"condition": condition, "split": "dev_base", "basis": "B0", "seed_set": 0,
                "row_start": 0, "n_rows": 16,
                "sparse_p": base["noise"]["conditions"]["S"]["p"] if condition == "S" else None,
                "dimension": base["noise"]["d"], "global_seed": base["global_seed"],
                "ids": ids, "dtype": "float32"}
    if (any(bank.get(key) != value for key, value in expected.items())
            or bank.get("file_sha256") != manifest.get("bank_sha256")
            or sha256(bank_path) != manifest.get("bank_sha256")):
        raise ValueError(f"{condition} bank identity/hash/p differs from run and base snapshot")
    return {"bank_manifest_sha256": sha256(bank_manifest_path), "bank_sha256": manifest["bank_sha256"]}


def _load_run(run_path, condition, root, config_root, source_root, baseline=None):
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError("Run YAML must be a mapping")
    if (run.get("split") != "dev_base" or run.get("condition") != condition
            or run.get("n_rows") != 16 or run.get("row_start", 0) != 0
            or run.get("temperature") != 0.5 or run.get("seed_set") != 0):
        raise ValueError("Pilot requires dev_base, the named A/C/S condition, rows 0..15, T=0.5, s0")
    if condition not in ("A", "C", "S"):
        raise ValueError("Only A/C/S pilot evidence is allowed")
    if run.get("basis") != (None if condition == "A" else "B0"):
        raise ValueError("Pilot basis must be null for A and B0 for C/S")
    base_path = _path(config_root, run.get("base_config", "configs/base.yaml"))
    base = load_config(base_path)
    _validate_run(run, base)
    if base["decode"]["k_main"] != 8:
        raise ValueError("Pilot requires k_main=8")
    if condition != "A" and not math.isfinite(float(run["alpha"])):
        raise ValueError("Pilot alpha must be finite")
    if condition == "S":
        p = base["noise"]["conditions"]["S"].get("p")
        if p is None or not math.isfinite(float(p)) or not 0 < p <= 1:
            raise ValueError("S p must already be fixed; no p is inferred or run launched")
    analysis_path = _path(config_root, base["stats"]["analysis_list"])
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    ids_path = source_root / "data/splits/dev_base.ids"
    ids = ids_path.read_text(encoding="ascii").splitlines()
    if (not ids or any(not item for item in ids) or len(set(ids)) != len(ids)
            or len(ids) != base["data"]["splits"]["dev_base"]):
        raise ValueError("Unexpected dev_base ID count or duplicates")
    name = run_id(run)
    manifest_path = root / "outputs/manifests" / f"{name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"run_id": name, "condition": condition, "split": "dev_base", "n_rows": 16,
                "row_start": 0, "problem_count": len(ids), "run_config_sha256": sha256(run_path),
                "config_sha256": config_sha256(base, analysis), "model_id": base["model"]["id"],
                "model_revision": base["model"]["revision"],
                "fewshot_sha256": sha256(_path(source_root, base["prompt"]["fewshot_file"])),
                "system_sha256": sha256(_path(source_root, base["prompt"]["system_file"]))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Manifest does not match actual YAML/config/prompt sources: {name}")
    optional = {"run_parameters": run, "base_config_snapshot": base, "alpha": run.get("alpha"),
                "temperature": run["temperature"], "id_version": run.get("id_version", 1)}
    if any(key in manifest and manifest[key] != value for key, value in optional.items()):
        raise ValueError("Manifest parameter snapshots disagree with actual YAML")
    if condition != "A" and manifest.get("rho_sha256") != sha256(_path(source_root, base["noise"]["rho_file"])):
        raise ValueError("Insufficient or mismatching execution-time rho hash")
    if condition != "A" and "rho_snapshot" in manifest:
        rho = json.loads(_path(source_root, base["noise"]["rho_file"]).read_text(encoding="utf-8"))
        if manifest["rho_snapshot"] != rho:
            raise ValueError("Execution-time rho snapshot differs from source")
    grouped = load_completed(root, manifest, ids)
    for items in grouped.values():
        for item in items:
            fields = {"cond": condition, "alpha": run.get("alpha"), "temperature": run["temperature"],
                      "basis": run.get("basis") or "NA", "seed_set": 0, "n_rows": 16,
                      "subset": item["k"] // 8}
            if any(key not in item or item[key] != value for key, value in fields.items()):
                raise ValueError("Raw response parameters disagree with run YAML")
    bank_evidence = _bank_evidence(root, manifest, base, condition, ids)
    signature = _signature(base, analysis, source_root, sha256(ids_path))
    signature["runtime"] = _runtime(manifest, base)
    if baseline is not None:
        previous, reference = baseline
        if ids != list(reference):
            raise ValueError("Pilot IDs differ from A")
        for key in ("model_id", "model_revision", "fewshot_sha256", "system_sha256"):
            if expected[key] != previous[key]:
                raise ValueError(f"Pilot {key} differs from A")
        differences = _different_paths(previous["execution_signature"], signature)
        if differences:
            raise ValueError("Execution signature differs from A: " + ", ".join(differences))
        previous_p = previous["sparse_p"]
        current_p = base["noise"]["conditions"]["S"].get("p")
        if previous_p is not None and current_p is not None and previous_p != current_p:
            raise ValueError("Non-null sparse p differs from A")
        for problem_id, items in grouped.items():
            if any(item["prompt_sha256"] != reference[problem_id][0]["prompt_sha256"] for item in items):
                raise ValueError("Pilot problem prompt differs from A")
    # Distinct INVALID answers count as separate outcomes, as in result_io.
    # Deliberately avoid summarize_run: pilot selection must never use AUROC.
    unique = [len({("invalid", k) if item["answer"] == INVALID else ("valid", item["answer"])
                   for k, item in enumerate(items[:8])}) for items in grouped.values()]
    invalid = sum(item["answer"] == INVALID for items in grouped.values() for item in items)
    evidence = {**expected, "run_config": str(run_path), "base_config": str(base_path),
                "base_yaml_sha256": sha256(base_path), "analysis_yaml_sha256": sha256(analysis_path),
                "ids_sha256": sha256(ids_path), "manifest_sha256": sha256(manifest_path),
                "execution_signature": signature,
                "execution_invocations": manifest["execution_invocations"],
                "sparse_p": base["noise"]["conditions"]["S"].get("p"),
                "rho_sha256": manifest.get("rho_sha256"), **bank_evidence,
                "output_sha256": manifest["output_sha256"], "alpha": run.get("alpha"),
                "mean_unique8_first_subset": sum(unique) / len(unique), "invalid": invalid / (16 * len(ids))}
    return evidence, grouped, base


def make_pilot_grids(plan_path):
    """Validate every named run, then create decisions exclusively or resume identically."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = _path(plan_path.parent, plan["root"])
    config_root = _path(plan_path.parent, plan.get("config_root", str(root)))
    source_root = _path(plan_path.parent, plan.get("source_root", str(root)))
    conditions = plan["conditions"]
    if not conditions or not set(conditions) <= {"C", "S"}:
        raise ValueError("Plan conditions must explicitly name C and/or S")
    a, reference, _ = _load_run(_path(config_root, plan["a_run_config"]), "A", root, config_root, source_root)
    seen = {a["run_id"]}
    nonnull_p = {a["sparse_p"]} - {None}
    sparse = None
    if "S" in conditions:
        sparse_spec = plan["sparse_decision"]
        sparse_path = _path(root, sparse_spec["path"])
        if sha256(sparse_path) != sparse_spec["sha256"]:
            raise ValueError("Sparse diagnostic decision hash does not match plan")
        sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
        if (sparse.get("status") != "complete" or sparse.get("p") is None
                or not math.isfinite(float(sparse["p"])) or not 0 < sparse["p"] <= 1):
            raise ValueError("Sparse diagnostic decision must already be complete with a fixed p")
    pending = {}
    for condition, spec in sorted(conditions.items()):
        rows = []
        configs = spec["run_configs"]
        required_count = 8 if condition == "C" else 6
        if len(configs) != required_count or len(spec["expected_alphas"]) != required_count:
            raise ValueError(f"Entire {condition} pilot requires {required_count} runs")
        for value in configs:
            item, _, base = _load_run(_path(config_root, value), condition, root, config_root, source_root,
                                      baseline=(a, reference))
            if list(spec["expected_alphas"]) != base["noise"][f"pilot_grid_{condition}"]:
                raise ValueError("Expected alphas differ from fixed base pilot grid")
            if item["run_id"] in seen:
                raise ValueError("Duplicate pilot run ID")
            seen.add(item["run_id"])
            nonnull_p.update({item["sparse_p"]} - {None})
            if len(nonnull_p) > 1:
                raise ValueError("Non-null sparse p differs across pilot snapshots")
            if condition == "S":
                if item["sparse_p"] != sparse["p"]:
                    raise ValueError("S p differs from diagnostic decision")
                item["sparse_decision_sha256"] = sparse_spec["sha256"]
            rows.append(item)
        rows.sort(key=lambda row: row["alpha"])
        decision = {"scope": "dev_base_pilot_only", "condition": condition,
                    "plan_sha256": sha256(plan_path), "expected_alphas": spec["expected_alphas"],
                    "inputs": {"A": a, "pilot_runs": rows}}
        decision["input_sha256"] = _fingerprint(decision)
        decision["grid"] = pilot_grid(rows, a["mean_unique8_first_subset"], spec["expected_alphas"])
        output = _path(root, spec["output"])
        if output in pending:
            raise ValueError("Conditions must have different decision output paths")
        pending[output] = decision
    # Check all existing decisions before creating even one new decision.
    for output, decision in pending.items():
        if output.exists() and json.loads(output.read_text(encoding="utf-8")) != decision:
            raise ValueError(f"Existing pilot decision conflicts with inputs or contents: {output}")
    for output, decision in pending.items():
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {str(path): decision for path, decision in pending.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_pilot_grids(args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
