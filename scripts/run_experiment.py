"""Run one prespecified GSM8K condition from a single run configuration file."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.bank import bank_path, load_bank
from src.config import config_sha256, load_config
from src.data import load_split
from src.extract import INVALID, extract_lenient, extract_main, is_correct
from src.prompt import load_messages, render_prompt
from src.rng import rng_for


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def tokenizer_execution_metadata(tokenizer) -> dict:
    from src.model import TOKENIZER_PROVENANCE_ATTRIBUTE
    provenance = getattr(tokenizer, TOKENIZER_PROVENANCE_ATTRIBUTE, None)
    verified = isinstance(provenance, dict) and provenance.get("verified") is True
    return {
        "tokenizer_hf_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_revision_loaded": provenance["revision"] if verified else None,
        "tokenizer_snapshot_files_sha256": dict(provenance["files_sha256"]) if verified else None,
        "tokenizer_vocab_sha256": provenance["vocab_sha256"] if verified else None,
    }


def execution_environment(model, tokenizer) -> dict:
    import torch
    import transformers
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    sources = [ROOT / "scripts/run_experiment.py", *(ROOT / "src").glob("*.py")]
    return {
        "python": platform.python_version(), "hostname": socket.gethostname(),
        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "gpu_total_memory": properties.total_memory,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_revision_loaded": getattr(model.config, "_commit_hash", None),
        **tokenizer_execution_metadata(tokenizer),
        "sampling": "cpu_float64_inverse_cdf", "prefill_optimization_O1": False,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted(sources)},
    }


def run_id(run: dict) -> str:
    alpha = "NA" if run.get("alpha") is None else f"{run['alpha']:.4f}"
    basis = run.get("basis") or "NA"
    name = (f"{run['split']}_{run['condition']}_a{alpha}_T{run['temperature']:.2f}_"
            f"{basis}_s{run['seed_set']}_k{run['n_rows']}")
    if run.get("row_start", 0):
        end = run["row_start"] + run["n_rows"] - 1
        name += f"_rows{run['row_start']}-{end}"
    version = run.get("id_version", 1)
    if version not in (1, 2):
        raise ValueError("unsupported run ID version")
    if version == 2:
        exact = {"alpha": None if run.get("alpha") is None else float(run["alpha"]),
                 "temperature": float(run["temperature"])}
        canonical = json.dumps(exact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        name += "_v2_" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]
    return name


def _validate_run(run: dict, base: dict) -> None:
    condition = run["condition"]
    if condition not in set(base["noise"]["conditions"]) | {"greedy"}:
        raise ValueError(f"unknown condition: {condition}")
    if run["split"] not in ("dev_base", "dev_select", "dev_anal", "test"):
        raise ValueError("unknown split")
    rows = int(run["n_rows"])
    row_start = int(run.get("row_start", 0))
    if condition == "greedy":
        if rows != 1 or run["temperature"] != 0 or row_start:
            raise ValueError("greedy labels require one unperturbed row and temperature zero")
    else:
        expected = base["decode"]["k_generate_rehearsal"] if run["split"] == "dev_anal" else (
            base["decode"]["k_generate_test"] if run["split"] == "test" else base["decode"]["k_generate_dev"]
        )
        if rows != expected or row_start not in (0, 32):
            raise ValueError("response row count or starting row violates the v1.2 design")
        if row_start == 32 and (run["split"] != "test" or condition not in ("C", "D", "E")):
            raise ValueError("rows 32–63 are reserved for test P1 boost of C, D and E")
    if condition in ("A", "Ap", "greedy"):
        if run.get("alpha") is not None:
            raise ValueError("temperature-only runs have no alpha")
    elif run.get("alpha") is None or run["alpha"] <= 0:
        raise ValueError("noise runs require a positive selected alpha")


def _layer_scales(base: dict, condition: str, alpha: float | None) -> dict[int, float]:
    layers = base["noise"]["layers_0idx"]
    if alpha is None:
        return {layer: 0.0 for layer in layers}
    if condition in ("B", "Bc"):
        return {layer: float(alpha) for layer in layers}
    rho_path = ROOT / base["noise"]["rho_file"]
    rho = json.loads(rho_path.read_text(encoding="utf-8"))["rho_by_layer_1idx"]
    d = int(base["noise"]["d"])
    return {layer: float(alpha) * float(rho[str(layer + 1)]) * np.sqrt(d) for layer in layers}


def _uniforms(base: dict, run: dict, problem_id: str) -> np.ndarray:
    split_key = 1 if problem_id.startswith("test-") else 0
    original_index = int(problem_id.split("-", 1)[1])
    maximum = int(base["decode"]["max_new_tokens"])
    start = int(run.get("row_start", 0))
    return np.stack([
        rng_for(9, run["seed_set"], split_key, original_index, start + k,
                global_seed=base["global_seed"]).random(maximum)
        for k in range(run["n_rows"])
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config", type=Path, help="one YAML file describing the run")
    args = parser.parse_args()
    run_path = args.run_config.resolve()
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    base = load_config(ROOT / run.get("base_config", "configs/base.yaml"))
    _validate_run(run, base)
    rows = load_split(ROOT, run["split"], base)  # test is sealed here, before loading the model
    if not base["model"]["revision"]:
        raise SystemExit("Pin model.revision before running an experiment; no output was written.")

    name = run_id(run)
    output_dir = ROOT / "outputs" / "samples" / name
    manifest_path = ROOT / "outputs" / "manifests" / f"{name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = yaml.safe_load((ROOT / base["stats"]["analysis_list"]).read_text(encoding="utf-8"))
    config_hash = config_sha256(base, analysis)
    run_hash = sha256(run_path)
    problem_ids = [row["id"] for row in rows]
    bank = None
    rho_path = ROOT / base["noise"]["rho_file"] if run["condition"] not in ("A", "Ap", "B", "Bc", "greedy") else None
    rho_digest = sha256(rho_path) if rho_path else None
    bank_file = None
    bank_manifest_path = None
    if run["condition"] not in ("A", "Ap", "greedy"):
        bank_file = bank_path(ROOT, run["condition"], run["basis"], run["seed_set"],
                              run["split"], run.get("row_start", 0), run["n_rows"])
        bank_manifest_path = bank_file.with_suffix(".json")
        bank = load_bank(bank_file, bank_manifest_path, problem_ids,
                         row_start=run.get("row_start", 0), n_rows=run["n_rows"])

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if rho_path and manifest.get("rho_sha256") != rho_digest:
            raise RuntimeError("rho calibration changed or its original hash was not recorded")
        if manifest["run_config_sha256"] != run_hash or manifest["config_sha256"] != config_hash:
            raise RuntimeError("existing run ID has a different configuration")
        if manifest["fewshot_sha256"] != sha256(ROOT / base["prompt"]["fewshot_file"]) or manifest["system_sha256"] != sha256(ROOT / base["prompt"]["system_file"]):
            raise RuntimeError("prompt files changed during this run")
        if bank_file and manifest["bank_sha256"] != sha256(bank_file):
            raise RuntimeError("noise bank changed during this run")
        if manifest.get("status") == "complete":
            final_path = ROOT / "outputs" / "samples" / f"{name}.jsonl"
            if not final_path.is_file() or sha256(final_path) != manifest.get("output_sha256"):
                raise RuntimeError("completed run output no longer matches its manifest")
            print(f"{name}: already complete ({manifest['sample_count']} responses)")
            return
    else:
        manifest = {
            "run_id": name, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_parameters": run, "base_config_snapshot": base,
            "alpha": run.get("alpha"), "temperature": run["temperature"],
            "id_version": run.get("id_version", 1),
            "rho_sha256": rho_digest,
            "rho_snapshot": json.loads(rho_path.read_text(encoding="utf-8")) if rho_path else None,
            "condition": run["condition"], "split": run["split"], "n_rows": run["n_rows"],
            "row_start": run.get("row_start", 0), "problem_count": len(rows),
            "config_sha256": config_hash, "run_config_sha256": run_hash,
            "model_id": base["model"]["id"], "model_revision": base["model"]["revision"],
            "git_head": git_head(), "bank_file": str(bank_file.relative_to(ROOT)) if bank_file else None,
            "bank_sha256": sha256(bank_file) if bank_file else None,
            "fewshot_sha256": sha256(ROOT / base["prompt"]["fewshot_file"]),
            "system_sha256": sha256(ROOT / base["prompt"]["system_file"]),
        }
        json_write(manifest_path, manifest)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from src.decode import decode
    from src.hook import MLPNoise
    from src.model import load_model

    model, tokenizer = load_model(base)
    observed = execution_environment(model, tokenizer)
    manifest.setdefault("execution_invocations", []).append({
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "this_invocation_only", "environment": observed,
        "existing_problem_files": len(list(output_dir.glob("*.json"))),
    })
    json_write(manifest_path, manifest)
    scales = _layer_scales(base, run["condition"], run.get("alpha"))
    if rho_path and sha256(rho_path) != rho_digest:
        raise RuntimeError("rho calibration changed while initializing the run")
    hook = MLPNoise(model, base["noise"]["layers_0idx"], scales)
    started = time.monotonic()
    completed = 0
    try:
        for bank_row, row in enumerate(rows):
            problem_started = time.monotonic()
            result_path = output_dir / f"{row['id']}.json"
            if result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if len(existing) != run["n_rows"] or any(item["run_id"] != name for item in existing):
                    raise RuntimeError(f"incomplete existing problem output: {row['id']}")
                completed += 1
                continue
            messages = load_messages(ROOT / base["prompt"]["system_file"],
                                     ROOT / base["prompt"]["fewshot_file"], row["question"])
            _, token_ids, prompt_sha = render_prompt(tokenizer, messages)
            prompt_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            noise_base = (torch.from_numpy(np.array(bank[bank_row], copy=True)).to("cuda")
                          if bank is not None else None)
            tokens, lengths, finishes = decode(
                model, prompt_ids, n_rows=run["n_rows"], temperature=run["temperature"],
                uniforms=_uniforms(base, run, row["id"]),
                max_new_tokens=base["decode"]["max_new_tokens"],
                eos_token_ids=base["decode"]["eos_token_ids"],
                noise_hook=hook, noise_base=noise_base,
            )
            result = []
            for index in range(run["n_rows"]):
                k = run.get("row_start", 0) + index
                raw_tokens = tokens[index, :lengths[index]].tolist()
                if finishes[index] == "eos":
                    raw_tokens = raw_tokens[:-1]
                text = tokenizer.decode(raw_tokens, skip_special_tokens=True)
                answer = extract_main(text)
                result.append({
                    "id": row["id"], "k": k, "subset": k // base["decode"]["k_main"],
                    "cond": run["condition"], "alpha": run.get("alpha"),
                    "basis": run.get("basis") or "NA", "seed_set": run["seed_set"],
                    "temperature": run["temperature"], "n_rows": run["n_rows"],
                    "text": text, "n_new_tokens": lengths[index],
                    "token_ids": tokens[index, :lengths[index]].tolist(),
                    "finish_reason": finishes[index], "answer": answer,
                    "extract_path": "anchor" if answer != INVALID else "invalid",
                    "answer_lenient": extract_lenient(text),
                    "correct": is_correct(answer, row["gold_number"]),
                    "prompt_sha256": prompt_sha,
                    "bank_file": str(bank_file.relative_to(ROOT)) if bank_file else None,
                    "bank_row": bank_row if bank is not None else None,
                    "run_id": name,
                })
            json_write(result_path, result)
            completed += 1
            print(json.dumps({"run_id": name, "completed": completed, "total": len(rows),
                              "problem_id": row["id"], "seconds": time.monotonic() - problem_started},
                             ensure_ascii=False), flush=True)
            if completed % 10 == 0:
                manifest.update({"completed_problems": completed, "elapsed_seconds": time.monotonic() - started})
                json_write(manifest_path, manifest)

        final_path = ROOT / "outputs" / "samples" / f"{name}.jsonl"
        temporary = final_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                result = json.loads((output_dir / f"{row['id']}.json").read_text(encoding="utf-8"))
                for item in result:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if final_path.exists():
            if sha256(final_path) != sha256(temporary):
                raise RuntimeError("existing aggregate output differs from resumed run")
            temporary.unlink()
        else:
            os.replace(temporary, final_path)
        manifest.update({
            "status": "complete", "completed_problems": completed,
            "sample_count": completed * run["n_rows"], "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": sha256(final_path),
            "gpu_max_memory_bytes": torch.cuda.max_memory_allocated(),
            "hook_calls": hook.calls,
        })
        json_write(manifest_path, manifest)
        print(f"{name}: {completed} problems, {manifest['sample_count']} responses")
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
