"""Finish G1/G2 computational checks for the prospective 1/16-row scope.

Legacy reports and raw files are read-only. Verified checkpoints are reused;
only missing checks run. A separate report and standalone stage lock are used.
"""

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_checkpoint(payload, check, *, condition, problem_id, repeat, prompt_sha, eos_ids):
    if (check.get("kind") != "sample_repeat" or check.get("condition") != condition
            or check.get("n_rows") != 16 or check.get("id") != problem_id or check.get("repeat") != repeat
            or payload.get("prompt_sha256") != prompt_sha or canonical_hash(payload) != check.get("output_sha256")):
        raise ValueError("Legacy checkpoint identity/prompt/hash mismatch")
    tokens, lengths, finishes = payload["tokens"], payload["lengths"], payload["finishes"]
    if len(tokens) != 16 or len(lengths) != 16 or len(finishes) != 16:
        raise ValueError("Expected exactly 16 response rows")
    widths = {len(row) for row in tokens}
    if len(widths) != 1 or not 1 <= next(iter(widths)) <= 512:
        raise ValueError("Invalid rectangular token buffer")
    for row, length, finish in zip(tokens, lengths, finishes):
        if (type(length) is not int or not 1 <= length <= len(row)
                or any(type(token) is not int or token < 0 for token in row)
                or finish not in ("eos", "length") or (finish == "length" and length != 512)
                or (finish == "eos" and row[length - 1] not in eos_ids)):
            raise ValueError("Invalid token/length/finish checkpoint")
    return canonical_hash(payload)


def validate_scope(protocol):
    if (protocol.get("protocol_amendment") != "budget120-v1.4"
            or protocol["pilot"]["rows"] != 16 or protocol["rehearsal"]["rows"] != 16
            or protocol["test"]["rows_per_condition"] != 16 or protocol["test"]["boost"] is not False):
        raise ValueError("Only the prospectively amended 1/16-row protocol is supported")


def main():
    import fcntl
    import numpy as np
    import torch
    from transformers import GenerationConfig
    from scripts.run_experiment import ROOT, _layer_scales, _uniforms, json_write, sha256
    from scripts.validate_gpu import hook_checks
    from src.config import load_config
    from src.data import load_split
    from src.decode import decode
    from src.hook import MLPNoise
    from src.model import load_model
    from src.noise import coefficients, direction
    from src.prompt import load_messages, render_prompt

    protocol_path = ROOT / "configs/budget120_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validate_scope(protocol)
    config = load_config(ROOT / "configs/base.yaml")
    rows = load_split(ROOT, "dev_base", config)
    logs = ROOT / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / "stage.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = ROOT / "outputs/diagnostics/gpu_validation_v14.json"
    state_path = logs / "gpu_validation_v14_20261112.state.json"
    if report_path.exists() or state_path.exists():
        raise RuntimeError("Preserve existing v1.4 evidence before any explicit recovery")
    legacy_path = ROOT / "outputs/diagnostics/gpu_validation.json"
    legacy = json.loads(legacy_path.read_text())
    report = {"status": "running", "full_gate": False, "protocol_amendment": "budget120-v1.4",
              "scope_rows": [1, 16], "checks": [], "legacy_report_sha256": sha256(legacy_path),
              "legacy_report_status_preserved": legacy.get("status"), "protocol_sha256": sha256(protocol_path),
              "configuration_snapshot": config, "base_yaml_sha256": sha256(ROOT / "configs/base.yaml"),
              "rho_sha256": sha256(ROOT / config["noise"]["rho_file"]),
              "source_hash_observation_scope": "current files only; not retroactive hashes of the old process",
              "reuse_basis": "Original validator settings and pinned model; fresh prompt checks; canonical raw hashes; verified tokenizer equivalence.",
              "unused_32_row_files_preserved": [str(p.relative_to(ROOT)) for p in (ROOT / "outputs/diagnostics/g1_raw").glob("*_32_*.json")],
              "remaining_outside_scope": legacy.get("remaining_outside_scope", [])}
    state = {"name": "gpu_validation_v14_20261112", "status": "running", "pid": os.getpid(), "report": str(report_path.relative_to(ROOT))}
    json_write(report_path, report)
    json_write(state_path, state)
    try:
        model, tokenizer = load_model(config)
        def prompt_for(row):
            messages = load_messages(ROOT / config["prompt"]["system_file"], ROOT / config["prompt"]["fewshot_file"], row["question"])
            _, tokens, digest = render_prompt(tokenizer, messages)
            return torch.tensor([tokens], device="cuda"), digest
        prompt, _ = prompt_for(rows[0])
        old_hooks = legacy.get("hooks", {})
        expected_hooks = {"zero_is_bitwise_identity", "nine_calls_per_forward", "prefill_shared_vector_every_position",
                          "decode_shared_vector", "repeat_bitwise", "perturbation_changes_output", "row_swap_no_leakage",
                          "last_layer_does_not_change_earlier_layers"}
        if set(old_hooks) == expected_hooks and all(value is True for value in old_hooks.values()):
            report["hooks"] = old_hooks
            report["hooks_reused"] = True
        else:
            report["hooks"] = hook_checks(model, config, prompt)
            report["hooks_reused"] = False
        generation = GenerationConfig(do_sample=False, max_new_tokens=512, use_cache=True, repetition_penalty=1.0,
            eos_token_id=config["decode"]["eos_token_ids"], pad_token_id=config["decode"]["eos_token_ids"][0])
        old_greedy = {item["id"]: item for item in legacy["checks"] if item.get("kind") == "greedy_repeat"}
        for index, row in enumerate(rows[:20]):
            prompt, digest = prompt_for(row)
            path = ROOT / f'outputs/samples/dev_base_greedy_aNA_T0.00_NA_s0_k1/{row["id"]}.json'
            previous = json.loads(path.read_text())[0]
            if previous["prompt_sha256"] != digest:
                raise RuntimeError("Fresh tokenizer prompt differs from preserved greedy evidence")
            old = old_greedy.get(row["id"])
            if old is None:
                tokens, lengths, _ = decode(model, prompt, n_rows=1, temperature=0, uniforms=np.zeros((1, 512)),
                    max_new_tokens=512, eos_token_ids=config["decode"]["eos_token_ids"])
                if tokens[0, :lengths[0]].tolist() != previous["token_ids"]:
                    raise RuntimeError(f"Greedy repeat mismatch at {row['id']}")
            if index < 5 and (old is None or old.get("standard_generate_checked") is not True):
                with torch.inference_mode():
                    standard = model.generate(input_ids=prompt, generation_config=generation)
                if standard[0, prompt.shape[1]:].tolist() != previous["token_ids"]:
                    raise RuntimeError(f"Standard greedy mismatch at {row['id']}")
            report["checks"].append({"kind": "greedy_repeat", "id": row["id"], "standard_generate_checked": index < 5,
                "greedy_reused": old is not None, "baseline_file_sha256": sha256(path), "prompt_sha256": digest})
        old_samples = {}
        for check in legacy["checks"]:
            if check.get("kind") == "sample_repeat" and check.get("n_rows") == 16:
                key = (check["condition"], check["id"], check["repeat"])
                if key in old_samples:
                    raise ValueError("Duplicate legacy checkpoint metadata")
                old_samples[key] = check
        json_write(report_path, report)
        for condition, alpha in (("A", None), ("C", .1)):
            for row in rows[:10]:
                prompt, prompt_sha = prompt_for(row)
                digests = []
                noise = None
                for repeat in range(2):
                    key = (condition, row["id"], repeat)
                    old_path = ROOT / f'outputs/diagnostics/g1_raw/{condition}_16_{row["id"]}_repeat{repeat}.json'
                    if key in old_samples:
                        payload = json.loads(old_path.read_text())
                        digest = validate_checkpoint(payload, old_samples[key], condition=condition, problem_id=row["id"],
                            repeat=repeat, prompt_sha=prompt_sha, eos_ids=config["decode"]["eos_token_ids"])
                        check = {**old_samples[key], "reused": True, "raw_file": str(old_path.relative_to(ROOT)), "raw_file_sha256": sha256(old_path)}
                    else:
                        if old_path.exists():
                            raise ValueError("Uncommitted legacy raw evidence exists; preserve and inspect")
                        if condition == "C" and noise is None:
                            noise = torch.tensor(np.stack([direction(coefficients("C", d=config["noise"]["d"], seed_set=0,
                                split_key=0, original_index=int(row["id"].split("-")[1]), response_index=k,
                                global_seed=config["global_seed"])) for k in range(16)]), device="cuda")
                        hook = MLPNoise(model, config["noise"]["layers_0idx"], _layer_scales(config, condition, alpha))
                        started = time.monotonic()
                        try:
                            tokens, lengths, finishes = decode(model, prompt, n_rows=16, temperature=.5,
                                uniforms=_uniforms(config, {"seed_set": 0, "n_rows": 16}, row["id"]),
                                max_new_tokens=512, eos_token_ids=config["decode"]["eos_token_ids"], noise_hook=hook, noise_base=noise)
                        finally:
                            hook.remove()
                        payload = {"tokens": tokens.tolist(), "lengths": lengths, "finishes": finishes, "prompt_sha256": prompt_sha}
                        digest = canonical_hash(payload)
                        raw_path = ROOT / f'outputs/diagnostics/g1_v14_raw/{condition}_16_{row["id"]}_repeat{repeat}.json'
                        if raw_path.exists():
                            raise ValueError("New raw evidence already exists; refusing overwrite")
                        json_write(raw_path, payload)
                        check = {"kind": "sample_repeat", "condition": condition, "n_rows": 16, "id": row["id"], "repeat": repeat,
                                 "seconds": time.monotonic() - started, "output_sha256": digest, "reused": False,
                                 "raw_file": str(raw_path.relative_to(ROOT)), "raw_file_sha256": sha256(raw_path)}
                    digests.append(digest)
                    report["checks"].append(check)
                    json_write(report_path, report)
                    print(json.dumps(check), flush=True)
                if len(set(digests)) != 1:
                    raise RuntimeError(f"Nonidentical {condition}/16/{row['id']}; no automatic retry")
        report["status"] = "passed_computational_checks"
        state["status"] = "complete"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        state.update(status="failed", error=report["error"])
        raise
    finally:
        json_write(report_path, report)
        json_write(state_path, state)
        lock.close()


if __name__ == "__main__":
    main()
