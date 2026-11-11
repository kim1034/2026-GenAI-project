"""Actual-model decoding, hook and repeatability checks for the G1/G2 gates."""

import gc
import hashlib
import json
import os
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from scripts.run_experiment import ROOT, _layer_scales, _uniforms, json_write
from src.config import load_config
from src.data import load_split
from src.decode import decode
from src.hook import MLPNoise
from src.model import load_model
from src.noise import coefficients, direction
from src.prompt import load_messages, render_prompt


def hook_checks(model, config, prompt):
    layers = config["noise"]["layers_0idx"]
    ids = prompt[:, -16:].repeat(2, 1)
    vectors = torch.tensor(np.stack([direction(coefficients("C", d=config["noise"]["d"], seed_set=0,
        split_key=0, original_index=0, response_index=k, global_seed=config["global_seed"])) for k in range(2)]), device="cuda")
    scales = _layer_scales(config, "C", 0.1)
    raw, altered = {}, {}
    before_handles = [model.model.layers[l].mlp.act_fn.register_forward_hook(
        lambda module, inputs, output, l=l: raw.__setitem__(l, output.detach().clone())) for l in layers]
    hook = MLPNoise(model, layers, scales)
    after_handles = [model.model.layers[l].mlp.act_fn.register_forward_hook(
        lambda module, inputs, output, l=l: altered.__setitem__(l, output.detach().clone())) for l in layers]
    checks = {}
    try:
        with torch.inference_mode():
            baseline = model.model(input_ids=ids, use_cache=False).last_hidden_state
            with hook.with_base(torch.zeros_like(vectors)):
                zero = model.model(input_ids=ids, use_cache=False).last_hidden_state
            checks["zero_is_bitwise_identity"] = torch.equal(baseline, zero)
            hook.calls = 0
            with hook.with_base(vectors):
                first = model.model(input_ids=ids, use_cache=True)
                checks["nine_calls_per_forward"] = hook.calls == 9
                for l in layers:
                    torch.testing.assert_close(altered[l] - raw[l], (scales[l] * vectors[:, None, :]).expand_as(raw[l]), rtol=1e-4, atol=1e-6)
                checks["prefill_shared_vector_every_position"] = True
                next_out = model.model(input_ids=ids[:, -1:], past_key_values=first.past_key_values, use_cache=True)
                for l in layers:
                    torch.testing.assert_close(altered[l] - raw[l], (scales[l] * vectors[:, None, :]).expand_as(raw[l]), rtol=1e-4, atol=1e-6)
                checks["decode_shared_vector"] = True
            with hook.with_base(vectors):
                repeat = model.model(input_ids=ids, use_cache=False).last_hidden_state
            checks["repeat_bitwise"] = torch.equal(first.last_hidden_state, repeat)
            checks["perturbation_changes_output"] = not torch.equal(baseline, repeat)
            with hook.with_base(vectors.flip(0)):
                swapped = model.model(input_ids=ids, use_cache=False).last_hidden_state
            checks["row_swap_no_leakage"] = torch.equal(repeat.flip(0), swapped)
    finally:
        hook.remove()
        for handle in before_handles + after_handles:
            handle.remove()
    early = {}
    handles = [model.model.layers[l].register_forward_hook(
        lambda module, inputs, output, l=l: early.__setitem__(l, output[0].detach().clone())) for l in range(27)]
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False)
            reference = dict(early)
            final_hook = MLPNoise(model, [27], {27: scales[27]})
            try:
                with final_hook.with_base(vectors):
                    model.model(input_ids=ids, use_cache=False)
            finally:
                final_hook.remove()
            checks["last_layer_does_not_change_earlier_layers"] = all(torch.equal(reference[l], early[l]) for l in range(27))
    finally:
        for handle in handles:
            handle.remove()
    if not all(checks.values()):
        raise RuntimeError(f"Hook check failed: {checks}")
    return checks


def main():
    config = load_config(ROOT / "configs/base.yaml")
    rows = load_split(ROOT, "dev_base", config)
    report_path = ROOT / "outputs/diagnostics/gpu_validation.json"
    if report_path.exists():
        raise RuntimeError("Validation evidence already exists; inspect before rerunning")
    report = {"status": "running", "checks": [], "full_gate": False,
              "remaining_outside_scope": ["manual extraction review", "development label prevalence", "bank checks", "gradient diagnostic"]}
    json_write(report_path, report)
    try:
        model, tokenizer = load_model(config)
        def prompt_for(row):
            messages = load_messages(ROOT / config["prompt"]["system_file"], ROOT / config["prompt"]["fewshot_file"], row["question"])
            _, tokens, digest = render_prompt(tokenizer, messages)
            return torch.tensor([tokens], device="cuda"), digest
        prompt, _ = prompt_for(rows[0])
        report["hooks"] = hook_checks(model, config, prompt)
        json_write(report_path, report)
        from transformers import GenerationConfig
        generation = GenerationConfig(do_sample=False, max_new_tokens=512, use_cache=True,
            repetition_penalty=1.0, eos_token_id=config["decode"]["eos_token_ids"],
            pad_token_id=config["decode"]["eos_token_ids"][0])
        for index, row in enumerate(rows[:20]):
            prompt, digest = prompt_for(row)
            previous = json.loads((ROOT / f'outputs/samples/dev_base_greedy_aNA_T0.00_NA_s0_k1/{row["id"]}.json').read_text())[0]
            tokens, lengths, finishes = decode(model, prompt, n_rows=1, temperature=0,
                uniforms=np.zeros((1, 512)), max_new_tokens=512, eos_token_ids=config["decode"]["eos_token_ids"])
            if digest != previous["prompt_sha256"] or tokens[0, :lengths[0]].tolist() != previous["token_ids"]:
                raise RuntimeError(f"Greedy repeat mismatch at {row['id']}")
            if index < 5:
                with torch.inference_mode():
                    standard = model.generate(input_ids=prompt, generation_config=generation)
                if standard[0, prompt.shape[1]:].tolist() != previous["token_ids"]:
                    raise RuntimeError(f"Standard generate greedy mismatch at {row['id']}")
            report["checks"].append({"kind": "greedy_repeat", "id": row["id"], "standard_generate_checked": index < 5})
            json_write(report_path, report)
        for condition, alpha in [("A", None), ("C", 0.1)]:
            for n_rows in (16, 32):
                for row in rows[:10]:
                    prompt, prompt_sha = prompt_for(row)
                    noise = None
                    if condition == "C":
                        noise = torch.tensor(np.stack([direction(coefficients("C", d=config["noise"]["d"], seed_set=0,
                            split_key=0, original_index=int(row["id"].split("-")[1]), response_index=k,
                            global_seed=config["global_seed"])) for k in range(n_rows)]), device="cuda")
                    digests = []
                    for repeat in range(2):
                        hook = MLPNoise(model, config["noise"]["layers_0idx"], _layer_scales(config, condition, alpha))
                        run = {"seed_set": 0, "n_rows": n_rows}
                        started = time.monotonic()
                        try:
                            tokens, lengths, finishes = decode(model, prompt, n_rows=n_rows, temperature=0.5,
                                uniforms=_uniforms(config, run, row["id"]), max_new_tokens=512,
                                eos_token_ids=config["decode"]["eos_token_ids"], noise_hook=hook, noise_base=noise)
                        finally:
                            hook.remove()
                        payload = {"tokens": tokens.tolist(), "lengths": lengths, "finishes": finishes, "prompt_sha256": prompt_sha}
                        serialized = json.dumps(payload, sort_keys=True).encode()
                        digest = hashlib.sha256(serialized).hexdigest()
                        digests.append(digest)
                        json_write(ROOT / f'outputs/diagnostics/g1_raw/{condition}_{n_rows}_{row["id"]}_repeat{repeat}.json', payload)
                        check = {"kind": "sample_repeat", "condition": condition, "n_rows": n_rows, "id": row["id"],
                                 "repeat": repeat, "seconds": time.monotonic() - started, "output_sha256": digest}
                        report["checks"].append(check)
                        json_write(report_path, report)
                        print(json.dumps(check), flush=True)
                    if len(set(digests)) != 1:
                        raise RuntimeError(f"Nonidentical {condition}/{n_rows}/{row['id']}")
                    gc.collect()
        report["status"] = "passed_computational_checks"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        json_write(report_path, report)


if __name__ == "__main__":
    main()
