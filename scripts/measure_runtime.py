"""Development-only runtime and hook validation before any pilot/selection runs."""

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
from scripts.budget import calculate
from src.config import load_config
from src.data import load_split
from src.decode import decode
from src.hook import MLPNoise
from src.model import load_model
from src.noise import coefficients, direction
from src.prompt import load_messages, render_prompt


def main():
    base = load_config(ROOT / "configs/base.yaml")
    rows = load_split(ROOT, "dev_base", base)[:3]
    report_path = ROOT / "outputs/diagnostics/runtime.json"
    if report_path.exists():
        raise RuntimeError("Runtime evidence already exists; inspect and preserve it before rerunning")
    report = {"status": "running", "checks": [], "full_gate": False,
              "settings": "V9.1: C alpha=0.1, C alpha=1.6, no noise T=1.7; first three dev_base problems",
              "cpu_threads": torch.get_num_threads()}
    json_write(report_path, report)
    try:
        model, tokenizer = load_model(base)
        # T-HK-1 and actual perturbation shape/magnitude on real activations.
        row = rows[0]
        messages = load_messages(ROOT / base["prompt"]["system_file"], ROOT / base["prompt"]["fewshot_file"], row["question"])
        _, prompt, _ = render_prompt(tokenizer, messages)
        ids = torch.tensor([prompt[-16:]], device="cuda")
        with torch.inference_mode():
            reference = model.model(input_ids=ids, use_cache=False).last_hidden_state
            hook = MLPNoise(model, base["noise"]["layers_0idx"], _layer_scales(base, "C", 0.1))
            with hook.with_base(torch.zeros(1, base["noise"]["d"], device="cuda")):
                zero = model.model(input_ids=ids, use_cache=False).last_hidden_state
            if not torch.equal(reference, zero):
                raise RuntimeError("Zero perturbation changes the model output")
            hook.remove()
        report["zero_hook_bitwise_equal"] = True
        del zero, reference
        for condition, alpha, temperature in [("C", 0.1, 0.5), ("C", 1.6, 0.5), ("Ap", None, 1.7)]:
            for n_rows in (16, 32):
                for row in rows:
                    run = {"condition": condition, "alpha": alpha, "temperature": temperature, "n_rows": n_rows, "seed_set": 0}
                    messages = load_messages(ROOT / base["prompt"]["system_file"], ROOT / base["prompt"]["fewshot_file"], row["question"])
                    _, prompt, prompt_sha = render_prompt(tokenizer, messages)
                    ids = torch.tensor([prompt], device="cuda")
                    noise = None
                    if condition == "C":
                        noise = torch.tensor(np.stack([direction(coefficients("C", d=base["noise"]["d"], seed_set=0,
                            split_key=0, original_index=int(row["id"].split("-")[1]), response_index=k,
                            global_seed=base["global_seed"])) for k in range(n_rows)]), device="cuda")
                    hook = MLPNoise(model, base["noise"]["layers_0idx"], _layer_scales(base, condition, alpha))
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    try:
                        tokens, lengths, finishes = decode(model, ids, n_rows=n_rows, temperature=temperature,
                            uniforms=_uniforms(base, run, row["id"]), max_new_tokens=512,
                            eos_token_ids=base["decode"]["eos_token_ids"], noise_hook=hook, noise_base=noise)
                    finally:
                        hook.remove()
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    expected_calls = len(base["noise"]["layers_0idx"]) * len(tokens[0]) if noise is not None else 0
                    if hook.calls != expected_calls:
                        raise RuntimeError(f"Hook calls {hook.calls} != expected {expected_calls}")
                    check = {"id": row["id"], **run, "seconds": elapsed, "peak_bytes": torch.cuda.max_memory_allocated(),
                             "hook_calls": hook.calls, "max_tokens": max(lengths), "prompt_sha256": prompt_sha,
                             "output_sha256": hashlib.sha256(tokens.tobytes()).hexdigest()}
                    report["checks"].append(check)
                    json_write(report_path, report)
                    print(json.dumps(check), flush=True)
        # Use the worst setting's mean, not the fastest problem, for budgeting.
        timings = {}
        for n in (16, 32):
            groups = [[c["seconds"] for c in report["checks"] if c["n_rows"] == n and c["condition"] == cond and c["alpha"] == alpha]
                      for cond, alpha in [("C", 0.1), ("C", 1.6), ("Ap", None)]]
            timings[n] = max(float(np.mean(group)) for group in groups)
        report["budget_worst_setting_mean"] = calculate(base, rows16_seconds=timings[16], rows32_seconds=timings[32])
        report["seconds_by_rows"] = timings
        report["status"] = "complete"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        json_write(report_path, report)


if __name__ == "__main__":
    main()
