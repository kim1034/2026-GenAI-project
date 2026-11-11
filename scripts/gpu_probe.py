"""Development-only GPU feasibility probe; never counts as a full G1 gate."""

import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts.run_experiment import json_write
from src.config import load_config
from src.data import load_split
from src.decode import decode
from src.hook import MLPNoise
from src.model import load_model
from src.prompt import load_messages, render_prompt
from src.rng import rng_for

ROOT = Path(__file__).resolve().parents[1]


def main():
    report_path = ROOT / "outputs/diagnostics/gpu_probe.json"
    report = {"status": "running", "full_gate": False, "checks": []}
    json_write(report_path, report)
    try:
        base = load_config(ROOT / "configs/base.yaml")
        row = load_split(ROOT, "dev_base", base)[0]
        model, tokenizer = load_model(base)
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), problem_id=row["id"])
        messages = load_messages(ROOT / base["prompt"]["system_file"],
                                 ROOT / base["prompt"]["fewshot_file"], row["question"])
        _, ids, prompt_sha = render_prompt(tokenizer, messages)
        prompt = torch.tensor([ids], device="cuda")
        report.update(prompt_tokens=len(ids), prompt_sha256=prompt_sha)
        # A short input avoids allocating a full-prompt all-vocabulary tensor.
        with torch.inference_mode():
            short = prompt[:, -16:]
            reference = model(input_ids=short, use_cache=False).logits[:, -1, :]
            last = model.lm_head(model.model(input_ids=short, use_cache=False).last_hidden_state[:, -1:, :])[:, -1, :]
            difference = (reference - last).abs().max().item()
            torch.testing.assert_close(reference, last, rtol=1e-5, atol=1e-5)
        report["last_logits_max_abs_error"] = difference
        del reference, last
        for n_rows, temperature, repeats in [(1, 0.0, 2), (16, 0.5, 2), (32, 0.5, 2)]:
            hashes = []
            for repeat in range(repeats):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                uniforms = np.stack([rng_for(9, 0, 0, int(row["id"].split("-")[1]), k).random(512)
                                     for k in range(n_rows)])
                torch.cuda.synchronize()
                started = time.monotonic()
                tokens, lengths, finishes = decode(model, prompt, n_rows=n_rows,
                    temperature=temperature, uniforms=uniforms, max_new_tokens=512,
                    eos_token_ids=base["decode"]["eos_token_ids"])
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                payload = {"tokens": tokens.tolist(), "lengths": lengths, "finishes": finishes}
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                hashes.append(digest)
                check = {"rows": n_rows, "repeat": repeat, "seconds": elapsed,
                         "peak_bytes": torch.cuda.max_memory_allocated(), "sha256": digest,
                         "max_tokens": max(lengths)}
                report["checks"].append(check)
                json_write(ROOT / f"outputs/diagnostics/probe_rows{n_rows}_repeat{repeat}.json", payload)
                json_write(report_path, report)
                print(json.dumps(check), flush=True)
            if len(set(hashes)) != 1:
                raise RuntimeError(f"nonidentical repeat at {n_rows} rows")
        report["status"] = "passed_initial_probe"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        json_write(report_path, report)


if __name__ == "__main__":
    main()
