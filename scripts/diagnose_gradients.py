"""Run the prespecified gradient diagnostic, recording failures explicitly."""

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from scripts.run_experiment import ROOT, json_write, sha256
from src.config import load_config
from src.data import load_split
from src.gradient_diag import GradientDiagnostic
from src.noise import direction
from src.prompt import load_messages, render_prompt
from src.rng import rng_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev_base", "dev_select", "dev_anal", "test"], required=True)
    parser.add_argument("--fd-only", action="store_true", help="Only first three dev_base problems; never a full diagnostic")
    args = parser.parse_args()
    if args.fd_only and args.split != "dev_base":
        parser.error("finite-difference pilot is dev_base only")
    config = load_config(ROOT / "configs/base.yaml")
    rows = load_split(ROOT, args.split, config)
    if args.fd_only:
        rows = rows[:3]
    folder = ROOT / "outputs/diag"
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / f'{args.split}{"_fd" if args.fd_only else ""}_status.json'
    report = {"status": "running", "split": args.split, "full_split": not args.fd_only, "completed": 0,
              "fd_rng_key": "(17, original_index)", "fd_rule": "both gradients relative_error < 0.05, h=1e-3*sqrt(d)"}
    identity = {"rho_sha256": sha256(ROOT / config["noise"]["rho_file"]),
                "source_sha256": sha256(ROOT / "src/gradient_diag.py"), "model_revision": config["model"]["revision"]}
    json_write(report_path, report)
    diagnostic = None
    started = time.monotonic()
    try:
        from src.model import load_model
        model, tokenizer = load_model(config)
        rho = json.loads((ROOT / config["noise"]["rho_file"]).read_text())["rho_by_layer_1idx"]
        diagnostic = GradientDiagnostic(model, tokenizer, config, rho)
        summaries, gradients_m, gradients_seq = [], [], []
        for index, row in enumerate(rows):
            greedy_path = ROOT / f'outputs/samples/{args.split}_greedy_aNA_T0.00_NA_s0_k1/{row["id"]}.json'
            greedy = json.loads(greedy_path.read_text())[0]
            entry_path = folder / args.split / f'{row["id"]}.json'
            npz_path = entry_path.with_suffix(".npz")
            expected = {**identity, "greedy_sha256": sha256(greedy_path)}
            if entry_path.exists():
                info = json.loads(entry_path.read_text())
                if any(info.get(k) != v for k, v in expected.items()):
                    raise RuntimeError("Diagnostic source or input changed; preserve previous outputs before rerun")
                arrays = np.load(npz_path)
                g_m, g_seq = arrays["g_m"], arrays["g_seq"]
            else:
                messages = load_messages(ROOT / config["prompt"]["system_file"], ROOT / config["prompt"]["fewshot_file"], row["question"])
                _, prompt, prompt_sha = render_prompt(tokenizer, messages)
                if prompt_sha != greedy["prompt_sha256"]:
                    raise RuntimeError("Diagnostic prompt differs from greedy prompt")
                u = None
                if args.split == "dev_base" and index < 3:
                    u = direction(rng_for(17, int(row["id"].split("-")[1])).standard_normal(config["noise"]["d"]))
                info, g_m, g_seq = diagnostic.compute(prompt, greedy["token_ids"], fd_direction=u)
                info.update(expected, id=row["id"], y=int(not greedy["correct"]), n_new_tokens=greedy["n_new_tokens"])
                entry_path.parent.mkdir(parents=True, exist_ok=True)
                with npz_path.with_suffix(".tmp").open("wb") as handle:
                    np.savez(handle, g_m=g_m, g_seq=g_seq)
                os.replace(npz_path.with_suffix(".tmp"), npz_path)
                json_write(entry_path, info)
            summaries.append(info)
            gradients_m.append(g_m)
            gradients_seq.append(g_seq)
            report.update(completed=index + 1, elapsed_seconds=time.monotonic() - started)
            json_write(report_path, report)
            print(json.dumps({"id": row["id"], "completed": index + 1, "deff": info["deff_g_m"], "fd": info.get("finite_difference")}), flush=True)
            if "finite_difference" in info and not all(x["passed"] for x in info["finite_difference"].values()):
                raise RuntimeError("T-DIAG-1 finite-difference error exceeds 5%; diagnostic gate failed")
        if not args.fd_only:
            np.save(folder / f"{args.split}_g_m.npy", np.stack(gradients_m))
            np.save(folder / f"{args.split}_g_seq.npy", np.stack(gradients_seq))
            keys = [key for key in summaries[0] if key not in ("finite_difference", "top_ids")]
            with (folder / f"{args.split}_diag.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(summaries)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if diagnostic:
            diagnostic.close()
        json_write(report_path, report)


if __name__ == "__main__":
    main()
