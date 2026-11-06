"""Calibrate per-layer activation RMS on dev_base greedy responses."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.calib import RMSRecorder, combine_rms
from src.config import load_config
from src.data import load_split
from src.prompt import load_messages, render_prompt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs/base.yaml")
    if not config["model"]["revision"]:
        raise SystemExit("Pin model.revision before GPU calibration.")
    rows = load_split(ROOT, "dev_base", config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from src.decode import decode
    from src.model import load_model

    model, tokenizer = load_model(config)
    recorder = RMSRecorder(model, config["noise"]["layers_0idx"])
    contributions = ROOT / "outputs" / "calib" / "dev_base"
    contributions.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for row in rows:
            path = contributions / f"{row['id']}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("model_revision") != config["model"]["revision"]:
                    raise RuntimeError(f"stale calibration contribution: {row['id']}")
                continue
            recorder.reset()
            messages = load_messages(ROOT / config["prompt"]["system_file"],
                                     ROOT / config["prompt"]["fewshot_file"], row["question"])
            _, token_ids, prompt_sha = render_prompt(tokenizer, messages)
            prompt_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            decode(model, prompt_ids, n_rows=1, temperature=0.0,
                   uniforms=np.zeros((1, config["decode"]["max_new_tokens"]), dtype=np.float64),
                   max_new_tokens=config["decode"]["max_new_tokens"],
                   eos_token_ids=config["decode"]["eos_token_ids"])
            content = {"id": row["id"], "prompt_sha256": prompt_sha,
                       "model_revision": config["model"]["revision"], **recorder.snapshot()}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
    finally:
        recorder.remove()
    snapshots = []
    for row in rows:
        snapshots.append(json.loads((contributions / f"{row['id']}.json").read_text(encoding="utf-8")))
    rho = combine_rms(snapshots)
    ids_path = ROOT / "data" / "splits" / "dev_base.ids"
    report = {
        "rho_by_layer_1idx": rho,
        "dev_base_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "problem_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = ROOT / config["noise"]["rho_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        for key in ("rho_by_layer_1idx", "dev_base_ids_sha256", "model_id", "model_revision", "problem_count"):
            if previous[key] != report[key]:
                raise RuntimeError("existing rho calibration differs; inspect before replacing it")
    else:
        output.write_text(text, encoding="utf-8")
    print(f"calibrated {len(rho)} layers from {len(rows)} dev_base problems: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
