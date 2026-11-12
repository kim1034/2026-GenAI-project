"""Prepare fixed test banks on CPU after sealing, alongside GPU inference."""

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import yaml

from scripts.budget_lock_preflight import BASE, FINAL, RECEIPT, _path, _read, verify_lock
from scripts.run_experiment import json_write, sha256, run_id
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def precompute(root, *, workers=8):
    root = Path(root).resolve()
    if type(workers) is not int or workers < 1:
        raise ValueError("Positive CPU worker count required")
    seal = verify_lock(root)
    verified = _read(root / "outputs/lock_verified.json")
    if verified != seal:
        raise ValueError("Stored seal verification does not match the current LOCK")
    receipt, final = _read(root / RECEIPT), _read(root / FINAL)
    base = load_config(root / BASE)
    ids_path = root / base["data"]["splits_dir"] / "test.ids"
    ids = ids_path.read_text(encoding="ascii").splitlines()
    if (len(ids) != 1319 or len(set(ids)) != 1319
            or sha256(ids_path) != receipt["evidence_sha256"][ids_path.relative_to(root).as_posix()]):
        raise ValueError("Test-bank IDs differ from LOCK")
    specs = receipt["required_test_runs"]
    if [s["condition"] for s in specs] != ["greedy", "C", "D", "E"]:
        raise ValueError("Unexpected registered test conditions")
    commands = []
    for spec in specs[1:]:
        condition, name = spec["condition"], spec["run_config"]
        path = _path(root, name)
        run = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {"base_config": BASE, "split": "test", "condition": condition,
                    "alpha": receipt["common_alpha"], "temperature": .5, "basis": "B0",
                    "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
        if (run != expected or final["run_configs"]["test"][condition] != name
                or sha256(path) != spec["run_config_sha256"] or run_id(run) != spec["run_id"]
                or spec["rows"] != 16 or spec["problem_count"] != 1319):
            raise ValueError("Test bank request differs from sealed fixed design")
        commands.append([sys.executable, "-u", "-m", "scripts.build_bank", name, "--workers", str(workers)])
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return {"status": "complete", "scope": "CPU_test_bank_preparation_only", "workers": workers,
            "conditions": ["C", "D", "E"], "problems": 1319, "rows": 16,
            "seal": seal, "test_question_or_output_records_parsed": False}


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logs = ROOT / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    # This worker lock is independent of the GPU's global stage lock.
    with (logs / "test_bank_precompute.worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        state = logs / "test_bank_precompute.state.json"
        try:
            if args.wait:
                json_write(state, {"status": "waiting_for_verified_LOCK", "pid": os.getpid()})
                while not (ROOT / "outputs/lock_verified.json").is_file():
                    time.sleep(30)
            result = precompute(ROOT, workers=args.workers)
        except BaseException as error:
            json_write(state, {"status": "failed", "error": repr(error)})
            raise
        json_write(state, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
