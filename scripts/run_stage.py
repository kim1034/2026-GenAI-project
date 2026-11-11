"""Run a reviewed, explicit command queue with durable logs and a single-worker lock."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_experiment import json_write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    logs = ROOT / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    prerequisite = plan.get("after_stage")
    if prerequisite:
        deadline = time.monotonic() + plan.get("wait_timeout_seconds", 10800)
        while True:
            previous_path = logs / f"{prerequisite}.state.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                if previous["status"] == "complete":
                    break
                if previous["status"] == "failed":
                    raise RuntimeError(f"Prerequisite stage {prerequisite} failed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Prerequisite stage {prerequisite} did not finish in time")
            time.sleep(30)
        # Previous worker writes complete immediately before releasing its lock.
        time.sleep(1)
    lock = (logs / "stage.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = logs / (plan["name"] + ".state.json")
    state = {"name": plan["name"], "status": "running", "steps": []}
    try:
        if plan.get("require_initial_probe"):
            probe = json.loads((ROOT / "outputs/diagnostics/gpu_probe.json").read_text())
            if probe["status"] != "passed_initial_probe":
                raise RuntimeError("Initial GPU probe has not passed")
        for index, command in enumerate(plan["commands"]):
            # Explicit argv, no shell interpolation; underlying scripts validate resume outputs.
            step = {"index": index, "argv": command, "started_at_utc": datetime.now(timezone.utc).isoformat()}
            state["steps"].append(step)
            json_write(state_path, state)
            with (logs / f'{plan["name"]}_{index:02d}.log').open("a", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            step.update(returncode=result.returncode, ended_at_utc=datetime.now(timezone.utc).isoformat())
            json_write(state_path, state)
            if result.returncode:
                raise RuntimeError(f"Step {index} failed with code {result.returncode}")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        json_write(state_path, state)
        lock.close()


if __name__ == "__main__":
    main()
