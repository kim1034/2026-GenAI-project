from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import build_bank as frontend
from scripts import precompute_locked_test_banks as worker
from scripts.run_experiment import sha256, run_id


def test_bank_lock_identity_ignores_alpha(monkeypatch, tmp_path):
    paths = []
    @contextmanager
    def lock(root, path):
        paths.append(path)
        yield
    monkeypatch.setattr(frontend, "bank_write_lock", lock)
    monkeypatch.setattr(frontend, "build_bank", lambda *args, **kwargs: (tmp_path / "b.npy", tmp_path / "b.json"))
    run = {"condition": "D", "split": "test", "seed_set": 0, "row_start": 0, "n_rows": 16, "alpha": .1}
    frontend.build_locked(tmp_path, ["test-1"], {}, run, workers=8)
    frontend.build_locked(tmp_path, ["test-1"], {}, {**run, "alpha": .2}, workers=1)
    assert paths[0] == paths[1] == tmp_path / "bank/D/B0/s0/test_rows0-15.npy"


@pytest.mark.skipif(os.name != "posix", reason="GPU worker uses Linux OS flock")
def test_real_process_writers_are_serialized(tmp_path):
    code = """import sys,time
from pathlib import Path
from scripts.build_bank import bank_write_lock
root=Path(sys.argv[1]); bank=root/'bank/test.npy'
with bank_write_lock(root,bank):
    with (root/'events').open('a') as f: f.write('enter\\n')
    time.sleep(.2)
    with (root/'events').open('a') as f: f.write('leave\\n')
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)]) for _ in range(2)]
    for process in processes:
        assert process.wait(timeout=20) == 0
    assert (tmp_path / "events").read_text().splitlines() == ["enter", "leave", "enter", "leave"]


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    path = root / worker.BASE
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    ids = root / base["data"]["splits_dir"] / "test.ids"
    ids.parent.mkdir(parents=True)
    ids.write_text("".join(f"test-{i:05d}\n" for i in range(1319)), encoding="ascii")
    specs, configs = [], {}
    for condition in ("greedy", "C", "D", "E"):
        run = {"base_config": worker.BASE, "split": "test", "condition": condition,
            "alpha": None if condition == "greedy" else .1, "temperature": 0.0 if condition == "greedy" else .5,
            "basis": None if condition == "greedy" else "B0", "seed_set": 0, "row_start": 0,
            "n_rows": 1 if condition == "greedy" else 16, "id_version": 2}
        path = root / "configs" / f"{condition}.yaml"
        path.write_text(yaml.safe_dump(run), encoding="utf-8")
        name = path.relative_to(root).as_posix()
        configs[condition] = name
        specs.append({"condition": condition, "run_config": name, "run_config_sha256": sha256(path),
                      "run_id": run_id(run), "rows": run["n_rows"], "problem_count": 1319})
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    seal = {"status": "verified_test_seal", "LOCK_sha256": "seal"}
    write(root / "outputs/lock_verified.json", seal)
    write(root / worker.RECEIPT, {"common_alpha": .1, "required_test_runs": specs,
        "evidence_sha256": {ids.relative_to(root).as_posix(): sha256(ids)}})
    write(root / worker.FINAL, {"run_configs": {"test": configs}})
    monkeypatch.setattr(worker, "verify_lock", lambda root: seal)
    calls = []
    monkeypatch.setattr(worker.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    return root, calls


def test_precompute_uses_only_fixed_cpu_bank_commands_after_seal(sealed):
    root, calls = sealed
    result = worker.precompute(root)
    assert len(calls) == 3
    for condition, (argv, kwargs) in zip("CDE", calls):
        assert argv[2:] == ["-m", "scripts.build_bank", f"configs/{condition}.yaml", "--workers", "8"]
        assert kwargs == {"cwd": root, "check": True}
    assert result["test_question_or_output_records_parsed"] is False
    assert not (root / "data/processed/test.jsonl").exists()


def test_bad_seal_prevents_bank_commands(sealed, monkeypatch):
    root, calls = sealed
    def reject(root):
        raise PermissionError("sealed")
    monkeypatch.setattr(worker, "verify_lock", reject)
    with pytest.raises(PermissionError):
        worker.precompute(root)
    assert calls == []


def test_changed_run_rejected_before_any_bank_write(sealed):
    root, calls = sealed
    path = root / "configs/E.yaml"
    run = yaml.safe_load(path.read_text(encoding="utf-8"))
    run["n_rows"] = 32
    path.write_text(yaml.safe_dump(run), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed fixed design"):
        worker.precompute(root)
    assert calls == []


def test_cpu_bank_failure_propagates_without_marking_complete(sealed, monkeypatch):
    root, _ = sealed
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(worker.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        worker.precompute(root)
