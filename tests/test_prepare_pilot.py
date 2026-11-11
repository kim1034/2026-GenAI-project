import json
from pathlib import Path

import pytest
import yaml

from scripts.prepare_pilot import prepare
from scripts.run_experiment import sha256


def test_pilot_snapshots_are_complete_stable_and_conflict_protected(tmp_path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    for folder in ("configs", "calib"):
        (tmp_path / folder).mkdir()
    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (tmp_path / "configs/analysis.yaml").write_text("analysis: fixed\n")
    (tmp_path / "calib/rho.json").write_text('{"rho_by_layer_1idx": {"20": 1}}')
    evidence = tmp_path / "calib/evidence.json"
    evidence.write_text('{"verified": true}')
    sparse = {"status": "complete", "p": .005, "base_config_sha256": sha256(tmp_path / "configs/base.yaml"),
              "inputs_sha256": {"calib/evidence.json": sha256(evidence)}}
    (tmp_path / "calib/sparse_decision.json").write_text(json.dumps(sparse))
    destination = tmp_path / "configs/generated/pilot"
    first = prepare(tmp_path, destination)
    assert first["run_count"] == 15
    assert prepare(tmp_path, destination) == first
    plan = json.loads((destination / "grid_plan.json").read_text())
    assert len(plan["conditions"]["C"]["run_configs"]) == 8
    assert len(plan["conditions"]["S"]["run_configs"]) == 6
    assert yaml.safe_load((destination / "base.yaml").read_text(encoding="utf-8"))["noise"]["conditions"]["S"]["p"] == .005
    (destination / "base.yaml").write_text("changed")
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(tmp_path, destination)
