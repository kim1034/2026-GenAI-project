import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import prepare_budget_pilot as preparation
from scripts.run_experiment import _validate_run, run_id, sha256


@pytest.fixture
def project(tmp_path):
    for name in ("configs/base.yaml", "configs/analysis.yaml", "configs/budget120_protocol.json",
                 "prompts/system.txt", "prompts/gsm8k_8shot.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    rho = tmp_path / "calib/rho.json"
    rho.parent.mkdir()
    rho.write_text('{"rho_by_layer_1idx": {"20": 1}}', encoding="utf-8")
    for split, count in {"dev_base": 50, "dev_select": 200, "dev_anal": 50, "test": 1319}.items():
        ids = tmp_path / "data/splits" / f"{split}.ids"
        ids.parent.mkdir(parents=True, exist_ok=True)
        ids.write_text("".join(f"{split}-{i}\n" for i in range(count)), encoding="ascii")
    return tmp_path, tmp_path / "configs/generated/budget120/pilot"


def test_fixed_nine_run_plan_and_unchanged_scientific_settings(project):
    root, destination = project
    original = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    first = preparation.prepare(root, destination)
    assert preparation.prepare(root, destination) == first
    assert first["run_count"] == 9
    base = yaml.safe_load((destination / "base.yaml").read_text(encoding="utf-8"))
    for section in ("model", "data", "prompt", "selection", "label", "score"):
        assert base[section] == original[section]
    expected = copy.deepcopy(original["noise"])
    expected["rho_file"] = "configs/generated/budget120/pilot/rho.json"
    assert base["noise"] == expected
    expected_decode = {**original["decode"], "k_generate_test": 16, "k_generate_rehearsal": 16}
    assert base["decode"] == expected_decode
    assert base["doc_version"] == "1.3"
    assert base["protocol_amendment"] == "budget120-v1.4"
    analysis = yaml.safe_load((destination / "analysis.yaml").read_text(encoding="utf-8"))
    original_analysis = yaml.safe_load((root / "configs/analysis.yaml").read_text(encoding="utf-8"))
    assert analysis["primary"] == original_analysis["primary"]
    assert analysis["secondary"] == {}
    assert analysis["descriptive"]["D_minus_E"] == original_analysis["secondary"]["D_minus_E"]
    assert analysis["descriptive"]["E_minus_C"]["bootstrap_key"] is None
    assert analysis["original_secondary_registry"] == original_analysis["secondary"]
    plan = json.loads((destination / "grid_plan.json").read_text())
    assert (destination / plan["root"]).resolve() == root
    assert set(plan["conditions"]) == {"C"}
    assert plan["conditions"]["C"]["expected_alphas"] == original["noise"]["pilot_grid_C"]
    paths = [plan["a_run_config"], *plan["conditions"]["C"]["run_configs"]]
    assert len(paths) == len({Path(path).stem for path in paths}) == 9
    for path in paths:
        run = yaml.safe_load((root / path).read_text())
        assert run["id_version"] == 2
        assert run["n_rows"] == 16 and run["row_start"] == 0
        assert run_id(run) == Path(path).stem
        _validate_run(run, base)
    assert (destination / "rho.json").read_bytes() == (root / "calib/rho.json").read_bytes()
    assert all(sha256(root / path) == digest for path, digest in plan["inputs_sha256"].items())
    stage = json.loads((destination / "stage_plan.json").read_text())
    assert stage["commands"][0] == ["-m", "scripts.build_bank", paths[1]]
    assert stage["commands"][1:10] == [["-m", "scripts.run_experiment", path] for path in paths]
    assert stage["commands"][-1] == ["-m", "scripts.make_pilot_grid", "--plan", first["grid_plan"]]
    assert not (root / "outputs").exists()
    protocol = json.loads((destination / "protocol.json").read_text())
    assert protocol["selection"]["indices_zero_based"] == [0, 2, 5]
    assert "original sorted six-point grid" in protocol["selection"]["edge"]["extrapolation_ratio"]
    assert protocol["analysis"]["confirmatory"] == ["D-C"]
    assert protocol["test"]["require_all_conditions_before_auroc"] == ["C", "D", "E"]
    assert protocol["resource_revision_count"] == 1
    assert protocol["separate_followup_design_count"] == 0
    assert protocol["ledger"] == "configs/budget120_ledger.json"


def test_conflict_preflights_entire_batch_and_protects_legacy(project):
    root, destination = project
    destination.mkdir(parents=True)
    (destination / "stage_plan.json").write_text("conflict")
    with pytest.raises(ValueError, match="Conflicting"):
        preparation.prepare(root, destination)
    assert list(destination.iterdir()) == [destination / "stage_plan.json"]
    with pytest.raises(ValueError, match="dedicated"):
        preparation.prepare(root, root / "configs/generated/pilot")


def test_colliding_run_ids_write_nothing(project, monkeypatch):
    root, destination = project
    monkeypatch.setattr(preparation, "run_id", lambda run: "collision")
    with pytest.raises(ValueError, match="colliding run IDs"):
        preparation.prepare(root, destination)
    assert not destination.exists()


def test_old_output_with_same_id_is_rejected_before_any_snapshot(project):
    root, destination = project
    baseline = {"split": "dev_base", "condition": "A", "alpha": None, "temperature": .5,
                "basis": None, "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
    manifest = root / "outputs/manifests" / f"{run_id(baseline)}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_config_sha256": "old", "config_sha256": "old"}))
    with pytest.raises(ValueError, match="Existing output"):
        preparation.prepare(root, destination)
    assert not destination.exists()


def test_rho_requires_original_calibration_path_without_legacy_fallback(project):
    root, destination = project
    (root / "calib/rho.json").unlink()
    legacy = root / "configs/generated/pilot/rho.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    with pytest.raises(FileNotFoundError):
        preparation.prepare(root, destination)
    assert not destination.exists()
