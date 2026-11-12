import copy
import json

import pytest
import yaml

from scripts import budget_lock_preflight as lock
from scripts.run_experiment import sha256


@pytest.fixture
def report():
    return {"schema_version": 1, "status": "complete", "protocol_amendment": "budget120-v1.4",
        "scope": "rehearsal", "confirmatory": False, "split": "dev_anal", "n": 50,
        "common_alpha": .1, "practical_delta": .02, "analysis_unavailable_reason": None,
        "bootstrap": {"keys": {"D-C": [10, 1], "D-E": [10, 2], "E-C": None},
            "attempts_per_registered_key": 10000, "global_seed": 2026, "se_ddof": 1, "ci": "percentile_95"},
        "comparisons": {name: {"rng_key": key, "confirmatory": False, "attempted": 10000 if key else 0,
            "valid": 9999 if key else 0, "skipped": 1 if key else 0,
            "ci95": [-.01, .02] if key else None, "se": .01 if key else None,
            "verdict": 8 if name == "D-C" else None}
            for name, key in [("D-C", [10, 1]), ("D-E", [10, 2]), ("E-C", None)]},
        "arithmetic_residual": 0.,
        "metrics": {c: {"label_classes_complete": True, "auroc_subsets": [.5, .7], "auroc": .6} for c in "CDE"},
        "k_curve": [{"condition": c, "K": 8, "auroc": .6} for c in "CDE"]}


def test_valid_rehearsal_is_calculation_only(report):
    lock.validate_rehearsal(report, .1)


@pytest.mark.parametrize("mutation", ["confirmatory", "missing_labels", "wrong_key", "ec_inference", "k8", "attempts"])
def test_scientific_gate_rejects_invalid_rehearsal(report, mutation):
    if mutation == "confirmatory":
        report["confirmatory"] = True
    elif mutation == "missing_labels":
        report["analysis_unavailable_reason"] = "one class"
    elif mutation == "wrong_key":
        report["comparisons"]["D-E"]["rng_key"] = [10, 1]
    elif mutation == "ec_inference":
        report["comparisons"]["E-C"]["ci95"] = [0, 1]
    elif mutation == "k8":
        report["k_curve"][0]["auroc"] = .7
    else:
        report["comparisons"]["D-C"]["valid"] = 9998
    with pytest.raises(ValueError):
        lock.validate_rehearsal(report, .1)


@pytest.fixture
def final(tmp_path):
    folder = tmp_path / lock.FINAL
    folder.parent.mkdir(parents=True)
    configs, commands = {}, []
    for c in ("greedy", "C", "D", "E"):
        greedy = c == "greedy"
        run = {"base_config": lock.BASE, "split": "test", "condition": c,
            "alpha": None if greedy else .1, "temperature": 0.0 if greedy else .5,
            "basis": None if greedy else "B0", "seed_set": 0, "row_start": 0,
            "n_rows": 1 if greedy else 16, "id_version": 2}
        path = folder.parent / f"test_{c}.yaml"
        path.write_text(yaml.safe_dump(run), encoding="utf-8")
        name = path.relative_to(tmp_path).as_posix()
        configs[c] = name
        if not greedy:
            commands.append(["-m", "scripts.build_bank", name])
        commands.append(["-m", "scripts.run_experiment", name])
    (folder.parent / "stage_test.json").write_text(json.dumps({"commands": commands}))
    return tmp_path, {"common_alpha": .1, "run_configs": {"test": configs}}


def test_test_metadata_validated_without_opening_dataset(final):
    root, plan = final
    result = lock.validate_test_plan(root, plan)
    assert [row["condition"] for row in result] == ["greedy", "C", "D", "E"]
    assert [row["rows"] for row in result] == [1, 16, 16, 16]
    assert not (root / "data").exists()


def test_existing_test_output_blocks_lock(final):
    root, plan = final
    required = lock.validate_test_plan(root, plan)
    path = root / "outputs/manifests" / (required[0]["run_id"] + ".json")
    path.parent.mkdir(parents=True)
    path.write_text("not even parsed")
    with pytest.raises(ValueError, match="already exists"):
        lock.validate_test_plan(root, plan)


def test_wrong_test_order_blocks_lock(final):
    root, plan = final
    stage = root / lock.FINAL
    path = stage.parent / "stage_test.json"
    value = json.loads(path.read_text())
    value["commands"].reverse()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="order"):
        lock.validate_test_plan(root, plan)


def test_unregistered_test_output_also_blocks_lock(final):
    root, plan = final
    path = root / "outputs/samples/test_legacy_run.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("sealed content must not be parsed")
    with pytest.raises(ValueError, match="already exists"):
        lock.validate_test_plan(root, plan)


def test_locked_evidence_mutation_rejected_before_seal_call(tmp_path, monkeypatch):
    item = tmp_path / "evidence.json"
    item.write_text("original")
    path = tmp_path / lock.RECEIPT
    path.parent.mkdir()
    path.write_text(json.dumps({"status": "ready_for_lock", "evidence_sha256": {"evidence.json": sha256(item)}}))
    (tmp_path / "LOCK.md").write_text("preflight_sha256: " + sha256(path))
    item.write_text("mutated")
    monkeypatch.setattr(lock, "assert_test_locked", lambda *args: pytest.fail("seal called before evidence check"))
    with pytest.raises(ValueError, match="evidence changed"):
        lock.verify_lock(tmp_path)


def test_dirty_tree_does_not_open_development_or_test_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "_git", lambda *args: " M pending.py")
    monkeypatch.setattr(lock, "prepare", lambda *args: pytest.fail("dirty preflight reached evidence preparation"))
    with pytest.raises(ValueError, match="Commit"):
        lock.preflight(tmp_path)
