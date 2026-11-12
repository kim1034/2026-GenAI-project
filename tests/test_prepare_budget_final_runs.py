import json
from pathlib import Path

import pytest
import yaml

from scripts import prepare_budget_final_runs as preparation
from scripts.run_experiment import run_id, sha256
from src.config import config_sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["protocol_amendment"] = "budget120-v1.4"
    base["decode"]["k_generate_test"] = base["decode"]["k_generate_rehearsal"] = 16
    base_path = tmp_path / preparation.BASE_CONFIG
    base_path.parent.mkdir(parents=True)
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    write_json(tmp_path / "configs/runs/dev_anal_greedy.yaml", {"split": "dev_anal", "condition": "greedy",
        "n_rows": 1, "temperature": 0, "alpha": None, "seed_set": 0})
    analysis_path = tmp_path / base["stats"]["analysis_list"]
    write_json(analysis_path, {})
    write_json(base_path.parent / "protocol.json", {"protocol_amendment": "budget120-v1.4"})
    for key in ("system_file", "fewshot_file"):
        path = tmp_path / base["prompt"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed synthetic prompt", encoding="utf-8")
    rho = tmp_path / base["noise"]["rho_file"]
    write_json(rho, {"rho_by_layer_1idx": {"20": 1}})
    ids = [f"train-{index:05d}" for index in range(200)]
    ids_path = tmp_path / "data/splits/dev_select.ids"
    ids_path.parent.mkdir(parents=True)
    ids_path.write_text("\n".join(ids) + "\n", encoding="ascii")
    train = tmp_path / "data/processed/train.jsonl"
    train.parent.mkdir()
    train.write_text("".join(json.dumps({"id": pid, "question": "synthetic", "gold_number": "1"}) + "\n"
                             for pid in ids), encoding="utf-8")
    environment = {"python": "3.12", "torch": "2.6", "transformers": "4.50", "cuda": "12.4",
                   "gpu_name": "A5000", "gpu_uuid": "synthetic", "gpu_total_memory": 24000,
                   "deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
                   "cpu_threads": 4, "cpu_interop_threads": 4, "cublas_workspace_config": ":4096:8",
                   "model_dtype": "torch.float32", "model_revision_loaded": base["model"]["revision"],
                   "tokenizer_revision_loaded": base["model"]["revision"],
                   "sampling": "cpu_float64_inverse_cdf", "prefill_optimization_O1": False,
                   "source_sha256": {"src/decode.py": "0" * 64}}
    refs = {"A": {}}
    for condition in ("C", "D", "E"):
        run = {"base_config": preparation.BASE_CONFIG, "split": "dev_select", "condition": condition,
               "alpha": .1, "temperature": .5, "basis": "B0", "seed_set": 0,
               "row_start": 0, "n_rows": 16, "id_version": 2}
        name = run_id(run)
        config = tmp_path / f"configs/{name}.yaml"
        config.write_text(yaml.safe_dump(run), encoding="utf-8")
        output = tmp_path / f"outputs/samples/{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps({"id": pid, "k": k, "subset": k // 8, "run_id": name,
            "prompt_sha256": f"prompt-{pid}", "cond": condition, "alpha": .1, "temperature": .5,
            "n_rows": 16, "basis": "B0", "seed_set": 0, "answer": "1"}) + "\n"
            for pid in ids for k in range(16)), encoding="utf-8")
        bank = tmp_path / f"bank/{condition}.npy"
        bank.parent.mkdir(exist_ok=True)
        bank.write_bytes(b"synthetic bank")
        write_json(bank.with_suffix(".json"), {"condition": condition, "split": "dev_select", "basis": "B0",
            "seed_set": 0, "row_start": 0, "n_rows": 16, "dimension": base["noise"]["d"], "global_seed": 2026,
            "ids": ids, "dtype": "float32", "sparse_p": None, "file_sha256": sha256(bank)})
        manifest = tmp_path / f"outputs/manifests/{name}.json"
        write_json(manifest, {"status": "complete", "run_id": name, "condition": condition, "split": "dev_select",
            "n_rows": 16, "row_start": 0, "problem_count": 200, "sample_count": 3200,
            "run_config_sha256": sha256(config), "output_sha256": sha256(output),
            "config_sha256": config_sha256(base, {}), "model_id": base["model"]["id"],
            "model_revision": base["model"]["revision"], "base_config_snapshot": base, "run_parameters": run,
            "fewshot_sha256": sha256(tmp_path / base["prompt"]["fewshot_file"]),
            "system_sha256": sha256(tmp_path / base["prompt"]["system_file"]),
            "bank_file": bank.relative_to(tmp_path).as_posix(), "bank_sha256": sha256(bank),
            "rho_sha256": sha256(rho), "execution_invocations": [{"scope": "this_invocation_only",
                "existing_problem_files": 0, "environment": environment}]})
        refs[condition] = {"run_config": config.relative_to(tmp_path).as_posix(), "run_config_sha256": sha256(config),
                           "manifest": manifest.relative_to(tmp_path).as_posix(), "manifest_sha256": sha256(manifest),
                           "output_sha256": sha256(output)}
    selection = tmp_path / "outputs/analysis/budget120_selection.json"
    decision = {"schema_version": 1, "protocol_amendment": "budget120-v1.4", "status": "complete",
                "split": "dev_select", "edge_checks_complete": True, "common_alpha": .1, "runs": refs,
                "inputs_sha256": {preparation.BASE_CONFIG: sha256(base_path)}}
    write_json(selection, decision)
    return tmp_path, selection, tmp_path / "configs/generated/budget120/final", decision


def test_prepares_seven_configs_and_ordered_stages_without_test_access(project, monkeypatch):
    root, selection, destination, _ = project
    original_open = Path.open
    reads = []

    def observe(path, mode="r", *args, **kwargs):
        if "r" in mode:
            rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            assert not rel.startswith(("outputs/samples/test_", "data/processed/test.", "data/raw/test."))
            reads.append(rel)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    frozen = (root / preparation.BASE_CONFIG).read_bytes()
    first = preparation.prepare(root, selection, destination)
    assert preparation.prepare(root, selection, destination) == first
    assert first["run_count"] == 7
    assert (root / preparation.BASE_CONFIG).read_bytes() == frozen
    plan = json.loads((destination / "final_plan.json").read_text())
    assert all(sha256(root / name) == digest for name, digest in plan["inputs_sha256"].items())
    assert all(sha256(root / name) == digest for name, digest in plan["files_sha256"].items())
    assert plan["test_analysis_requires_all"] == ["C", "D", "E"]
    configs = plan["run_configs"]
    assert set(configs["dev_anal"]) == {"C", "D", "E"}
    assert set(configs["test"]) == {"greedy", "C", "D", "E"}
    for split, names in configs.items():
        for condition, name in names.items():
            run = yaml.safe_load((root / name).read_text())
            assert run["base_config"] == preparation.BASE_CONFIG and run["split"] == split
            assert run["n_rows"] == (1 if condition == "greedy" else 16)
            assert run["alpha"] == (None if condition == "greedy" else .1)
            assert run["temperature"] == (0 if condition == "greedy" else .5)
            assert run["id_version"] == 2 and run["seed_set"] == 0 and run["row_start"] == 0
    rehearsal = json.loads((destination / "stage_rehearsal.json").read_text())
    assert rehearsal["commands"] == [*[['-m', 'scripts.build_bank', configs['dev_anal'][c]] for c in 'CDE'],
                                       *[['-m', 'scripts.run_experiment', configs['dev_anal'][c]] for c in 'CDE']]
    test = json.loads((destination / "stage_test.json").read_text())
    expected = [["-m", "scripts.run_experiment", configs["test"]["greedy"]]]
    for c in "CDE":
        expected.extend([["-m", "scripts.build_bank", configs["test"][c]],
                         ["-m", "scripts.run_experiment", configs["test"][c]]])
    assert test["commands"] == expected and test["requires_test_lock"] is True
    for name, split in (("rehearsal", "dev_anal"), ("test", "test")):
        analysis = json.loads((destination / f"{name}_analysis_plan.json").read_text())
        assert (destination / analysis["root"]).resolve() == root
        assert analysis["split"] == split and analysis["base_config"] == preparation.BASE_CONFIG
        assert analysis["selection_file"] == selection.relative_to(root).as_posix()
        assert analysis["output_dir"] == f"outputs/analysis/budget120/{split}"
        assert analysis["run_configs"]["C"] == configs[split]["C"]
        assert analysis["run_configs"]["greedy"] == ("configs/runs/dev_anal_greedy.yaml"
                                                        if split == "dev_anal" else configs["test"]["greedy"])
    assert not (root / "data/processed/test.jsonl").exists()
    assert any(name.startswith("outputs/samples/dev_select_") for name in reads)


@pytest.mark.parametrize("failure", ["missing", "alpha", "raw", "edges"])
def test_incomplete_or_inconsistent_sources_write_nothing(project, failure):
    root, selection, destination, decision = project
    if failure == "missing":
        (root / decision["runs"]["D"]["manifest"]).unlink()
    elif failure == "alpha":
        decision["common_alpha"] = .2
    elif failure == "edges":
        decision["edge_checks_complete"] = False
    else:
        manifest = json.loads((root / decision["runs"]["D"]["manifest"]).read_text())
        (root / "outputs/samples" / f"{manifest['run_id']}.jsonl").write_text("corrupted")
    write_json(selection, decision)
    with pytest.raises((ValueError, FileNotFoundError)):
        preparation.prepare(root, selection, destination)
    assert not destination.exists()


def test_conflicting_snapshot_blocks_entire_batch(project):
    root, selection, destination, _ = project
    destination.mkdir(parents=True)
    conflict = destination / "final_plan.json"
    conflict.write_text("conflict")
    with pytest.raises(ValueError, match="Conflicting"):
        preparation.prepare(root, selection, destination)
    assert list(destination.iterdir()) == [conflict]


def test_existing_run_id_configuration_collision_is_rejected(project):
    root, selection, destination, _ = project
    run = {"split": "test", "condition": "greedy", "alpha": None, "temperature": 0,
           "basis": None, "seed_set": 0, "row_start": 0, "n_rows": 1, "id_version": 2}
    write_json(root / f"outputs/manifests/{run_id(run)}.json", {"run_config_sha256": "other"})
    with pytest.raises(ValueError, match="conflicting configuration"):
        preparation.prepare(root, selection, destination)
    assert not destination.exists()


def test_duplicate_run_ids_are_rejected(project, monkeypatch):
    root, selection, destination, _ = project
    monkeypatch.setattr(preparation, "run_id", lambda run: "collision")
    with pytest.raises(ValueError, match="IDs collide"):
        preparation.prepare(root, selection, destination)
    assert not destination.exists()
