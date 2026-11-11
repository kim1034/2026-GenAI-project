import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.make_pilot_grid import make_pilot_grids
from scripts.run_experiment import run_id, sha256
from src.config import config_sha256
from src.extract import INVALID


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    # Separate immutable configuration/prompt snapshots from run outputs.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["data"]["splits"]["dev_base"] = 2
    analysis = {"score": "fixed", "selection": "fixed"}
    (snapshot / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (snapshot / "configs").mkdir()
    (snapshot / "configs/analysis.yaml").write_text(yaml.safe_dump(analysis), encoding="utf-8")
    (snapshot / "data/splits").mkdir(parents=True)
    ids = ["train-00001", "train-00002"]
    (snapshot / "data/splits/dev_base.ids").write_text("\n".join(ids) + "\n", encoding="ascii")
    for relative in (base["prompt"]["fewshot_file"], base["prompt"]["system_file"], base["noise"]["rho_file"]):
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable source", encoding="utf-8")

    def run(condition, alpha, *, changed=None, answers=None, selected_base=None):
        selected_base = selected_base or base
        config = {"base_config": "base.yaml", "split": "dev_base", "condition": condition,
                  "alpha": alpha, "temperature": 0.5, "basis": None if condition == "A" else "B0",
                  "seed_set": 0, "row_start": 0, "n_rows": 16, "id_version": 2}
        config.update(changed or {})
        name = run_id(config)
        yaml_path = snapshot / f"{name}.yaml"
        yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        samples = [{"id": pid, "run_id": name, "k": k, "prompt_sha256": f"prompt-{pid}",
                    "cond": condition, "alpha": alpha, "temperature": config["temperature"],
                    "basis": config["basis"] or "NA", "seed_set": config["seed_set"],
                    "n_rows": 16, "subset": k // 8,
                    "answer": (answers or ["1"] * 16)[k]} for pid in ids for k in range(16)]
        output = tmp_path / "outputs/samples" / f"{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        manifest_path = tmp_path / "outputs/manifests" / f"{name}.json"
        manifest = {"status": "complete", "run_id": name, "condition": condition, "split": config["split"],
                    "n_rows": 16, "row_start": 0, "problem_count": 2, "sample_count": 32,
                    "output_sha256": sha256(output), "run_config_sha256": sha256(yaml_path),
                    "config_sha256": config_sha256(selected_base, analysis), "model_id": base["model"]["id"],
                    "model_revision": base["model"]["revision"],
                    "fewshot_sha256": sha256(snapshot / base["prompt"]["fewshot_file"]),
                    "system_sha256": sha256(snapshot / base["prompt"]["system_file"])}
        manifest.update(run_parameters=config, base_config_snapshot=selected_base, alpha=alpha,
                        temperature=config["temperature"], id_version=config["id_version"],
                        rho_sha256=None if condition == "A" else sha256(snapshot / base["noise"]["rho_file"]))
        if condition != "A":
            bank_path = tmp_path / f"bank_{condition}.npy"
            bank_path.write_bytes(b"synthetic preserved bank bytes")
            bank = {"condition": condition, "split": "dev_base", "basis": "B0", "seed_set": 0,
                    "row_start": 0, "n_rows": 16, "sparse_p": selected_base["noise"]["conditions"]["S"]["p"]
                    if condition == "S" else None,
                    "dimension": base["noise"]["d"], "global_seed": base["global_seed"],
                    "ids": ids, "dtype": "float32", "file_sha256": sha256(bank_path)}
            write_json(bank_path.with_suffix(".json"), bank)
            manifest.update(bank_file=bank_path.name, bank_sha256=sha256(bank_path))
        environment = {"python": "3.12", "torch": "2.6", "transformers": "4.50", "cuda": "12.4",
                       "gpu_name": "RTX A5000", "gpu_total_memory": 24000, "hostname": "host",
                       "deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
                       "cpu_threads": 4, "cpu_interop_threads": 4, "cublas_workspace_config": ":4096:8",
                       "model_dtype": "torch.float32", "model_revision_loaded": base["model"]["revision"],
                       "tokenizer_revision_loaded": base["model"]["revision"],
                       "sampling": "cpu_float64_inverse_cdf", "prefill_optimization_O1": False,
                       "source_sha256": {"src/decode.py": "fixed-source"}}
        manifest["execution_invocations"] = [{"scope": "this_invocation_only", "existing_problem_files": 0,
                                              "environment": environment}]
        write_json(manifest_path, manifest)
        return yaml_path.name, manifest_path, output

    a_path, _, _ = run("A", None)
    c_runs = [run("C", alpha, answers=["1", "2"] * 8) for alpha in base["noise"]["pilot_grid_C"]]
    plan = {"root": ".", "config_root": "snapshot", "source_root": "snapshot", "a_run_config": a_path,
            "conditions": {"C": {"expected_alphas": base["noise"]["pilot_grid_C"],
                                  "run_configs": [item[0] for item in c_runs], "output": "grid_C.json"}}}
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    return tmp_path, snapshot, base, plan, plan_path, c_runs, run


def test_complete_grid_and_identical_resume_conflicting_output(evidence):
    root, _, _, _, plan_path, _, _ = evidence
    result = next(iter(make_pilot_grids(plan_path).values()))
    assert result["grid"]["alpha_lo"] == 0.0125
    assert result["grid"]["alpha_hi"] == 1.6
    assert result["grid"]["launch_allowed"]
    assert len(result["grid"]["grid"]) == 6
    assert result["inputs"]["pilot_runs"][0]["mean_unique8_first_subset"] == 2
    assert "auroc" not in result
    assert all("auroc" not in row for row in result["inputs"]["pilot_runs"])
    raw = (root / "grid_C.json").read_bytes()
    make_pilot_grids(plan_path)
    assert (root / "grid_C.json").read_bytes() == raw
    changed = json.loads(raw)
    changed["grid"]["alpha_hi"] = 999
    write_json(root / "grid_C.json", changed)
    with pytest.raises(ValueError, match="conflicts"):
        make_pilot_grids(plan_path)
    assert json.loads((root / "grid_C.json").read_text())["grid"]["alpha_hi"] == 999


@pytest.mark.parametrize("failure", ["incomplete", "missing_k", "yaml_hash", "output_hash", "prompt"])
def test_incomplete_or_changed_evidence_never_writes_grid(evidence, failure):
    root, snapshot, _, _, plan_path, runs, _ = evidence
    yaml_name, manifest_path, output = runs[-1]
    manifest = json.loads(manifest_path.read_text())
    if failure == "incomplete":
        manifest["status"] = "running"
    elif failure == "yaml_hash":
        with (snapshot / yaml_name).open("a", encoding="utf-8") as handle:
            handle.write("\n# changed bytes\n")
    elif failure == "output_hash":
        with output.open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if failure == "missing_k":
            rows[15]["k"] = 14
        else:
            for row in rows:
                row["prompt_sha256"] = "different"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest["output_sha256"] = sha256(output)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        make_pilot_grids(plan_path)
    assert not (root / "grid_C.json").exists()


@pytest.mark.parametrize("field,value", [("split", "test"), ("condition", "D"), ("temperature", 0.6),
                                         ("seed_set", 1), ("n_rows", 32), ("row_start", 32), ("basis", "B1")])
def test_wrong_condition_is_rejected_before_reading_outputs(evidence, field, value):
    root, snapshot, _, _, plan_path, runs, _ = evidence
    path = snapshot / runs[0][0]
    config = yaml.safe_load(path.read_text())
    config[field] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Pilot"):
        make_pilot_grids(plan_path)
    assert not (root / "grid_C.json").exists()


def test_partial_grid_and_unavailable_s_do_not_generate_any_decision(evidence):
    root, _, base, plan, plan_path, _, _ = evidence
    plan["conditions"]["S"] = {"expected_alphas": base["noise"]["pilot_grid_S"], "run_configs": [],
                               "output": "grid_S.json"}
    write_json(root / "sparse.json", {"status": "complete", "p": 0.005})
    plan["sparse_decision"] = {"path": "sparse.json", "sha256": sha256(root / "sparse.json")}
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match="requires 6"):
        make_pilot_grids(plan_path)
    assert not (root / "grid_C.json").exists()
    assert not (root / "grid_S.json").exists()


def test_signature_blocks_decode_drift_but_accepts_budget_metadata(evidence):
    root, snapshot, base, plan, plan_path, runs, run = evidence
    altered = copy.deepcopy(base)
    altered["budget"]["hours_planned"] = 1
    (snapshot / "metadata.yaml").write_text(yaml.safe_dump(altered), encoding="utf-8")
    replacement, _, _ = run("C", base["noise"]["pilot_grid_C"][0],
                             changed={"base_config": "metadata.yaml"}, selected_base=altered)
    plan["conditions"]["C"]["run_configs"][0] = replacement
    write_json(plan_path, plan)
    make_pilot_grids(plan_path)
    (root / "grid_C.json").unlink()
    altered["decode"]["top_p"] = 0.9
    (snapshot / "metadata.yaml").write_text(yaml.safe_dump(altered), encoding="utf-8")
    run("C", base["noise"]["pilot_grid_C"][0], changed={"base_config": "metadata.yaml"}, selected_base=altered)
    with pytest.raises(ValueError, match=r"base.decode.top_p"):
        make_pilot_grids(plan_path)


def test_invalid_singletons_and_no_hi_preserve_launch_block(evidence):
    root, _, base, plan, plan_path, _, run = evidence
    plan["conditions"]["C"]["run_configs"] = [run("C", alpha, answers=[INVALID] * 16)[0]
                                               for alpha in base["noise"]["pilot_grid_C"]]
    write_json(plan_path, plan)
    result = next(iter(make_pilot_grids(plan_path).values()))
    assert result["inputs"]["pilot_runs"][0]["mean_unique8_first_subset"] == 8
    assert result["inputs"]["pilot_runs"][0]["invalid"] == 1
    assert result["grid"]["status"] == "blocked_no_hi"
    assert not result["grid"]["launch_allowed"]


@pytest.mark.parametrize("failure", [None, "bank_p", "decision_hash", "unset_p"])
def test_s_requires_complete_diagnostic_fixed_p_and_matching_bank(evidence, failure):
    root, snapshot, base, plan, plan_path, _, run = evidence
    sparse_base = copy.deepcopy(base)
    sparse_base["noise"]["conditions"]["S"]["p"] = None if failure == "unset_p" else 0.005
    (snapshot / "sparse_base.yaml").write_text(yaml.safe_dump(sparse_base), encoding="utf-8")
    write_json(root / "sparse.json", {"status": "complete", "p": 0.005})
    plan["sparse_decision"] = {"path": "sparse.json", "sha256": sha256(root / "sparse.json")}
    if failure == "decision_hash":
        plan["sparse_decision"]["sha256"] = "wrong"
    bank_path = root / "sparse_bank.npy"
    bank_path.write_bytes(b"synthetic bank bytes, no generation")
    bank = {"condition": "S", "split": "dev_base", "basis": "B0", "seed_set": 0,
            "row_start": 0, "n_rows": 16, "sparse_p": 0.002 if failure == "bank_p" else 0.005,
            "dimension": base["noise"]["d"], "global_seed": base["global_seed"],
            "ids": ["train-00001", "train-00002"], "dtype": "float32", "file_sha256": sha256(bank_path)}
    write_json(bank_path.with_suffix(".json"), bank)
    s_runs = []
    for alpha in base["noise"]["pilot_grid_S"]:
        yaml_name, manifest_path, _ = run("S", alpha, changed={"base_config": "sparse_base.yaml"},
                                         selected_base=sparse_base, answers=["1", "2"] * 8)
        manifest = json.loads(manifest_path.read_text())
        manifest.update(bank_file="sparse_bank.npy", bank_sha256=sha256(bank_path))
        write_json(manifest_path, manifest)
        s_runs.append(yaml_name)
    plan["conditions"]["S"] = {"expected_alphas": base["noise"]["pilot_grid_S"], "run_configs": s_runs,
                               "output": "grid_S.json"}
    write_json(plan_path, plan)
    if failure:
        with pytest.raises(ValueError, match="[Ss]parse|S p|S bank"):
            make_pilot_grids(plan_path)
        assert not (root / "grid_C.json").exists()
        assert not (root / "grid_S.json").exists()
    else:
        results = make_pilot_grids(plan_path)
        assert len(results) == 2
        s = results[str(root / "grid_S.json")]
        assert s["grid"]["launch_allowed"]
        assert all(row["sparse_decision_sha256"] == plan["sparse_decision"]["sha256"]
                   for row in s["inputs"]["pilot_runs"])


@pytest.mark.parametrize("failure", ["legacy", "prefill_optimization_O1", "source_sha256", "gpu_name"])
def test_runtime_evidence_is_required_and_differences_are_not_assumed_equivalent(evidence, failure):
    root, _, _, _, plan_path, runs, _ = evidence
    manifest_path = runs[-1][1]
    manifest = json.loads(manifest_path.read_text())
    if failure == "legacy":
        del manifest["execution_invocations"]
    else:
        value = {"prefill_optimization_O1": True, "source_sha256": {"src/decode.py": "changed-source"},
                 "gpu_name": "other GPU"}[failure]
        manifest["execution_invocations"][0]["environment"][failure] = value
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Runtime|runtime|signature|Insufficient"):
        make_pilot_grids(plan_path)
    assert not (root / "grid_C.json").exists()


@pytest.mark.parametrize("failure", ["raw_alpha", "snapshot", "bank", "rho"])
def test_self_consistent_output_hash_cannot_hide_parameter_or_provenance_drift(evidence, failure):
    root, _, _, _, plan_path, runs, _ = evidence
    _, manifest_path, output = runs[-1]
    manifest = json.loads(manifest_path.read_text())
    if failure == "raw_alpha":
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        rows[0]["alpha"] = 999
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest["output_sha256"] = sha256(output)
    elif failure == "snapshot":
        manifest["run_parameters"]["alpha"] = 999
    elif failure == "bank":
        (root / manifest["bank_file"]).write_bytes(b"wrong bank")
    else:
        del manifest["rho_sha256"]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="parameters|snapshots|bank|rho"):
        make_pilot_grids(plan_path)
    assert not (root / "grid_C.json").exists()
