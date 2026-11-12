"""Synthetic evidence only: never loads the workspace's sealed test data."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

import scripts.analyze_budget120 as analysis
from scripts.run_experiment import run_id, sha256
from src.config import config_sha256
from src.rng import rng_for
from src.stats import shared_subset_bootstrap


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    for key in ("k_generate_dev", "k_generate_test", "k_generate_rehearsal"):
        base["decode"][key] = 16
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    write(tmp_path / base["stats"]["analysis_list"], {"estimand": "mean subset AUROC"})
    for key in ("system_file", "fewshot_file"):
        path = tmp_path / base["prompt"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic prompt", encoding="utf-8")
    write(tmp_path / base["noise"]["rho_file"], {"rho_by_layer_1idx": {"20": 1}})
    write(tmp_path / "configs/budget120_protocol.json", {
        "protocol_amendment": analysis.REVISION,
        "analysis": {"confirmatory": ["D-C"], "E_minus_C_bootstrap_key": None, "practical_delta": .02}})
    source = tmp_path / "src/synthetic.py"
    source.parent.mkdir()
    source.write_text("# synthetic runtime provenance only", encoding="utf-8")
    environment = {"python": "3.12", "torch": "2.6", "transformers": "4.50", "cuda": "12.4",
                   "gpu_name": "synthetic", "gpu_uuid": "synthetic", "gpu_total_memory": 24000,
                   "deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
                   "cpu_threads": 4, "cpu_interop_threads": 4, "cublas_workspace_config": ":4096:8",
                   "model_dtype": "torch.float32", "model_revision_loaded": base["model"]["revision"],
                   "tokenizer_revision_loaded": base["model"]["revision"], "sampling": "cpu_float64_inverse_cdf",
                   "prefill_optimization_O1": False, "source_sha256": {"src/synthetic.py": sha256(source)}}
    ids_by_split = {"dev_select": [f"train-{i:05d}" for i in range(200)],
                    "dev_anal": [f"train-{i:05d}" for i in range(200, 250)]}
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir()
    for split, ids in ids_by_split.items():
        (tmp_path / f"data/splits/{split}.ids").write_text("\n".join(ids) + "\n", encoding="ascii")
    (tmp_path / "data/processed/train.jsonl").write_text("".join(
        json.dumps({"id": pid, "question": "synthetic only", "gold_number": "1"}) + "\n"
        for ids in ids_by_split.values() for pid in ids), encoding="utf-8")

    def make_run(split, condition):
        ids = ids_by_split[split]
        rows = 1 if condition == "greedy" else 16
        run = {"base_config": "configs/base.yaml", "split": split, "condition": condition,
               "alpha": .1 if condition in analysis.CONDITIONS else None,
               "temperature": 0 if condition == "greedy" else .5, "basis": "B0" if condition in analysis.CONDITIONS else None,
               "seed_set": 0, "row_start": 0, "n_rows": rows, "id_version": 2}
        name = run_id(run)
        run_path = tmp_path / f"configs/{name}.yaml"
        run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
        rng = np.random.default_rng(101 + ["greedy", "A", "C", "D", "E"].index(condition))
        unique = rng.integers(1, 9, size=(len(ids), 2))
        raw = []
        for i, pid in enumerate(ids):
            for k in range(rows):
                answer = ("1" if i % 2 == 0 else "INVALID" if i == 1 else "9") if condition == "greedy" else str(k % int(unique[i, k // 8]))
                raw.append({"id": pid, "run_id": name, "k": k, "subset": k // 8,
                            "prompt_sha256": f"synthetic-prompt-{pid}", "answer": answer, "finish_reason": "length" if k == 15 else "eos",
                            "cond": condition, "alpha": run["alpha"], "temperature": run["temperature"],
                            "basis": run["basis"] or "NA", "seed_set": 0, "n_rows": rows})
        output = tmp_path / f"outputs/samples/{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
        manifest = {"status": "complete", "run_id": name, "condition": condition, "split": split,
                    "n_rows": rows, "row_start": 0, "problem_count": len(ids), "sample_count": rows * len(ids),
                    "output_sha256": sha256(output), "run_config_sha256": sha256(run_path),
                    "config_sha256": config_sha256(base, {"estimand": "mean subset AUROC"}),
                    "model_id": base["model"]["id"], "model_revision": base["model"]["revision"],
                    "fewshot_sha256": sha256(tmp_path / base["prompt"]["fewshot_file"]),
                    "system_sha256": sha256(tmp_path / base["prompt"]["system_file"]),
                    "execution_invocations": [{"scope": "this_invocation_only", "existing_problem_files": 0,
                                               "environment": environment}]}
        if condition == "greedy":
            # Original labels with legacy tokenizer metadata are explicitly supported.
            manifest["execution_invocations"][0]["environment"] = {**environment, "tokenizer_revision_loaded": None}
        if condition in analysis.CONDITIONS:
            bank = tmp_path / f"bank/{split}_{condition}.npy"
            bank.parent.mkdir(exist_ok=True)
            bank.write_bytes(b"synthetic preserved bank")
            write(bank.with_suffix(".json"), {"condition": condition, "split": split, "basis": "B0", "seed_set": 0,
                "row_start": 0, "n_rows": 16, "dimension": base["noise"]["d"], "global_seed": 2026,
                "ids": ids, "dtype": "float32", "sparse_p": None, "file_sha256": sha256(bank)})
            manifest.update(bank_file=str(bank.relative_to(tmp_path)), bank_sha256=sha256(bank),
                            rho_sha256=sha256(tmp_path / base["noise"]["rho_file"]))
        manifest_path = tmp_path / f"outputs/manifests/{name}.json"
        write(manifest_path, manifest)
        return {"run_config": str(run_path.relative_to(tmp_path)), "run_config_sha256": sha256(run_path),
                "manifest": str(manifest_path.relative_to(tmp_path)), "manifest_sha256": sha256(manifest_path),
                "output_sha256": sha256(output)}

    selection = {"schema_version": 1, "status": "complete", "split": "dev_select", "protocol_amendment": analysis.REVISION,
                 "common_alpha": .1, "inputs_sha256": {"src/synthetic.py": sha256(source)},
                 "runs": {name: make_run("dev_select", name) for name in ("A", "C", "D", "E")}}
    write(tmp_path / "selection.json", selection)
    selected_greedy = make_run("dev_select", "greedy")
    original_greedy = tmp_path / selected_greedy["run_config"]
    legacy_greedy = tmp_path / "configs/runs/dev_select_greedy.yaml"
    legacy_greedy.parent.mkdir(parents=True)
    legacy_greedy.write_bytes(original_greedy.read_bytes())
    final = {name: make_run("dev_anal", name) for name in ("greedy", "C", "D", "E")}
    plan = {"root": ".", "split": "dev_anal", "base_config": "configs/base.yaml", "selection_file": "selection.json",
            "run_configs": {name: spec["run_config"] for name, spec in final.items()}, "output_dir": "outputs/analysis/rehearsal"}
    write(tmp_path / "plan.json", plan)
    return tmp_path, plan, final


@pytest.fixture
def fast_bootstrap(monkeypatch):
    calls = []
    def synthetic(labels, scores, comparisons, **kwargs):
        assert kwargs == {"repeats": 10000, "analysis_no": 1 if comparisons == [("D", "C")] else 2, "global_seed": 2026}
        calls.append(kwargs["analysis_no"])
        return shared_subset_bootstrap(labels, scores, comparisons, repeats=16,
                                       analysis_no=kwargs["analysis_no"], global_seed=2026)
    monkeypatch.setattr(analysis, "shared_subset_bootstrap", synthetic)
    return calls


def test_synthetic_flow_estimand_labels_outputs_and_resume(evidence, fast_bootstrap):
    root, _, _ = evidence
    report = analysis.analyze_budget120(root / "plan.json")
    assert fast_bootstrap == [1, 2]
    assert report["scope"] == "rehearsal" and report["confirmatory"] is False
    assert report["PR6"]["outcome"] is None
    assert report["comparisons"]["D-E"]["verdict"] is None
    assert report["comparisons"]["E-C"]["ci95"] is None
    assert report["comparisons"]["E-C"]["rng_key"] is None
    assert report["arithmetic_residual"] == pytest.approx(0)
    output = root / "outputs/analysis/rehearsal"
    with np.load(output / "bootstrap.npz", allow_pickle=False) as saved:
        y, x = saved["labels"], saved["condition_scores"]
        assert y[1] == 1  # INVALID greedy is a positive error label.
        assert y[0] == 0 and y[3] == 1
        for i, name in enumerate(analysis.CONDITIONS):
            auc = np.mean([roc_auc_score(y, x[i, :, j]) for j in range(2)])
            ap = np.mean([average_precision_score(y, x[i, :, j]) for j in range(2)])
            assert report["metrics"][name]["auroc"] == pytest.approx(auc)
            assert report["metrics"][name]["auprc"] == pytest.approx(ap)
            assert report["metrics"][name]["truncated"] == pytest.approx(1 / 16)
        assert not np.array_equal(saved["D-C_indices"], saved["D-E_indices"])
        assert np.array_equal(saved["D-C_indices"], rng_for(10, 1).integers(0, 50, size=(16, 50)))
        assert report["metrics"]["C"]["auroc"] != pytest.approx(roc_auc_score(y, x[0].mean(axis=1)))
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    analysis.analyze_budget120(root / "plan.json")
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    (output / "metrics.csv").write_text("preserve changed output", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        analysis.analyze_budget120(root / "plan.json")
    assert (output / "metrics.csv").read_text() == "preserve changed output"
    assert not (root / "data/processed/test.jsonl").exists()


@pytest.mark.parametrize("failure", ["incomplete", "missing_row", "hash", "runtime", "alpha", "unsafe_output"])
def test_rejects_before_any_statistics(evidence, monkeypatch, failure):
    root, plan, final = evidence
    manifest_path = root / final["E"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if failure == "incomplete":
        manifest["status"] = "running"
    elif failure == "missing_row":
        output = root / f"outputs/samples/{manifest['run_id']}.jsonl"
        rows = output.read_text().splitlines()
        output.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        manifest["output_sha256"] = sha256(output)
    elif failure == "hash":
        manifest["output_sha256"] = "0" * 64
    elif failure == "runtime":
        manifest["execution_invocations"][0]["environment"]["gpu_uuid"] = "changed"
    elif failure == "alpha":
        path = root / final["E"]["run_config"]
        run = yaml.safe_load(path.read_text())
        run["alpha"] = .2
        path.write_text(yaml.safe_dump(run), encoding="utf-8")
    else:
        plan["output_dir"] = "../outside"
    write(manifest_path, manifest)
    write(root / "plan.json", plan)
    monkeypatch.setattr(analysis, "compute_statistics", lambda *_: pytest.fail("Statistics called before gate completed"))
    with pytest.raises(ValueError):
        analysis.analyze_budget120(root / "plan.json")
    assert not (root / "outputs/analysis").exists()


def test_test_seal_checked_before_selection_or_test_access(evidence, monkeypatch):
    root, plan, _ = evidence
    plan["split"] = "test"
    write(root / "plan.json", plan)
    def sealed(*_):
        raise PermissionError("synthetic test seal")
    monkeypatch.setattr(analysis, "assert_test_locked", sealed)
    monkeypatch.setattr(analysis, "_verify_selection", lambda *_: pytest.fail("selection reached before test seal"))
    monkeypatch.setattr(analysis, "load_split", lambda *_: pytest.fail("test read before test seal"))
    monkeypatch.setattr(analysis, "compute_statistics", lambda *_: pytest.fail("statistics before test seal"))
    with pytest.raises(PermissionError, match="synthetic test seal"):
        analysis.analyze_budget120(root / "plan.json")


def test_incomplete_test_gate_never_enters_statistics(evidence, monkeypatch):
    root, plan, final = evidence
    plan["split"] = "test"
    write(root / "plan.json", plan)
    monkeypatch.setattr(analysis, "assert_test_locked", lambda *_: None)
    original_load = analysis.load_split
    monkeypatch.setattr(analysis, "load_split", lambda r, *_: original_load(r, "dev_anal", yaml.safe_load((r / "configs/base.yaml").read_text())))
    # A synthetic count guard rejects incomplete test evidence without opening a test file.
    monkeypatch.setattr(analysis, "compute_statistics", lambda *_: pytest.fail("statistics called for incomplete test"))
    with pytest.raises(ValueError, match="split count"):
        analysis.analyze_budget120(root / "plan.json")


def test_all_test_conditions_must_complete_before_statistics_root_mock(evidence, monkeypatch):
    root, plan, _ = evidence
    plan["split"] = "test"
    write(root / "plan.json", plan)
    data = [{"id": f"synthetic-{i}", "gold_number": "1"} for i in range(1319)]
    monkeypatch.setattr(analysis, "assert_test_locked", lambda *_: None)
    monkeypatch.setattr(analysis, "_verify_selection", lambda *_: ({"common_alpha": .1}, {"synthetic": True}))
    monkeypatch.setattr(analysis, "load_split", lambda *_: data)
    called = []
    def verified(_root, _path, condition, *_args):
        called.append(condition)
        if condition == "E":
            raise ValueError("synthetic final E has a missing row")
        return {"runtime": {"synthetic": True}}, {row["id"]: [{"prompt_sha256": "same"}] for row in data}
    monkeypatch.setattr(analysis, "_load_verified", verified)
    monkeypatch.setattr(analysis, "compute_statistics", lambda *_: pytest.fail("incomplete test calculated statistics"))
    monkeypatch.setattr(analysis, "render_outputs", lambda *_: pytest.fail("incomplete test rendered output"))
    with pytest.raises(ValueError, match="missing row"):
        analysis.analyze_budget120(root / "plan.json")
    assert called == ["greedy", "C", "D", "E"]
    assert not (root / "outputs/analysis").exists()


def test_single_class_returns_null_instead_of_manufactured_statistics(evidence, monkeypatch):
    root, _, _ = evidence
    verified = analysis.verify_inputs(root / "plan.json")
    for items in verified["runs"]["greedy"].values():
        items[0]["answer"] = "1"
    monkeypatch.setattr(analysis, "shared_subset_bootstrap", lambda *_args, **_kwargs: pytest.fail("one-class bootstrap"))
    report, _, _ = analysis.compute_statistics(verified)
    for metric in report["metrics"].values():
        assert metric["auroc"] is None and metric["auprc"] is None
    for comparison in report["comparisons"].values():
        assert comparison["delta"] is None and comparison["ci95"] is None and comparison["se"] is None
    assert report["analysis_unavailable_reason"]


def test_invalid_singletons_majority_valid_only_earliest_tie_and_all16_rates(evidence):
    root, _, _ = evidence
    verified = analysis.verify_inputs(root / "plan.json")
    for items in verified["runs"]["greedy"].values():
        items[0]["answer"] = "1"  # Single class prevents unnecessary bootstrap.
    for items in verified["runs"]["C"].values():
        answers = ["1", "1"] + ["INVALID"] * 6 + ["2", "1"] + ["INVALID"] * 6
        for item, answer in zip(items, answers):
            item["answer"] = answer
    report, _, arrays = analysis.compute_statistics(verified)
    metric = report["metrics"]["C"]
    assert metric["maj_acc"] == .5  # Valid majority first group, earliest valid tie second group.
    assert metric["invalid"] == 12 / 16
    assert metric["truncated"] == 1 / 16
    assert arrays["condition_scores"][0, 0, 0] == pytest.approx(-.25 * np.log(.25) - 6 * .125 * np.log(.125))
    assert arrays["condition_scores"][0, 0, 1] == pytest.approx(np.log(8))


def test_bootstrap_skips_single_class_attempts_without_redraw():
    y = np.array([0, 1])
    scores = {"D": np.array([[0., 1.], [1., 0.]]), "C": np.array([[1., 1.], [0., 0.]])}
    result = shared_subset_bootstrap(y, scores, [("D", "C")], repeats=30, analysis_no=1)
    expected = rng_for(10, 1).integers(0, 2, size=(30, 2))
    assert np.array_equal(result["indices"], expected)
    assert np.array_equal(result["valid_mask"], y[expected].min(axis=1) != y[expected].max(axis=1))
    assert result["valid"] + result["skipped"] == 30
    assert np.isnan(result["subset_deltas"][~result["valid_mask"]]).all()


def test_npz_fixed_zip_metadata_is_deterministic():
    arrays = {"x": np.array([1, 2]), "nan": np.array([np.nan])}
    assert analysis._npz(arrays) == analysis._npz(arrays)


def _pipeline_plan(evidence, *, policy=True):
    root, plan, _ = evidence
    selection = json.loads((root / "selection.json").read_text())
    plan = {**plan, "split": "dev_select", "output_dir": "outputs/analysis/budget120/dev_select",
            "run_configs": {"greedy": "configs/runs/dev_select_greedy.yaml",
                            **{name: selection["runs"][name]["run_config"] for name in analysis.CONDITIONS}}}
    if policy:
        write(root / "configs/development_check_policy.json", {
            "generate_dev_anal_responses": False, "pipeline_check_split": "dev_select"})
    write(root / "pipeline_plan.json", plan)
    return root, plan


def test_reused_dev_select_full_cpu_pipeline_is_explicitly_nonindependent(evidence, fast_bootstrap):
    root, _ = _pipeline_plan(evidence)
    verified = analysis.verify_inputs(root / "pipeline_plan.json")
    assert verified["inputs"]["runs"]["greedy"]["runtime"] is None
    assert "legacy label reference" in verified["inputs"]["runs"]["greedy"]["runtime_comparison"]
    raw_inputs = {path: path.read_bytes() for path in (root / "outputs/samples").glob("*.jsonl")}
    report = analysis.analyze_budget120(root / "pipeline_plan.json")
    assert fast_bootstrap == [1, 2]
    assert report["n"] == 200 and report["scope"] == "development_pipeline_check"
    assert report["selection_data_reused"] is True
    assert report["independent_evaluation"] is False and report["confirmatory"] is False
    assert report["PR6"]["outcome"] is None
    assert all(row["confirmatory"] is False for row in report["comparisons"].values())
    assert report["inputs"]["development_check_policy_sha256"] == sha256(root / "configs/development_check_policy.json")
    output = root / "outputs/analysis/budget120/dev_select"
    assert {path.name for path in output.iterdir()} == {
        "analysis.json", "bootstrap.npz", "metrics.csv", "comparisons.csv", "k_curve.csv",
        "problem_scores.csv", "figure2.png", "실험결과.md"}
    with np.load(output / "bootstrap.npz", allow_pickle=False) as saved:
        assert saved["condition_scores"].shape == (3, 200, 2)
        assert saved["D-C_indices"].shape == (16, 200)
    assert all(path.read_bytes() == raw for path, raw in raw_inputs.items())


@pytest.mark.parametrize("failure", ["no_policy", "wrong_policy", "selection_incomplete", "different_C", "different_greedy"])
def test_pipeline_check_requires_policy_complete_selection_and_exact_reused_runs(evidence, monkeypatch, failure):
    root, plan = _pipeline_plan(evidence, policy=failure != "no_policy")
    if failure == "wrong_policy":
        write(root / "configs/development_check_policy.json", {
            "generate_dev_anal_responses": True, "pipeline_check_split": "dev_select"})
    elif failure == "selection_incomplete":
        path = root / "selection.json"
        selection = json.loads(path.read_text())
        selection["status"] = "running"
        write(path, selection)
    elif failure in ("different_C", "different_greedy"):
        name = "C" if failure == "different_C" else "greedy"
        original = root / plan["run_configs"][name]
        duplicate = root / "configs/duplicate.yaml"
        duplicate.write_bytes(original.read_bytes())
        plan["run_configs"][name] = "configs/duplicate.yaml"
        write(root / "pipeline_plan.json", plan)
    monkeypatch.setattr(analysis, "compute_statistics", lambda *_: pytest.fail("statistics before valid selected-data reuse"))
    with pytest.raises(ValueError):
        analysis.analyze_budget120(root / "pipeline_plan.json")
    assert not (root / "outputs/analysis").exists()


def test_present_policy_is_hashed_for_other_modes_without_changing_test_guard(evidence, monkeypatch):
    root, plan, _ = evidence
    policy = root / "configs/development_check_policy.json"
    write(policy, {"generate_dev_anal_responses": False, "pipeline_check_split": "dev_select"})
    assert analysis.verify_inputs(root / "plan.json")["inputs"]["development_check_policy_sha256"] == sha256(policy)
    plan["split"] = "test"
    write(root / "plan.json", plan)
    monkeypatch.setattr(analysis, "assert_test_locked", lambda *_: (_ for _ in ()).throw(PermissionError("still sealed")))
    with pytest.raises(PermissionError, match="still sealed"):
        analysis.verify_inputs(root / "plan.json")
