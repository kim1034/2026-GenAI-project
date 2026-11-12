import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import roc_auc_score

import scripts.project_power as power
from scripts.run_experiment import run_id, sha256
from src.config import config_sha256
from src.rng import rng_for


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["noise"]["conditions"]["S"]["p"] = .005
    (source / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    analysis = {"score": "mean_of_subset_entropy_auroc", "selection": "fixed"}
    (source / "configs").mkdir()
    (source / "configs/analysis.yaml").write_text(yaml.safe_dump(analysis), encoding="utf-8")
    for key in ("system_file", "fewshot_file"):
        path = source / base["prompt"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed prompt source", encoding="utf-8")
    rho = source / base["noise"]["rho_file"]
    write_json(rho, {"rho_by_layer_1idx": {"20": 1.0}})
    ids = [f"train-{i:05d}" for i in range(200)]
    (source / "data/splits").mkdir(parents=True)
    (source / "data/splits/dev_select.ids").write_text("\n".join(ids) + "\n", encoding="ascii")
    (source / "data/processed").mkdir()
    (source / "data/processed/train.jsonl").write_text("".join(
        json.dumps({"id": pid, "question": "synthetic only", "gold_number": "1"}) + "\n" for pid in ids),
        encoding="utf-8")
    # Deliberately provide no test IDs, test answers, or test outputs.
    environment = {"python": "3.12", "torch": "2.6", "transformers": "4.50", "cuda": "12.4",
                   "gpu_name": "RTX A5000", "gpu_uuid": "synthetic-only", "gpu_total_memory": 24000,
                   "deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
                   "cpu_threads": 4, "cpu_interop_threads": 4, "cublas_workspace_config": ":4096:8",
                   "model_dtype": "torch.float32", "model_revision_loaded": base["model"]["revision"],
                   "tokenizer_revision_loaded": base["model"]["revision"],
                   "sampling": "cpu_float64_inverse_cdf", "prefill_optimization_O1": False,
                   "source_sha256": {"src/decode.py": "0" * 64}}

    def make_run(condition, alpha, temperature=.5):
        count = 1 if condition == "greedy" else 16
        run = {"base_config": "base.yaml", "split": "dev_select", "condition": condition, "alpha": alpha,
               "temperature": 0.0 if condition == "greedy" else temperature,
               "basis": "B0" if condition in ("C", "D", "E", "S") else None,
               "seed_set": 0, "row_start": 0, "n_rows": count, "id_version": 2}
        name = run_id(run)
        yaml_path = source / f"{name}.yaml"
        yaml_path.write_text(yaml.safe_dump(run), encoding="utf-8")
        rng = np.random.default_rng(100 + ["greedy", "C", "D", "E", "S", "A", "Ap"].index(condition))
        unique_counts = rng.integers(1, 9, size=(200, 2))
        samples = []
        for i, pid in enumerate(ids):
            for k in range(count):
                answer = ("1" if i % 2 == 0 else "INVALID" if i == 1 else "9") if condition == "greedy" else str(k % int(unique_counts[i, k // 8]))
                samples.append({"id": pid, "k": k, "subset": k // 8, "run_id": name,
                                "prompt_sha256": f"prompt-{pid}", "answer": answer,
                                "cond": condition, "alpha": alpha, "temperature": run["temperature"],
                                "basis": run["basis"] or "NA", "seed_set": 0, "n_rows": count})
        output = tmp_path / "outputs/samples" / f"{name}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        manifest = {"status": "complete", "run_id": name, "condition": condition, "split": "dev_select",
                    "n_rows": count, "row_start": 0, "problem_count": 200, "sample_count": 200 * count,
                    "output_sha256": sha256(output), "run_config_sha256": sha256(yaml_path),
                    "config_sha256": config_sha256(base, analysis), "model_id": base["model"]["id"],
                    "model_revision": base["model"]["revision"], "run_parameters": run,
                    "base_config_snapshot": base, "alpha": alpha, "temperature": run["temperature"], "id_version": 2,
                    "fewshot_sha256": sha256(source / base["prompt"]["fewshot_file"]),
                    "system_sha256": sha256(source / base["prompt"]["system_file"])}
        if condition != "greedy":
            manifest["execution_invocations"] = [{"scope": "this_invocation_only", "existing_problem_files": 0,
                                                  "environment": environment}]
        if condition in ("C", "D", "E", "S"):
            bank_path = tmp_path / f"bank_{condition}.npy"
            bank_path.write_bytes(b"preserved synthetic bank content")
            write_json(bank_path.with_suffix(".json"), {"condition": condition, "split": "dev_select", "basis": "B0",
                "seed_set": 0, "row_start": 0, "n_rows": 16, "dimension": base["noise"]["d"], "global_seed": 2026,
                "ids": ids, "dtype": "float32", "sparse_p": .005 if condition == "S" else None,
                "file_sha256": sha256(bank_path)})
            manifest.update(bank_file=bank_path.name, bank_sha256=sha256(bank_path), rho_sha256=sha256(rho))
        manifest_path = tmp_path / "outputs/manifests" / f"{name}.json"
        write_json(manifest_path, manifest)
        return {"run_config": yaml_path.name, "run_config_sha256": sha256(yaml_path),
                "manifest": manifest_path.relative_to(tmp_path).as_posix(), "manifest_sha256": sha256(manifest_path),
                "edge_check_complete": True,
                "evidence": [{"path": "selection_evidence.json", "sha256": sha256(tmp_path / "selection_evidence.json")} ]}

    write_json(tmp_path / "selection_evidence.json", {"schema_version": 1, "status": "complete", "split": "dev_select",
               "scope": "development_selection_evidence", "description": "synthetic completed selection evidence"})
    greedy = make_run("greedy", None)
    runs = {name: make_run(name, .1 if name in ("C", "D", "E") else .05 if name == "S" else None)
            for name in ("C", "D", "E", "S", "A")}
    selection = {"schema_version": 1, "status": "complete", "scope": "development_selection", "split": "dev_select",
                 "edge_checks_complete": True, "common_alpha": .1, "alpha_S": .05, "T_best": .5,
                 "Ap_is_A": True, "runs": runs}
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    config = {"root": ".", "config_root": "snapshot", "source_root": "snapshot", "selection_file": "selection.json",
              "baseline_greedy_manifest": greedy["manifest"], "baseline_greedy_run_config": greedy["run_config"],
              "manifests": {name: item["manifest"] for name, item in runs.items()}, "output_prefix": "outputs/power/projection"}
    config_path = tmp_path / "power.json"
    write_json(config_path, config)
    return tmp_path, config_path, config, selection, make_run


def test_complete_synthetic_flow_has_shared_indices_and_no_test_access(evidence, monkeypatch):
    root, config_path, _, _, _ = evidence
    actual_bootstrap = power.shared_subset_bootstrap
    captured = {}
    def observe(labels, scores, comparisons, **kwargs):
        assert kwargs == {"repeats": 2000, "analysis_no": 50, "global_seed": 2026}
        captured["bootstrap"] = actual_bootstrap(labels, scores, comparisons, **kwargs)
        return captured["bootstrap"]
    monkeypatch.setattr(power, "shared_subset_bootstrap", observe)
    report = power.project_power(config_path)
    assert report["budget_status"] == "pending_runtime"
    assert report["boost_final"] is None and report["test_rows_final"] is None
    assert report["inputs"]["selected_runs"]["Ap"]["alias_of"] == "A"
    npz_path = root / "outputs/power/projection.npz"
    assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == report["npz_sha256"]
    with np.load(npz_path, allow_pickle=False) as saved:
        assert saved["condition_scores"].shape == (6, 200, 2)
        assert saved["bootstrap_subset_deltas"].shape == (2000, 4, 2)
        assert np.array_equal(saved["bootstrap_indices"], rng_for(10, 50).integers(0, 200, size=(2000, 200)))
        assert np.array_equal(saved["condition_scores"][4], saved["condition_scores"][5])
        assert saved["labels"][1] == 1  # INVALID greedy answers are positive labels.
        index = saved["bootstrap_indices"][0]
        values = saved["condition_scores"]
        y = saved["labels"][index]
        for c, (left, right) in enumerate(power.COMPARISONS):
            a, b = power.CONDITIONS.index(left), power.CONDITIONS.index(right)
            expected = [roc_auc_score(y, values[a, index, j]) - roc_auc_score(y, values[b, index, j]) for j in range(2)]
            assert saved["bootstrap_subset_deltas"][0, c] == pytest.approx(expected)
        assert saved["point_subset_deltas"] == pytest.approx(np.array(
            [report["comparisons"]["-".join(pair)]["point_subset_deltas"] for pair in power.COMPARISONS]))
    assert (root / "reports/power.md").exists()
    assert not (root / "snapshot/data/processed/test.jsonl").exists()
    # Resume checks every generated byte, using the already observed deterministic core result.
    monkeypatch.setattr(power, "shared_subset_bootstrap", lambda *args, **kwargs: captured["bootstrap"])
    raw = npz_path.read_bytes()
    power.project_power(config_path)
    assert npz_path.read_bytes() == raw
    report_path = root / "reports/power.md"
    report_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        power.project_power(config_path)
    assert report_path.read_text() == "changed"


@pytest.mark.parametrize("failure", ["selection_incomplete", "edges", "hash", "output", "wrong_alpha", "rows", "one_class", "budget"])
def test_invalid_inputs_produce_no_power_artifacts(evidence, failure):
    root, config_path, config, selection, _ = evidence
    if failure == "selection_incomplete":
        selection["status"] = "running"
    elif failure == "edges":
        selection["edge_checks_complete"] = False
    elif failure == "hash":
        selection["runs"]["D"]["manifest_sha256"] = "wrong"
    elif failure == "budget":
        write_json(root / "budget.json", {"status": "complete", "boost_final": True})
        config["budget_decision_file"] = "budget.json"
        write_json(config_path, config)
    else:
        manifest_path = root / (config["baseline_greedy_manifest"] if failure == "one_class" else selection["runs"]["D"]["manifest"])
        manifest = json.loads(manifest_path.read_text())
        output = root / "outputs/samples" / f'{manifest["run_id"]}.jsonl'
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if failure == "output":
            rows.pop()
        elif failure == "wrong_alpha":
            rows[0]["alpha"] = .8
        elif failure == "rows":
            rows[15]["k"] = 14
        else:
            for row in rows:
                row["answer"] = "1"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest["output_sha256"] = sha256(output)
        write_json(manifest_path, manifest)
        if failure != "one_class":
            selection["runs"]["D"]["manifest_sha256"] = sha256(manifest_path)
    write_json(root / "selection.json", selection)
    with pytest.raises(ValueError):
        power.project_power(config_path)
    assert not (root / "outputs/power/projection.npz").exists()
    assert not (root / "outputs/power/projection.json").exists()
    assert not (root / "reports/power.md").exists()


@pytest.mark.parametrize("temperature,alias", [(.5, False), (.8, True)])
def test_temperature_alias_must_match_final_selection(evidence, temperature, alias):
    root, config_path, _, selection, _ = evidence
    selection.update(T_best=temperature, Ap_is_A=alias)
    write_json(root / "selection.json", selection)
    with pytest.raises(ValueError, match="Ap_is_A"):
        power.project_power(config_path)


def test_separate_selected_temperature_and_skipped_draw_artifacts(evidence, monkeypatch):
    root, config_path, config, selection, make_run = evidence
    reference = make_run("Ap", None, temperature=.8)
    selection.update(T_best=.8, Ap_is_A=False)
    selection["runs"]["Ap"] = reference
    config["manifests"]["Ap"] = reference["manifest"]
    write_json(root / "selection.json", selection)
    write_json(config_path, config)
    def routing_check(labels, scores, comparisons, **kwargs):
        assert kwargs["repeats"] == 2000
        assert not np.array_equal(scores["Ap"], scores["A"])
        # The statistical core is exercised end-to-end above and in its own
        # shared-bootstrap tests. Here inspect routing/persistence of a skip.
        mask = np.ones(2000, dtype=bool)
        mask[0] = False
        deltas = np.zeros((2000, 4, 2))
        deltas[0] = np.nan
        return {"indices": rng_for(10, 50).integers(0, 200, size=(2000, 200)), "valid_mask": mask,
                "subset_deltas": deltas, "point_subset_deltas": np.zeros((4, 2)),
                "attempted": 2000, "valid": 1999, "skipped": 1}
    monkeypatch.setattr(power, "shared_subset_bootstrap", routing_check)
    report = power.project_power(config_path)
    assert "alias_of" not in report["inputs"]["selected_runs"]["Ap"]
    assert report["bootstrap"]["skipped"] == 1
    assert report["boost_recommendation"]["recommended"] is None
    with np.load(root / "outputs/power/projection.npz", allow_pickle=False) as saved:
        assert np.isnan(saved["bootstrap_subset_deltas"][0]).all()
        assert saved["bootstrap_indices"].shape[0] == 2000


def _bootstrap(draws):
    return {"subset_deltas": np.repeat(np.asarray(draws, float)[:, None, :], 4, axis=1),
            "valid_mask": np.ones(len(draws), dtype=bool), "point_subset_deltas": np.zeros((4, 2))}


def test_projection_retains_negative_and_zero_states():
    comparisons, boost = power.projection_report(_bootstrap([[-1, 1], [0, 0], [1, -1]]))
    result = comparisons["D-C"]
    assert result["Cv"] == -1
    assert result["projections"]["J4"]["raw_variance"] == -.5
    assert result["projections"]["J4"]["se"] is None
    assert result["projections"]["J4"]["mde_80"] is None
    assert result["projections"]["J4"]["verdict7_probability_at_true_delta_zero"] is None
    assert result["projections"]["J2"]["se"] == 0
    assert boost["recommended"] is None
    _, zero = power.projection_report(_bootstrap(np.zeros((3, 2))))
    assert zero["diagnostic_status"] == "degenerate_zero_variance"
    assert zero["recommended"] is None


def test_boost_uses_only_strict_j4_threshold_and_j2_matches_sample_variance(monkeypatch):
    actual = power.projection_details
    def boundary(draws, **kwargs):
        result = actual(draws, **kwargs)
        if kwargs["target_subsets"] == 4:
            result["se"] = .0072
        return result
    monkeypatch.setattr(power, "projection_details", boundary)
    draws = np.array([[-1, -.5], [0, 1], [1, -.5]])
    comparisons, boost = power.projection_report(_bootstrap(draws))
    assert boost["recommended"] is False
    assert comparisons["D-C"]["projections"]["J2"]["raw_variance"] == pytest.approx(draws.mean(axis=1).var(ddof=1))
    assert "hypothetical" in comparisons["S-C"]["J8_execution_scope"]
