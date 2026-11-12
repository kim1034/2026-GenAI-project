import json
import subprocess

import pytest

from scripts import budget_lock_preflight as lock
from scripts import seal_and_run_budget_test as sealer
from scripts.run_experiment import sha256


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path
    git(root, "init")
    git(root, "config", "user.name", "Seal test")
    git(root, "config", "user.email", "seal@local.invalid")
    git(root, "config", "core.autocrlf", "false")
    (root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    (root / "core.py").write_text("reviewed code\n", encoding="utf-8")
    (root / "base.yaml").write_text('doc_version: "1.3"\n', encoding="utf-8")
    git(root, "add", ".gitignore", "core.py", "base.yaml")
    git(root, "commit", "-m", "Reviewed code R")
    anchor = git(root, "rev-parse", "HEAD")
    folder = root / sealer.SELECTION_DIR
    folder.mkdir(parents=True)
    names = ["A.yaml", "C0.yaml", "C 공통.yaml", "C2.yaml"]
    configs = [sealer.SELECTION_DIR + name for name in names]
    for index, name in enumerate(configs):
        (root / name).write_text(f"synthetic_configuration: {index}\n", encoding="utf-8")
    write_json(root / "outputs/grid.json", {"grid": "verified elsewhere"})
    write_json(root / "outputs/pilot.json", {"pilot": "verified elsewhere"})
    selection = {"a_run_config": configs[0], "c_run_configs": configs[1:],
                 "grid_decision": "outputs/grid.json", "pilot_plan": "outputs/pilot.json"}
    write_json(root / sealer.SELECTION, selection)
    write_json(folder / "stage_plan.json", {"commands": "exact preparer reconstruction"})
    selected_files = {name: sha256(root / name) for name in [*configs, sealer.SELECTION,
                      sealer.SELECTION_DIR + "stage_plan.json"]}
    final_path = root / sealer.FINAL
    analysis_path = final_path.parent / "pipeline_check_analysis_plan.json"
    write_json(analysis_path, {"output_dir": "outputs/check"})
    (final_path.parent / "test_C.yaml").write_text("frozen_test: C\n", encoding="utf-8")
    final = {"common_alpha": .1, "selection": "outputs/selection.json",
             "inputs_sha256": {name: digest for name, digest in selected_files.items() if not name.endswith("stage_plan.json")},
             "files_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in (analysis_path, final_path.parent / "test_C.yaml")},
             "development_check": {"mode": "reuse_selected_development", "analysis_plan": analysis_path.relative_to(root).as_posix()}}
    write_json(root / "outputs/selection.json", {"status": "complete"})
    write_json(final_path, final)
    write_json(root / "outputs/check/analysis.json", {"input_sha256": "synthetic-analysis-input"})
    artifacts = {"outputs/check/analysis.json": sha256(root / "outputs/check/analysis.json")}
    write_json(root / sealer.CPU_RECEIPT, {"schema_version": 1, "status": "passed_development_pipeline_check",
        "scope": "development_pipeline_check", "selection_data_reused": True, "independent_evaluation": False,
        "separate_CDE_rehearsal_omitted": True, "final_plan_sha256": sha256(final_path),
        "analysis_plan_sha256": sha256(analysis_path), "analysis_input_sha256": "synthetic-analysis-input",
        "artifacts_sha256": artifacts})
    monkeypatch.setattr(sealer, "prepare_final", lambda *args: {"final_plan": sealer.FINAL,
        "files_sha256": {**final["files_sha256"], sealer.FINAL: sha256(final_path)}})
    monkeypatch.setattr(sealer, "prepare_selection", lambda *args: {"files_sha256": selected_files})
    monkeypatch.setattr(sealer, "validate_rehearsal", lambda *args, **kwargs: None)
    monkeypatch.setattr(sealer, "_require_outer_lock", lambda root: None)
    monkeypatch.setattr(sealer, "_expected_parent", lambda root, reviewed: {"pid": 123, "reviewed_anchor": reviewed})
    commands = []
    def assert_seal(config, path):
        assert git(root, "rev-parse", "HEAD") == git(root, "rev-parse", "lock-v1^{commit}")
        assert not git(root, "status", "--porcelain")
    monkeypatch.setattr(lock, "assert_test_locked", assert_seal)
    def command(project, argv):
        commands.append(argv)
        if argv == ["scripts.budget_lock_preflight", "--write-lock"]:
            assert not git(root, "status", "--porcelain")
            receipt = {"status": "ready_for_lock", "reviewed_code_commit": git(root, "rev-parse", "HEAD"),
                       "base_config": "base.yaml", "evidence_sha256": {name: sha256(root / name)
                       for name in sealer._tracked(root, "HEAD")}}
            receipt["evidence_sha256"][sealer.CPU_RECEIPT] = sha256(root / sealer.CPU_RECEIPT)
            write_json(root / sealer.RECEIPT, receipt)
            (root / "LOCK.md").write_text("# LOCK\npreflight_sha256: " + sha256(root / sealer.RECEIPT)
                + "\nreviewed_code_commit: " + receipt["reviewed_code_commit"] + "\n", encoding="utf-8")
        elif argv == ["scripts.budget_lock_preflight", "--verify-lock"]:
            lock.verify_lock(root)  # Exercise the actual hash/receipt verifier.
        elif argv == ["scripts.run_budget_test"]:
            write_json(root / sealer.TEST_RECEIPT, {"status": "test_and_analysis_complete", "seal": lock.verify_lock(root),
                                                   "analysis_artifacts_sha256": {}})
        else:
            pytest.fail("Unexpected command " + repr(argv))
    monkeypatch.setattr(sealer, "_command", command)
    return root, anchor, final, commands


def test_only_verified_configs_F_then_LOCK_only_L_and_real_bundle(repository):
    root, anchor, _, commands = repository
    result = sealer.seal_and_run(root, reviewed_code_commit=anchor)
    reviewed, locked = result["reviewed_code_commit"], result["lock_commit"]
    assert git(root, "rev-parse", reviewed + "^") == anchor
    assert git(root, "rev-parse", locked + "^") == reviewed
    assert git(root, "diff", "--name-status", reviewed, locked) == "A\tLOCK.md"
    assert git(root, "rev-parse", "lock-v1^{commit}") == locked
    assert not git(root, "status", "--porcelain")
    assert commands == [["scripts.budget_lock_preflight", "--write-lock"],
                        ["scripts.budget_lock_preflight", "--verify-lock"], ["scripts.run_budget_test"]]
    git(root, "bundle", "verify", str(root / result["bundle"]))
    assert sha256(root / result["bundle"]) == result["bundle_sha256"]
    assert sealer.seal_and_run(root, reviewed_code_commit=anchor) == result
    assert git(root, "rev-list", "--count", "HEAD") == "3"


@pytest.mark.parametrize("dirty", ["core.py", "configs/generated/budget120/selection/unknown.yaml"])
def test_unknown_dirty_or_tracked_source_never_committed(repository, dirty):
    root, anchor, _, commands = repository
    path = root / dirty
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unreviewed modification\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty|verified generated"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == anchor
    assert not commands
    assert path.read_text() == "unreviewed modification\n"


@pytest.mark.parametrize("kind", ["commit", "blob"])
def test_conflicting_tag_is_rejected_before_any_commit(repository, kind):
    root, anchor, _, commands = repository
    if kind == "commit":
        git(root, "tag", "lock-v1", anchor)
    else:
        blob = git(root, "hash-object", "-w", "core.py")
        git(root, "update-ref", "refs/tags/lock-v1", blob)
    with pytest.raises(ValueError, match="lock-v1"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == anchor
    assert not commands


def test_cpu_artifact_tampering_fails_before_config_commit(repository):
    root, anchor, _, commands = repository
    (root / "outputs/check/analysis.json").write_text('{"input_sha256":"changed"}')
    with pytest.raises(ValueError, match="CPU pipeline"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == anchor
    assert not commands


def test_uncommitted_LOCK_resumes_without_another_F_or_preflight(repository, monkeypatch):
    root, anchor, _, commands = repository
    original = sealer._commit
    def interrupted(root, paths, message, **kwargs):
        if paths == ["LOCK.md"]:
            raise InterruptedError("simulated interruption after write-lock")
        return original(root, paths, message, **kwargs)
    monkeypatch.setattr(sealer, "_commit", interrupted)
    with pytest.raises(InterruptedError):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    reviewed = git(root, "rev-parse", "HEAD")
    assert (root / "LOCK.md").exists()
    monkeypatch.setattr(sealer, "_commit", original)
    result = sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert result["reviewed_code_commit"] == reviewed
    assert commands.count(["scripts.budget_lock_preflight", "--write-lock"]) == 1
    assert git(root, "rev-list", "--count", "HEAD") == "3"
    assert list((root / "outputs/logs/seal_budget_test").glob("*_failure.json"))


def test_receipt_or_source_tampering_on_resume_is_rejected(repository):
    root, anchor, _, _ = repository
    result = sealer.seal_and_run(root, reviewed_code_commit=anchor)
    receipt = root / sealer.RECEIPT
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="pin"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == result["lock_commit"]


def test_committed_LOCK_without_tag_resumes_same_L(repository, monkeypatch):
    root, anchor, _, commands = repository
    original = sealer._git
    def interrupted(project, *args, **kwargs):
        if args[:2] == ("tag", "lock-v1"):
            raise InterruptedError("simulated interruption immediately before tag")
        return original(project, *args, **kwargs)
    monkeypatch.setattr(sealer, "_git", interrupted)
    with pytest.raises(InterruptedError):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    locked = git(root, "rev-parse", "HEAD")
    assert not git(root, "status", "--porcelain")
    monkeypatch.setattr(sealer, "_git", original)
    result = sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert result["lock_commit"] == locked
    assert commands.count(["scripts.budget_lock_preflight", "--write-lock"]) == 1
    assert git(root, "rev-list", "--count", "HEAD") == "3"


@pytest.mark.parametrize("mutation", ["blob", "extra"])
def test_git_add_is_followed_by_exact_index_set_and_blob_validation(repository, monkeypatch, mutation):
    root, anchor, _, commands = repository
    original = sealer._git
    changed = []
    def alter_index(project, *args, **kwargs):
        result = original(project, *args, **kwargs)
        if args[0] == "add" and not changed:
            changed.append(True)
            if mutation == "blob":
                blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=root,
                                      input=b"wrong staged bytes", stdout=subprocess.PIPE, check=True).stdout.decode().strip()
                original(project, "update-index", "--cacheinfo", "100644", blob, args[2])
            else:
                (root / "core.py").write_text("hook-staged extra source", encoding="utf-8")
                original(project, "add", "--", "core.py")
        return result
    monkeypatch.setattr(sealer, "_git", alter_index)
    with pytest.raises(ValueError, match="Staged"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == anchor
    assert not commands


def test_unreviewed_new_HEAD_is_not_mistaken_for_generated_F(repository):
    root, anchor, _, commands = repository
    (root / "core.py").write_text("unreviewed committed change", encoding="utf-8")
    git(root, "add", "core.py")
    git(root, "commit", "-m", "Unreviewed change")
    unreviewed = git(root, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="Generated F"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert git(root, "rev-parse", "HEAD") == unreviewed
    assert not commands


def test_post_commit_generated_byte_change_is_detected(repository, monkeypatch):
    root, anchor, _, commands = repository
    original = sealer._git
    changed = []
    def hook(project, *args, **kwargs):
        if "commit" in args and not changed:
            name = args[args.index("--") + 1]
            (root / name).write_text("changed by commit hook\n", encoding="utf-8")
            original(project, "add", "--", name)
            changed.append(name)
        return original(project, *args, **kwargs)
    monkeypatch.setattr(sealer, "_git", hook)
    with pytest.raises(ValueError, match="Committed generated bytes"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert not commands
    assert not (root / "LOCK.md").exists()


def test_canonical_conflicting_hashes_are_not_silently_overwritten(repository):
    root, _, _, _ = repository
    with pytest.raises(ValueError, match="Conflicting hashes"):
        sealer._merge(root, {"base.yaml": "one"}, {"./base.yaml": "two"})


def test_symlink_config_is_rejected(repository):
    root, anchor, _, commands = repository
    target = root / (sealer.SELECTION_DIR + "A.yaml")
    original = target.read_bytes()
    target.unlink()
    try:
        target.symlink_to(root / "core.py")
    except OSError:
        target.write_bytes(original)
        pytest.skip("This Windows account cannot create symbolic links")
    with pytest.raises(ValueError, match="Symlink"):
        sealer.seal_and_run(root, reviewed_code_commit=anchor)
    assert not commands
