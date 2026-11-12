"""Commit only verified generated configs, seal LOCK, run test and bundle code.

Runs exclusively inside an existing outer run_stage lock. No forced tags,
source edits, human-signature claims, GPU return or elapsed-time stopping.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

from scripts.budget_lock_preflight import (FINAL, RECEIPT, lock_document, validate_pipeline_receipt,
                                           validate_rehearsal, verify_lock)
from scripts.finalize_budget_selection import _command, _require_outer_lock
from scripts.prepare_budget_final_runs import prepare as prepare_final
from scripts.prepare_budget_selection import prepare as prepare_selection
from scripts.run_experiment import ROOT, sha256


CPU_RECEIPT = "outputs/reproducibility/development_pipeline_check.json"
TEST_RECEIPT = "outputs/reproducibility/final_test_receipt.json"
SELECTION = "configs/generated/budget120/selection/selection_plan.json"
SELECTION_DIR = "configs/generated/budget120/selection/"
FINAL_DIR = "configs/generated/budget120/final/"
OUTER_STAGE = "budget120_sealed_test_followthrough"


def _git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def _text(root, *args):
    return _git(root, *args).stdout.decode("utf-8").strip()


def _path(root, name):
    path = Path(os.path.abspath(root / name))
    if not path.is_relative_to(root):
        raise ValueError("Seal evidence escapes project root")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError("Symlink evidence is forbidden: " + str(path))
        if candidate == root:
            break
    if path.resolve() != path:
        raise ValueError("Noncanonical/symlink artifact path is forbidden")
    return path


def _read(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Evidence must be a regular file without symlink parents")
    return json.loads(path.read_text(encoding="utf-8"))


def _pins(root, hashes):
    for name, expected in hashes.items():
        path = _path(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Seal evidence changed: " + name)


def _merge(root, *maps):
    result = {}
    for mapping in maps:
        for name, digest in mapping.items():
            key = _path(root, name).relative_to(root).as_posix()
            if key in result and result[key] != digest:
                raise ValueError("Conflicting hashes for canonical evidence path: " + key)
            result[key] = digest
    return result


def _changes(root):
    fields = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split(b"\0")
    changes, index = [], 0
    while index < len(fields) and fields[index]:
        item = fields[index]
        status, name = item[:2].decode("ascii"), item[3:].decode("utf-8", errors="surrogateescape")
        changes.append((status, name))
        index += 2 if "R" in status or "C" in status else 1
    return changes


def _head(root):
    return _text(root, "rev-parse", "HEAD")


def _tag(root):
    exists = _git(root, "show-ref", "--verify", "--quiet", "refs/tags/lock-v1", check=False)
    if exists.returncode == 1:
        return None
    if exists.returncode != 0:
        raise ValueError("Invalid existing lock-v1 reference")
    result = _git(root, "rev-parse", "--verify", "refs/tags/lock-v1^{commit}", check=False)
    if result.returncode:
        raise ValueError("Existing lock-v1 tag does not identify a commit")
    return result.stdout.decode().strip()


def _record(root, phase, **data):
    folder = _path(root, "outputs/logs/seal_budget_test")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    receipt = {"phase": phase, "at_utc": stamp.isoformat(), **data}
    raw = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with (folder / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + "_" + phase + ".json")).open("xb") as handle:
        handle.write(raw)


def _tracked(root, revision):
    return {name.decode("utf-8", errors="surrogateescape") for name in
            _git(root, "ls-tree", "-r", "--name-only", "-z", revision).stdout.split(b"\0") if name}


def _lock_child(root, reviewed, lock_commit):
    parents = _text(root, "rev-list", "--parents", "-n", "1", lock_commit).split()
    if parents != [lock_commit, reviewed]:
        raise ValueError("LOCK commit must be the direct single-parent child of reviewed F")
    if _git(root, "diff", "--name-status", "-z", reviewed, lock_commit, "--").stdout != b"A\0LOCK.md\0":
        raise ValueError("LOCK commit must add only LOCK.md")


def _check_index(root, paths, expected):
    records = _git(root, "diff", "--cached", "--name-status", "-z", "HEAD", "--").stdout.split(b"\0")
    observed = {tuple(records[i:i + 2]) for i in range(0, len(records) - 1, 2)}
    if observed != {(b"A", name.encode("utf-8")) for name in paths} or set(expected) != set(paths):
        raise ValueError("Staged file set differs from the exact authorized new files")
    for name in paths:
        header = _git(root, "ls-files", "--stage", "--", name).stdout.split(b"\t", 1)[0].split()
        if len(header) != 3 or header[0] not in (b"100644", b"100755") or header[2] != b"0":
            raise ValueError("Staged configuration must be a regular, resolved file")
        if hashlib.sha256(_git(root, "show", ":" + name).stdout).hexdigest() != expected[name]:
            raise ValueError("Staged blob SHA differs from verified generated bytes")


def _commit(root, paths, message, *, expected_sha256):
    if paths:
        _git(root, "add", "--", *paths)
    _check_index(root, paths, expected_sha256)
    args = ["-c", "user.name=Codex", "-c", "user.email=codex@local.invalid", "commit", "-m", message]
    args.extend(["--only", "--", *paths] if paths else ["--allow-empty"])
    _git(root, *args)
    return _head(root)


def _expected_parent(root, reviewed):
    parent = Path("/proc") / str(os.getppid())
    argv = (parent / "cmdline").read_bytes().split(b"\0")
    if (parent / "cwd").resolve() != root or b"-m" not in argv:
        raise ValueError("Sealer must be a direct child of the expected outer run_stage")
    index = argv.index(b"-m")
    if len(argv) <= index + 2 or argv[index + 1] != b"scripts.run_stage":
        raise ValueError("Unexpected direct parent module")
    plan_path = _path(root, os.fsdecode(argv[index + 2]))
    plan = _read(plan_path)
    expected = [["-m", "scripts.seal_and_run_budget_test", "--reviewed-code-commit", reviewed]]
    if plan.get("name") != OUTER_STAGE or plan.get("commands") != expected:
        raise ValueError("Outer stage does not identify this reviewed seal invocation")
    return {"pid": os.getppid(), "plan": plan_path.relative_to(root).as_posix(), "plan_sha256": sha256(plan_path)}


def _revalidate_generated(root):
    final_path = root / FINAL
    final = _read(final_path)
    _merge(root, final["inputs_sha256"], final["files_sha256"])
    prepared = prepare_final(root, _path(root, final["selection"]), final_path.parent)
    final = _read(final_path)
    if prepared["final_plan"] != FINAL or set(prepared["files_sha256"]) != set(final["files_sha256"]) | {FINAL}:
        raise ValueError("Final preparer file set differs from the pinned plan")
    _pins(root, final["inputs_sha256"])
    _pins(root, final["files_sha256"])
    _pins(root, prepared["files_sha256"])
    for name, digest in final["files_sha256"].items():
        if prepared["files_sha256"][name] != digest or not name.startswith(FINAL_DIR):
            raise ValueError("Final generated whitelist differs from preparer bytes")
    selection_path = root / SELECTION
    selection = _read(selection_path)
    prepared_selection = prepare_selection(root, _path(root, selection["grid_decision"]),
                                            _path(root, selection["pilot_plan"]), selection_path.parent)
    expected = {selection["a_run_config"], *selection["c_run_configs"], SELECTION,
                selection_path.parent.relative_to(root).as_posix() + "/stage_plan.json"}
    if len(expected) != 6 or set(prepared_selection["files_sha256"]) != expected:
        raise ValueError("Selection preparer must reproduce exactly four YAMLs, selection_plan and stage_plan")
    _pins(root, prepared_selection["files_sha256"])
    if SELECTION not in final["inputs_sha256"]:
        raise ValueError("Final evidence does not pin the original selection plan")
    whitelist = {name: digest for name, digest in final["inputs_sha256"].items() if name.startswith(SELECTION_DIR)}
    whitelist = _merge(root, whitelist, final["files_sha256"], prepared_selection["files_sha256"])
    whitelist[FINAL] = sha256(final_path)
    if any(not name.startswith((SELECTION_DIR, FINAL_DIR)) or Path(name).suffix not in (".json", ".yaml") for name in whitelist):
        raise ValueError("Whitelist must contain only verified generated configuration files")
    _pins(root, whitelist)
    development = final["development_check"]
    if development.get("mode") != "reuse_selected_development":
        raise ValueError("This orchestrator requires the completed CPU selection-data pipeline check")
    plan_path = _path(root, development["analysis_plan"])
    plan = _read(plan_path)
    folder = _path(root, plan["output_dir"])
    report = _read(folder / "analysis.json")
    artifacts = {p.relative_to(root).as_posix(): sha256(p) for p in folder.iterdir() if p.is_file()}
    validate_rehearsal(report, final["common_alpha"], reuse_selection=True)
    validate_pipeline_receipt(_read(root / CPU_RECEIPT), report, final_sha=sha256(final_path),
                              plan_sha=sha256(plan_path), artifact_sha=artifacts)
    return whitelist


def _generated_child(root, anchor, reviewed, whitelist):
    if _text(root, "rev-list", "--parents", "-n", "1", reviewed).split() != [reviewed, anchor]:
        raise ValueError("Generated F must be a direct single-parent child of reviewed code R")
    records = _git(root, "diff", "--name-status", "-z", anchor, reviewed, "--").stdout.split(b"\0")
    for index in range(0, len(records) - 1, 2):
        status, name = records[index:index + 2]
        key = name.decode("utf-8", errors="surrogateescape")
        if status != b"A" or key not in whitelist:
            raise ValueError("Generated F changed more than verified new configurations")
    for name, digest in whitelist.items():
        if hashlib.sha256(_git(root, "show", reviewed + ":" + name).stdout).hexdigest() != digest:
            raise ValueError("Committed generated bytes do not match the verified file hash")


def _generated_commit(root, whitelist, anchor):
    before = _head(root)
    if before != anchor:
        _generated_child(root, anchor, before, whitelist)
        if _changes(root):
            raise ValueError("Resumed generated F must have a clean worktree")
        return before
    changes = _changes(root)
    tracked_before = _tracked(root, before)
    for status, name in changes:
        if status not in ("??", "A ") or name not in whitelist or name in tracked_before:
            raise ValueError("Only new verified generated configs may be committed: " + name)
        if status == "A ":
            staged = _git(root, "show", ":" + name).stdout
            if hashlib.sha256(staged).hexdigest() != whitelist[name]:
                raise ValueError("Staged generated configuration differs from pinned bytes")
    paths = sorted(name for _, name in changes)
    reviewed = _commit(root, paths, "Freeze verified budget120 generated configurations",
                       expected_sha256={name: whitelist[name] for name in paths})
    records = _git(root, "diff", "--name-status", "-z", before, reviewed, "--").stdout.split(b"\0")
    if {tuple(records[i:i + 2]) for i in range(0, len(records) - 1, 2)} != {(b"A", p.encode("utf-8")) for p in paths}:
        raise ValueError("Generated commit changed files outside its exact whitelist")
    _generated_child(root, anchor, reviewed, whitelist)
    if _changes(root):
        raise ValueError("Worktree must be clean after generated config commit")
    for name, digest in whitelist.items():
        blob = _git(root, "show", reviewed + ":" + name).stdout
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Committed generated bytes do not match the verified file hash")
    return reviewed


def _pending_seal(root):
    receipt_path = root / RECEIPT
    receipt = _read(receipt_path)
    lock_path = _path(root, "LOCK.md")
    if not lock_path.is_file():
        raise ValueError("LOCK must be a regular file")
    content = lock_path.read_text(encoding="utf-8")
    reviewed = receipt["reviewed_code_commit"]
    if (receipt.get("status") != "ready_for_lock"
            or re.search(r"(?m)^preflight_sha256: " + re.escape(sha256(receipt_path)) + r"$", content) is None
            or re.search(r"(?m)^reviewed_code_commit: " + re.escape(reviewed) + r"$", content) is None
            or "LOCK.md" in _tracked(root, reviewed)):
        raise ValueError("LOCK does not pin a valid preflight receipt/reviewed code commit")
    if lock_path.read_bytes() != lock_document(receipt, sha256(receipt_path)):
        raise ValueError("LOCK content differs from canonical preflight bytes before commit")
    _pins(root, receipt["evidence_sha256"])
    if receipt["evidence_sha256"].get(CPU_RECEIPT) != sha256(root / CPU_RECEIPT):
        raise ValueError("LOCK receipt must pin the verified CPU pipeline receipt")
    if not _tracked(root, reviewed) <= set(receipt["evidence_sha256"]):
        raise ValueError("LOCK receipt omits reviewed tracked source/config evidence")
    head = _head(root)
    changes = _changes(root)
    if head == reviewed:
        if any(name != "LOCK.md" or status not in ("??", "A ") for status, name in changes):
            raise ValueError("Only the new LOCK.md may be dirty at reviewed F")
        if _tag(root) is not None:
            raise ValueError("Existing lock-v1 tag conflicts before the LOCK-only child commit")
    else:
        _lock_child(root, reviewed, head)
        if changes:
            raise ValueError("Committed LOCK child must have a clean worktree")
        if _tag(root) not in (None, head):
            raise ValueError("Existing lock-v1 tag points to a different commit")
    return reviewed, head


def _bundle(root, anchor, reviewed, locked):
    if _head(root) != locked or _tag(root) != locked or _changes(root):
        raise ValueError("Test changed the sealed source commit or tag")
    test_receipt = _read(root / TEST_RECEIPT)
    if (test_receipt.get("status") != "test_and_analysis_complete"
            or test_receipt["seal"]["lock_commit"] != locked
            or test_receipt["seal"]["LOCK_sha256"] != sha256(root / "LOCK.md")):
        raise ValueError("Sealed test completion receipt is missing or inconsistent")
    _pins(root, test_receipt["analysis_artifacts_sha256"])
    path = _path(root, "outputs/reproducibility/budget120_lock_v1.bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _git(root, "bundle", "create", str(path), "HEAD", "refs/tags/lock-v1")
    _git(root, "bundle", "verify", str(path))
    with path.open("rb") as handle:
        header = handle.read().split(b"\n\n", 1)[0].splitlines()
    if any(line.startswith(b"-") for line in header):
        raise ValueError("Code bundle must include full history, not require external prerequisite commits")
    heads = {name: sha for sha, name in (line.split(" ", 1) for line in _text(root, "bundle", "list-heads", str(path)).splitlines())}
    if heads.get("HEAD") != locked or heads.get("refs/tags/lock-v1") != _text(root, "rev-parse", "refs/tags/lock-v1"):
        raise ValueError("Bundle does not contain the sealed HEAD and lock-v1 tag")
    # F and R must be in the ancestry included by the sealed L/HEAD bundle.
    _git(root, "merge-base", "--is-ancestor", anchor, locked)
    _git(root, "merge-base", "--is-ancestor", reviewed, locked)
    result = {"status": "verified_sealed_test_bundle", "reviewed_anchor_commit": anchor, "reviewed_code_commit": reviewed, "lock_commit": locked,
              "bundle_scope": "tracked_git_history_and_lock_tag",
              "tag": "lock-v1", "bundle": path.relative_to(root).as_posix(), "bundle_sha256": sha256(path),
              "LOCK_sha256": sha256(root / "LOCK.md"), "preflight_sha256": sha256(root / RECEIPT),
              "CPU_receipt_sha256": sha256(root / CPU_RECEIPT), "test_receipt_sha256": sha256(root / TEST_RECEIPT)}
    target = path.with_name("budget120_lock_bundle_receipt.json")
    raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("Existing bundle receipt differs; no overwrite is allowed")
    if not target.exists():
        with target.open("xb") as handle:
            handle.write(raw)
    return result


def seal_and_run(root, *, reviewed_code_commit):
    root = Path(root).resolve()
    try:
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", reviewed_code_commit) is None:
            raise ValueError("Explicit full reviewed code commit R is required")
        anchor = _text(root, "rev-parse", reviewed_code_commit + "^{commit}")
        parent = _expected_parent(root, anchor)
        _require_outer_lock(root)
        lock_exists = (root / "LOCK.md").exists()
        if _tag(root) is not None and not lock_exists:
            raise ValueError("Existing lock-v1 tag blocks a new seal; never force or move it")
        if lock_exists:
            reviewed, _ = _pending_seal(root)  # Validate resume before preparers create files.
            if _text(root, "rev-list", "--parents", "-n", "1", reviewed).split() != [reviewed, anchor]:
                raise ValueError("Resumed F is not the direct child of the reviewed R anchor")
        else:
            for status, name in _changes(root):
                if status not in ("??", "A ") or not name.startswith((SELECTION_DIR, FINAL_DIR)):
                    raise ValueError("Unexpected dirty source/configuration before sealing: " + name)
        whitelist = _revalidate_generated(root)
        _record(root, "generated_evidence_verified", reviewed_anchor_commit=anchor, parent_stage=parent,
                files_sha256=whitelist, CPU_receipt_sha256=sha256(root / CPU_RECEIPT))
        if not lock_exists:
            if (root / RECEIPT).exists():
                existing = _read(root / RECEIPT)
                if existing["reviewed_code_commit"] != _head(root):
                    raise ValueError("Existing preflight receipt belongs to another reviewed commit")
                _pins(root, existing["evidence_sha256"])
            reviewed = _generated_commit(root, whitelist, anchor)
            _record(root, "generated_config_commit", reviewed_code_commit=reviewed)
            if _tag(root) is not None:
                raise ValueError("lock-v1 appeared before LOCK creation")
            _command(root, ["scripts.budget_lock_preflight", "--write-lock"])
            if _head(root) != reviewed:
                raise ValueError("Preflight changed the reviewed Git commit")
            observed_reviewed, _ = _pending_seal(root)
            if observed_reviewed != reviewed:
                raise ValueError("Preflight receipt does not identify reviewed config commit F")
        reviewed, head = _pending_seal(root)
        _generated_child(root, anchor, reviewed, whitelist)
        lock_hash = sha256(root / "LOCK.md")
        if head == reviewed:
            locked = _commit(root, ["LOCK.md"], "Seal budget120 LOCK evidence", expected_sha256={"LOCK.md": lock_hash})
            _lock_child(root, reviewed, locked)
        else:
            locked = head
        if (_changes(root) or sha256(root / "LOCK.md") != lock_hash
                or hashlib.sha256(_git(root, "show", locked + ":LOCK.md").stdout).hexdigest() != lock_hash):
            raise ValueError("LOCK commit/hook changed verified bytes or left a dirty worktree")
        if _tag(root) is None:
            _git(root, "tag", "lock-v1", locked)
        elif _tag(root) != locked:
            raise ValueError("Refusing to move an existing lock-v1 tag")
        _record(root, "lock_committed_and_tagged", reviewed_code_commit=reviewed, lock_commit=locked)
        _command(root, ["scripts.budget_lock_preflight", "--verify-lock"])
        seal = verify_lock(root)
        _record(root, "seal_verified", **seal)
        _command(root, ["scripts.run_budget_test"])
        result = _bundle(root, anchor, reviewed, locked)
        _record(root, "test_and_bundle_verified", **result)
        return result
    except BaseException as error:
        _record(root, "failure", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-code-commit", required=True)
    arguments = parser.parse_args()
    print(json.dumps(seal_and_run(ROOT, reviewed_code_commit=arguments.reviewed_code_commit), indent=2))
