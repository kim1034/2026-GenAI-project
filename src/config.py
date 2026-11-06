"""Load and fingerprint experiment settings for the pre-test LOCK guard."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml


LOCK_PATTERN = re.compile(r"(?m)^config_sha256:\s*([0-9a-f]{64})\s*$")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("doc_version") != "1.3":
        raise ValueError("expected a v1.3 experiment config")
    return config


def config_sha256(config: dict, analysis_config: dict) -> str:
    canonical = yaml.safe_dump(
        {"base": config, "analysis": analysis_config}, sort_keys=True, allow_unicode=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_data_source(config: dict, root: Path) -> None:
    """Match the configured OpenAI commit and both local files to source records."""
    raw = root / "data" / "raw"
    source = json.loads((raw / "SOURCE.json").read_text(encoding="utf-8"))
    if (config["data"].get("source_repository") != source["repository"]
            or config["data"].get("revision") != source["repository_commit"]):
        raise ValueError("configured data source does not match the recorded OpenAI commit")
    recorded_sha256 = {}
    for line in (raw / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        recorded_sha256[name] = digest
    for name, entry in source["files"].items():
        content = (raw / name).read_bytes()
        blob = f"blob {len(content)}\0".encode("ascii") + content
        if (hashlib.sha1(blob).hexdigest() != entry["git_blob_sha"]
                or hashlib.sha256(content).hexdigest() != recorded_sha256.get(name)):
            raise ValueError(f"local data source file changed: {name}")


def assert_test_locked(config: dict, lock_path: Path) -> None:
    try:
        verify_data_source(config, lock_path.parent)
    except (OSError, KeyError, ValueError) as error:
        raise PermissionError("test split is sealed until the OpenAI source is verified") from error
    if not lock_path.is_file():
        raise PermissionError("test split is sealed until LOCK.md exists")
    content = lock_path.read_text(encoding="utf-8")
    match = LOCK_PATTERN.search(content)
    analysis_path = lock_path.parent / config["stats"]["analysis_list"]
    analysis_config = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    if match is None or match.group(1) != config_sha256(config, analysis_config):
        raise PermissionError("test split is sealed because the LOCK config hash does not match")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=lock_path.parent, capture_output=True, text=True
    )
    if status.returncode or status.stdout.strip():
        raise PermissionError("test split is sealed because the Git worktree is not clean")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lock_path.parent,
                          capture_output=True, text=True)
    locked = subprocess.run(["git", "rev-list", "-n", "1", "lock-v1"], cwd=lock_path.parent,
                            capture_output=True, text=True)
    if head.returncode or locked.returncode or head.stdout.strip() != locked.stdout.strip():
        raise PermissionError("test split is sealed because HEAD is not the lock-v1 commit")
