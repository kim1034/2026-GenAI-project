"""Inventory project evidence for incremental verified backups; never include credentials."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["outputs/samples", "outputs/manifests", "outputs/diag", "outputs/diagnostics", "outputs/calib", "outputs/reproducibility", "outputs/analysis",
           "outputs/logs", "calib", "bank", "reports", "configs", "src", "scripts", "tests", "prompts", "data/splits", "data/raw", "data/processed", "models"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    cache_path = ROOT / "outputs/backup_inventory_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    inventory, updated = [], {}
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if path.name.endswith((".tmp", ".lock", ".state.json", ".partial.npy")):
                # Stage state is included below; bank state describes unfinished arrays.
                if not (folder == "outputs/logs" and path.name.endswith(".state.json")):
                    continue
            relative = path.relative_to(ROOT).as_posix()
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError("Evidence path escaped project root")
            before = path.stat()
            signature = [before.st_size, before.st_mtime_ns]
            prior = cache.get(relative)
            if prior and prior["signature"] == signature and not args.full_hash:
                digest = prior["sha256"]
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                after = path.stat()
                if [after.st_size, after.st_mtime_ns] != signature:
                    continue
                digest = hasher.hexdigest()
            updated[relative] = {"signature": signature, "sha256": digest}
            inventory.append({"path": relative, "size": before.st_size, "sha256": digest})
    # Root protocol documents and LOCK are reproducibility evidence too.
    roots = set(ROOT.glob("*.md")) | set(ROOT.glob("requirements*.txt"))
    roots.update(ROOT / name for name in ("outputs/lock_preflight.json", "outputs/lock_verified.json"))
    for path in sorted(roots):
        if path.is_file() and not path.is_symlink():
            inventory.append({"path": path.relative_to(ROOT).as_posix(),
                              "size": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(updated), encoding="utf-8")
    temporary.replace(cache_path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    print(json.dumps({"git_head": head, "git_status": status, "files": inventory}))


if __name__ == "__main__":
    main()
