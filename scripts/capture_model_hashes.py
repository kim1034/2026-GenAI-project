"""Hash the pinned local model snapshot without downloading or changing it."""

import hashlib
from pathlib import Path

from scripts.run_experiment import ROOT, json_write
from src.config import load_config


def main():
    config = load_config(ROOT / "configs/base.yaml")
    folder = Path.home() / ".cache/huggingface/hub" / ("models--" + config["model"]["id"].replace("/", "--")) / "snapshots" / config["model"]["revision"]
    if not folder.is_dir():
        raise RuntimeError("Pinned model snapshot is not present in the default cache")
    entries = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"file": str(path.relative_to(folder)), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    if not any(item["file"].endswith(".safetensors") for item in entries):
        raise RuntimeError("No cached model weight file found")
    json_write(ROOT / "models/files.json", {"model_id": config["model"]["id"], "revision": config["model"]["revision"], "files": entries})
    (ROOT / "models/SHA256SUMS").write_text("".join(f'{item["sha256"]}  {item["file"]}\n' for item in entries), encoding="utf-8")
    print(f"Hashed {len(entries)} pinned model files")


if __name__ == "__main__":
    main()
