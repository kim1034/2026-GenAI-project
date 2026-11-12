"""Precompute a noise bank from one run configuration, without model inference."""

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path

import yaml

from src.bank import bank_path, build_bank
from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def bank_write_lock(root, path):
    """Serialize writers to the actual bank, independent of alpha/run ID."""
    import fcntl
    key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    folder = root / "outputs/logs/bank_locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (key + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_locked(root, ids, config, run, *, workers):
    basis = "B1" if run["condition"] in ("S_H", "T5_H") else "B0"
    path = bank_path(root, run["condition"], basis, run["seed_set"], run["split"],
                     run.get("row_start", 0), run["n_rows"])
    with bank_write_lock(root, path):
        # build_bank validates existing/partial/final identity and SHA under the lock.
        return build_bank(root, ids, config, condition=run["condition"], split=run["split"],
                          seed_set=run["seed_set"], row_start=run.get("row_start", 0),
                          n_rows=run["n_rows"], workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Independent GUE/GOE coefficient workers")
    args = parser.parse_args()
    run = yaml.safe_load(args.run_config.read_text(encoding="utf-8"))
    config = load_config(ROOT / run.get("base_config", "configs/base.yaml"))
    split = run["split"]
    ids = (ROOT / "data" / "splits" / f"{split}.ids").read_text(encoding="ascii").splitlines()
    path, manifest = build_locked(ROOT, ids, config, run, workers=args.workers)
    print(f"bank: {path.relative_to(ROOT)}")
    print(f"manifest: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
