"""Precompute a noise bank from one run configuration, without model inference."""

import argparse
from pathlib import Path

import yaml

from src.bank import build_bank
from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config", type=Path)
    args = parser.parse_args()
    run = yaml.safe_load(args.run_config.read_text(encoding="utf-8"))
    config = load_config(ROOT / run.get("base_config", "configs/base.yaml"))
    split = run["split"]
    ids = (ROOT / "data" / "splits" / f"{split}.ids").read_text(encoding="ascii").splitlines()
    path, manifest = build_bank(
        ROOT, ids, config, condition=run["condition"], split=split,
        seed_set=run["seed_set"], row_start=run.get("row_start", 0), n_rows=run["n_rows"],
    )
    print(f"bank: {path.relative_to(ROOT)}")
    print(f"manifest: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
