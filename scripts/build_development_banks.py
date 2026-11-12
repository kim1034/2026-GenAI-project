"""Build development banks without choosing any inference magnitudes or reading test."""

import argparse

from scripts.run_experiment import ROOT
from src.bank import build_bank
from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev_base", "dev_select", "dev_anal"], default="dev_base")
    parser.add_argument("--conditions", nargs="+", choices=["B", "C", "D", "E", "T5", "S"], default=["B", "C", "D", "E", "T5"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.base_config)
    ids = (ROOT / f"data/splits/{args.split}.ids").read_text().splitlines()
    n_rows = config["decode"]["k_generate_rehearsal"] if args.split == "dev_anal" else config["decode"]["k_generate_dev"]
    for condition in args.conditions:
        print(f"Building {args.split} {condition}", flush=True)
        path, manifest = build_bank(ROOT, ids, config, condition=condition, split=args.split,
            seed_set=0, row_start=0, n_rows=n_rows, workers=args.workers)
        print(f"Completed {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
