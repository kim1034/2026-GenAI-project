"""Freeze S sparsity from verified complete development gradients before noise inference."""

import csv
import json
import numpy as np

from scripts.run_experiment import ROOT, json_write, sha256
from src.config import load_config
from src.diag import effective_dimension, sparse_decision


def main():
    base = load_config(ROOT / "configs/base.yaml")
    folder = ROOT / "outputs/diag"
    sources, values = {}, []
    fd_path = folder / "dev_base_fd_status.json"
    fd = json.loads(fd_path.read_text())
    if fd["status"] != "passed" or fd["completed"] != 3:
        raise RuntimeError("Finite-difference diagnostic has not passed")
    sources[str(fd_path.relative_to(ROOT))] = sha256(fd_path)
    for split in ("dev_base", "dev_select", "dev_anal"):
        status_path = folder / f"{split}_status.json"
        status = json.loads(status_path.read_text())
        ids_path = ROOT / f"data/splits/{split}.ids"
        ids = ids_path.read_text().splitlines()
        if status["status"] != "passed" or not status["full_split"] or status["completed"] != len(ids):
            raise RuntimeError(f"Incomplete gradient diagnostic: {split}")
        csv_path, array_path = folder / f"{split}_diag.csv", folder / f"{split}_g_m.npy"
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [row["id"] for row in rows] != ids:
            raise RuntimeError("Gradient diagnostic IDs differ from the frozen split")
        gradients = np.load(array_path, mmap_mode="r")
        if gradients.shape != (len(ids), base["noise"]["d"]):
            raise RuntimeError("Wrong gradient array shape")
        for row, gradient in zip(rows, gradients):
            recorded = float(row["deff_g_m"])
            if not np.isclose(recorded, effective_dimension(gradient), rtol=1e-6, atol=1e-5):
                raise RuntimeError("Gradient effective dimension differs from CSV")
            values.append(recorded)
        for path in (status_path, ids_path, csv_path, array_path):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    decision = {**sparse_decision(values, dimension=base["noise"]["d"]), "inputs_sha256": sources}
    if len(values) != 300:
        raise RuntimeError("S selection requires all 300 development problems")
    path = ROOT / "calib/sparse_decision.json"
    if path.exists():
        if json.loads(path.read_text()) != decision:
            raise RuntimeError("Existing sparsity decision differs; preserve it and inspect")
    else:
        json_write(path, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
