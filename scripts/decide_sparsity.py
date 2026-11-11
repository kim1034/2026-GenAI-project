"""Freeze S sparsity from verified complete development gradients before noise inference."""

import csv
import json
import numpy as np

from scripts.run_experiment import ROOT, json_write, sha256
from src.config import load_config
from src.diag import effective_dimension, sparse_decision


def main():
    base = load_config(ROOT / "configs/base.yaml")
    if base["noise"]["sparse_rule"]["candidates"] != [.001, .002, .005, .01, .02] or base["noise"]["heterogeneity"]["threshold"] != .3:
        raise RuntimeError("Configured sparsity rule differs from the implemented frozen rule")
    folder = ROOT / "outputs/diag"
    sources, values = {}, []
    all_ids = set()
    split_hashes = dict((name, digest) for digest, name in (line.split() for line in (ROOT / "data/splits/SHA256SUMS").read_text().splitlines()))
    fd_path = folder / "dev_base_fd_status.json"
    fd = json.loads(fd_path.read_text())
    if fd["status"] != "passed" or fd["completed"] != 3 or fd["split"] != "dev_base" or fd["full_split"]:
        raise RuntimeError("Finite-difference diagnostic has not passed")
    sources[str(fd_path.relative_to(ROOT))] = sha256(fd_path)
    for split in ("dev_base", "dev_select", "dev_anal"):
        status_path = folder / f"{split}_status.json"
        status = json.loads(status_path.read_text())
        ids_path = ROOT / f"data/splits/{split}.ids"
        ids = ids_path.read_text().splitlines()
        if len(ids) != {"dev_base": 50, "dev_select": 200, "dev_anal": 50}[split] or len(set(ids)) != len(ids) or all_ids.intersection(ids) or any(not i.startswith("train-") for i in ids):
            raise RuntimeError("Development split identity or size is invalid")
        all_ids.update(ids)
        if sha256(ids_path) != split_hashes[ids_path.name]:
            raise RuntimeError("Development split differs from its frozen SHA256SUMS")
        if status["status"] != "passed" or status["split"] != split or not status["full_split"] or status["completed"] != len(ids):
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
            entry_path = folder / split / f'{row["id"]}.json'
            array_entry = entry_path.with_suffix(".npz")
            entry = json.loads(entry_path.read_text())
            greedy_path = ROOT / f'outputs/samples/{split}_greedy_aNA_T0.00_NA_s0_k1/{row["id"]}.json'
            expected = {"rho_sha256": sha256(ROOT / base["noise"]["rho_file"]),
                        "source_sha256": sha256(ROOT / "src/gradient_diag.py"), "model_revision": base["model"]["revision"],
                        "greedy_sha256": sha256(greedy_path)}
            if any(entry.get(key) != value or row.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Gradient diagnostic provenance differs from its frozen inputs")
            with np.load(array_entry) as stored:
                if not np.array_equal(stored["g_m"], gradient):
                    raise RuntimeError("Combined and problem-level gradient arrays differ")
            if split == "dev_base" and row["id"] in ids[:3]:
                checks = entry.get("finite_difference", {})
                if set(checks) != {"g_m", "g_seq"} or any(not check.get("passed") or not np.isfinite(check["relative_error"]) or not 0 <= check["relative_error"] < .05 for check in checks.values()):
                    raise RuntimeError("Missing or failed original finite-difference evidence")
            for evidence in (entry_path, array_entry, greedy_path):
                sources[str(evidence.relative_to(ROOT))] = sha256(evidence)
            recorded = float(row["deff_g_m"])
            if not np.isclose(recorded, effective_dimension(gradient), rtol=1e-6, atol=1e-5):
                raise RuntimeError("Gradient effective dimension differs from CSV")
            values.append(recorded)
        for path in (status_path, ids_path, csv_path, array_path):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    decision = {**sparse_decision(values, dimension=base["noise"]["d"]), "inputs_sha256": sources}
    decision["base_config_sha256"] = sha256(ROOT / "configs/base.yaml")
    decision["rule"] = {"candidates": [.001, .002, .005, .01, .02], "inverse_clip": [.001, .02], "heterogeneity_threshold": .3}
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
