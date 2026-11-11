"""T-NS-2: compare independent dense and tridiagonal GUE spectra at d=1000."""

import json
import argparse
import numpy as np
from scipy.stats import ks_2samp

from scripts.run_experiment import ROOT, json_write
from scripts.check_manipulation import gap_ratio
from src.noise import beta_hermite_spectrum
from src.rng import rng_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true", help="One fixed extension to 100 replicates, retaining the original ten")
    args = parser.parse_args()
    stem = "dense_gue_n100" if args.extended else "dense_gue"
    path = ROOT / f"outputs/diagnostics/{stem}.json"
    if path.exists():
        raise RuntimeError("Preserve and inspect existing evidence before repeating this diagnostic")
    d, repeats = 1000, 100 if args.extended else 10
    dense, tridiagonal = [], []
    if args.extended:
        previous = np.load(ROOT / "outputs/diagnostics/dense_gue_spectra.npz")
        if previous["dense"].shape != (10, d) or previous["tridiagonal"].shape != (10, d):
            raise RuntimeError("Expected the preserved ten-replicate diagnostic")
        dense, tridiagonal = list(previous["dense"]), list(previous["tridiagonal"])
    # Key 18 is reserved here for this generator implementation check, not model responses.
    for i in range(len(dense), repeats):
        r = rng_for(18, 0, i)
        matrix = r.standard_normal((d, d)) + 1j * r.standard_normal((d, d))
        matrix = (matrix + matrix.conj().T) / 2
        dense.append(np.linalg.eigvalsh(matrix))
        tridiagonal.append(beta_hermite_spectrum(d, rng_for(18, 1, i), beta=2))
    dense, tridiagonal = np.stack(dense), np.stack(tridiagonal)
    dense_ratio, tri_ratio = gap_ratio(dense), gap_ratio(tridiagonal)
    ks = float(ks_2samp(dense.ravel() / np.sqrt(d), tridiagonal.ravel() / np.sqrt(d), method="asymp").statistic)
    per_dense = np.array([gap_ratio(row[None, :]) for row in dense])
    per_tri = np.array([gap_ratio(row[None, :]) for row in tridiagonal])
    result = {"dimension": d, "repeats": repeats, "rng_key": "(18, implementation, replicate)",
              "dense_gap_ratio": dense_ratio, "tridiagonal_gap_ratio": tri_ratio,
              "gap_ratio_difference": abs(dense_ratio - tri_ratio), "spectrum_ks": ks,
              "ratio_difference_se": float(np.sqrt(per_dense.var(ddof=1) / repeats + per_tri.var(ddof=1) / repeats)),
              "includes_original_ten": args.extended,
              "passed": abs(dense_ratio - tri_ratio) < 0.01 and ks < 0.05}
    np.savez(ROOT / f"outputs/diagnostics/{stem}_spectra.npz", dense=dense, tridiagonal=tridiagonal)
    json_write(path, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Dense/tridiagonal equivalence gate failed; inspect evidence without reseeding")


if __name__ == "__main__":
    main()
