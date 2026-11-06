"""Precompute coefficient directions for the prespecified noise conditions."""

from math import sqrt

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from src.rng import rng_for


def beta_hermite_spectrum(d: int, rng: np.random.Generator, *, beta: int) -> np.ndarray:
    if d < 2 or beta not in (1, 2):
        raise ValueError("dimension must exceed one and beta must be GOE=1 or GUE=2")
    diagonal = rng.normal(0.0, sqrt(2.0), d)
    off_diagonal = np.sqrt(rng.chisquare(beta * np.arange(d - 1, 0, -1)))
    return eigvalsh_tridiagonal(diagonal, off_diagonal) / sqrt(2.0)


def control_pool(d: int, count: int, *, global_seed: int) -> np.ndarray:
    return np.stack(
        [beta_hermite_spectrum(d, rng_for(4, index, global_seed=global_seed), beta=2)
         for index in range(count)]
    )


def coefficients(
    condition: str,
    *,
    d: int,
    seed_set: int,
    split_key: int,
    original_index: int,
    response_index: int,
    global_seed: int,
    pool: np.ndarray | None = None,
    sparse_p: float | None = None,
) -> np.ndarray:
    keys = (seed_set, split_key, original_index, response_index)
    if condition == "C":
        return rng_for(6, *keys, global_seed=global_seed).standard_normal(d)
    if condition in ("D", "O"):
        generator = rng_for(3 if condition == "D" else 11, *keys, global_seed=global_seed)
        return generator.permutation(beta_hermite_spectrum(d, generator, beta=2 if condition == "D" else 1))
    if condition == "E":
        if pool is None or pool.ndim != 2 or pool.shape[1] != d:
            raise ValueError("E requires a control pool with the matching dimension")
        generator = rng_for(5, *keys, global_seed=global_seed)
        return pool[generator.integers(0, len(pool), d), generator.integers(0, d, d)]
    if condition == "L":
        return rng_for(13, *keys, global_seed=global_seed).laplace(0.0, 1 / sqrt(2), d)
    if condition == "T5":
        return rng_for(14, *keys, global_seed=global_seed).standard_t(5, d) * sqrt(3 / 5)
    if condition == "S":
        if sparse_p is None or not 0 < sparse_p <= 1:
            raise ValueError("S requires a preselected sparse probability")
        generator = rng_for(15, *keys, global_seed=global_seed)
        count = round(sparse_p * d)
        if count < 8:
            raise ValueError("the sparse condition requires at least eight active coordinates")
        values = np.zeros(d, dtype=np.float64)
        positions = generator.choice(d, count, replace=False)
        values[positions] = generator.standard_normal(count)
        return values
    raise ValueError(f"unknown relative noise condition: {condition}")


def uniform_base(*, d: int, seed_set: int, split_key: int, original_index: int,
                 response_index: int, global_seed: int) -> np.ndarray:
    return rng_for(7, seed_set, split_key, original_index, response_index,
                   global_seed=global_seed).random(d).astype(np.float32)


def direction(coefficients_array: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    centered = np.asarray(coefficients_array, dtype=np.float64)
    centered = centered - centered.mean()
    if rotation is not None:
        if rotation.shape != (len(centered), len(centered)):
            raise ValueError("rotation matrix has the wrong shape")
        centered = rotation @ centered
        centered = centered - centered.mean()
    length = np.linalg.norm(centered)
    if length == 0 or not np.isfinite(length):
        raise ValueError("noise direction is undefined")
    return (centered / length).astype(np.float32)


def haar_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((d, d))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs
