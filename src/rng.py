"""Independent, named NumPy random streams for reproducible experiment inputs."""

import numpy as np


def rng_for(*key: int, global_seed: int = 2026) -> np.random.Generator:
    if any(not isinstance(part, int) or part < 0 for part in key):
        raise ValueError("random stream keys must be nonnegative integers")
    return np.random.default_rng(np.random.SeedSequence(global_seed, spawn_key=key))
