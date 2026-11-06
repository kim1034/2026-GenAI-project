import numpy as np
import pytest

from src.noise import beta_hermite_spectrum, coefficients, direction, haar_orthogonal, uniform_base
from src.bank import build_bank, load_bank
from src.rng import rng_for
from src.sampling import sample_from_logits


def test_inverse_cdf_skips_zero_probability_and_greedy_ties():
    logits = np.array([[-1000.0, np.log(0.2), -1000.0, np.log(0.5), np.log(0.3)]])
    assert sample_from_logits(logits, np.array([0.0]), 1.0).tolist() == [1]
    assert sample_from_logits(logits, np.array([0.21]), 1.0).tolist() == [3]
    assert sample_from_logits(logits, np.array([0.8]), 1.0).tolist() == [4]
    assert sample_from_logits(np.array([[1.0, 1.0, 0.0]]), np.array([0.9]), 0).tolist() == [0]


def test_noise_keys_and_direction_are_deterministic():
    args = dict(d=64, seed_set=0, split_key=0, original_index=42, response_index=3, global_seed=2026)
    first = coefficients("C", **args)
    assert np.array_equal(first, coefficients("C", **args))
    vector = direction(first)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert abs(vector.mean()) < 1e-7
    assert np.array_equal(uniform_base(**args), uniform_base(**args))


def test_gue_goe_and_rotation_small_dimension():
    gue = beta_hermite_spectrum(32, rng_for(3, 0, global_seed=2026), beta=2)
    goe = beta_hermite_spectrum(32, rng_for(11, 0, global_seed=2026), beta=1)
    assert len(gue) == len(goe) == 32
    assert np.all(np.diff(gue) > 0)
    assert not np.array_equal(gue, goe)
    rotation = haar_orthogonal(32, rng_for(2, 1, global_seed=2026))
    assert np.allclose(rotation.T @ rotation, np.eye(32), atol=1e-12)
    rotated = direction(gue, rotation)
    assert np.linalg.norm(rotated) == pytest.approx(1, abs=1e-6)
    assert abs(rotated.mean()) < 1e-7


def test_bank_is_saved_and_checked_before_use(tmp_path):
    config = {"global_seed": 2026, "noise": {"d": 16, "control_pool_size": 2,
              "conditions": {"S": {"p": 0.5}}}}
    ids = ["train-00001", "train-00042"]
    base_path, base_manifest = build_bank(tmp_path, ids, config, condition="B", split="dev",
                                          seed_set=0, row_start=0, n_rows=2)
    centered_path, centered_manifest = build_bank(tmp_path, ids, config, condition="Bc", split="dev",
                                                  seed_set=0, row_start=0, n_rows=2)
    base = load_bank(base_path, base_manifest, ids, row_start=0, n_rows=2)
    centered = load_bank(centered_path, centered_manifest, ids, row_start=0, n_rows=2)
    assert base.shape == (2, 2, 16)
    assert np.allclose(centered, base - 0.5, atol=1e-7)
    with pytest.raises(ValueError):
        load_bank(base_path, base_manifest, ids[::-1], row_start=0, n_rows=2)
