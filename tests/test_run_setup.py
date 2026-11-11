from pathlib import Path

import numpy as np
import pytest

from scripts.run_experiment import _uniforms, _validate_run, run_id
from src.config import load_config, verify_data_source
from src.data import load_split


def test_run_id_and_common_random_numbers():
    config = load_config(Path("configs/base.yaml"))
    run = {"split": "dev_base", "condition": "C", "alpha": 0.1, "temperature": 0.5,
           "basis": "B0", "seed_set": 0, "row_start": 0, "n_rows": 16}
    _validate_run(run, config)
    assert run_id(run) == "dev_base_C_a0.1000_T0.50_B0_s0_k16"
    first = _uniforms(config, run, "train-00042")
    same = _uniforms(config, {**run, "condition": "D"}, "train-00042")
    assert first.shape == (16, 512)
    assert np.array_equal(first, same)
    assert not np.array_equal(first[0], first[1])


def test_new_id_version_distinguishes_exact_magnitudes_without_changing_legacy_ids():
    first = {"split": "dev_base", "condition": "C", "alpha": .00625, "temperature": .5,
             "basis": "B0", "seed_set": 0, "row_start": 0, "n_rows": 16}
    second = {**first, "alpha": .00626}
    assert run_id(first) == run_id(second)
    assert run_id({**first, "id_version": 2}) != run_id({**second, "id_version": 2})
    assert run_id({**first, "id_version": 2}).startswith(run_id(first) + "_v2_")
    assert run_id({**first, "id_version": 2, "base_config": "another_snapshot.yaml"}) == run_id({**first, "id_version": 2})
    with pytest.raises(ValueError, match="unsupported"):
        run_id({**first, "id_version": 3})


def test_invalid_test_row_shape_is_rejected_before_gpu():
    config = load_config(Path("configs/base.yaml"))
    run = {"split": "test", "condition": "D", "alpha": 0.1, "temperature": 0.5,
           "basis": "B0", "seed_set": 0, "row_start": 0, "n_rows": 16}
    with pytest.raises(ValueError):
        _validate_run(run, config)


def test_pinned_openai_source_is_verified_before_test():
    config = load_config(Path("configs/base.yaml"))
    verify_data_source(config, Path("."))
    wrong_revision = {**config, "data": {**config["data"], "revision": "0" * 40}}
    with pytest.raises(ValueError, match="configured data source"):
        verify_data_source(wrong_revision, Path("."))


def test_test_split_remains_sealed_before_lock():
    config = load_config(Path("configs/base.yaml"))
    with pytest.raises(PermissionError, match="LOCK.md"):
        load_split(Path("."), "test", config)
