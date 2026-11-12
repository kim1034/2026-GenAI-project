import copy

import pytest

from scripts.validate_gpu_v14 import canonical_hash, validate_checkpoint, validate_scope


def test_checkpoint_requires_same_condition_prompt_hash_and_complete_sixteen_rows():
    payload = {"tokens": [[42, 99]] * 16, "lengths": [2] * 16, "finishes": ["eos"] * 16, "prompt_sha256": "prompt"}
    check = {"kind": "sample_repeat", "condition": "A", "n_rows": 16, "id": "train-00001", "repeat": 0,
             "output_sha256": canonical_hash(payload)}
    kwargs = dict(condition="A", problem_id="train-00001", repeat=0, prompt_sha="prompt", eos_ids=[99])
    assert validate_checkpoint(payload, check, **kwargs) == check["output_sha256"]
    changed = copy.deepcopy(payload)
    changed["tokens"][0][0] = 41
    with pytest.raises(ValueError, match="hash"):
        validate_checkpoint(changed, check, **kwargs)
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(payload, check, **{**kwargs, "condition": "C"})
    with pytest.raises(ValueError, match="prompt"):
        validate_checkpoint(payload, check, **{**kwargs, "prompt_sha": "other"})
    bad = {**payload, "tokens": payload["tokens"][:15]}
    with pytest.raises(ValueError, match="16"):
        validate_checkpoint(bad, {**check, "output_sha256": canonical_hash(bad)}, **kwargs)


def test_scope_excludes_unused_32_rows_and_boost():
    protocol = {"protocol_amendment": "budget120-v1.4", "pilot": {"rows": 16}, "rehearsal": {"rows": 16},
                "test": {"rows_per_condition": 16, "boost": False}}
    validate_scope(protocol)
    with pytest.raises(ValueError):
        validate_scope({**protocol, "test": {"rows_per_condition": 32, "boost": False}})
