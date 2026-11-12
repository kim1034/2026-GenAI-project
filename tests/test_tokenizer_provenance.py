import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.run_experiment import tokenizer_execution_metadata
from src.model import TOKENIZER_PROVENANCE_ATTRIBUTE, load_pinned_tokenizer


REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@pytest.fixture
def cached_tokenizer(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    serialized = {"model": {"vocab": {"a": 0, "b": 1}},
                  "added_tokens": [{"content": "<eos>", "id": 2}]}
    (snapshot / "tokenizer.json").write_text(json.dumps(serialized), encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "merges.txt").write_text("a b\n", encoding="utf-8")
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": None}, get_vocab=lambda: {"a": 0, "b": 1, "<eos>": 2})
    observed = {}
    def download(**kwargs):
        observed["download"] = kwargs
        return str(snapshot)
    def load(path, **kwargs):
        observed["load"] = (path, kwargs)
        return tokenizer
    factory = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=factory))
    config = {"model": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": REVISION}}
    return config, snapshot, tokenizer, observed, factory


def test_cached_full_revision_and_real_vocab_establish_provenance(cached_tokenizer):
    config, snapshot, tokenizer, observed, _ = cached_tokenizer
    assert load_pinned_tokenizer(config) is tokenizer
    assert observed["download"] == {"repo_id": config["model"]["id"], "revision": REVISION, "local_files_only": True}
    assert observed["load"] == (str(snapshot), {"local_files_only": True})
    provenance = getattr(tokenizer, TOKENIZER_PROVENANCE_ATTRIBUTE)
    assert provenance["vocab_size"] == 3
    canonical = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert provenance["vocab_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert provenance["files_sha256"]["tokenizer.json"] == hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    metadata = tokenizer_execution_metadata(tokenizer)
    assert metadata["tokenizer_revision_loaded"] == REVISION
    assert metadata["tokenizer_hf_commit_hash"] is None
    assert metadata["tokenizer_snapshot_files_sha256"] == provenance["files_sha256"]
    assert metadata["tokenizer_vocab_sha256"] == provenance["vocab_sha256"]


@pytest.mark.parametrize("revision", [None, "main", REVISION[:12]])
def test_unpinned_revision_is_rejected_before_cache_lookup(cached_tokenizer, revision):
    config, _, _, observed, _ = cached_tokenizer
    config["model"]["revision"] = revision
    with pytest.raises(ValueError, match="exact full"):
        load_pinned_tokenizer(config)
    assert not observed


def test_cache_path_must_identify_the_exact_revision(cached_tokenizer, monkeypatch):
    config, snapshot, _, observed, _ = cached_tokenizer
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot.parent)))
    with pytest.raises(ValueError, match="snapshot directory"):
        load_pinned_tokenizer(config)
    assert "load" not in observed


def test_vocab_drift_is_not_certified(cached_tokenizer):
    config, _, tokenizer, _, _ = cached_tokenizer
    tokenizer.get_vocab = lambda: {"a": 0, "b": 2, "<eos>": 1}
    with pytest.raises(ValueError, match="vocabulary/added tokens"):
        load_pinned_tokenizer(config)
    assert not hasattr(tokenizer, TOKENIZER_PROVENANCE_ATTRIBUTE)


def test_changing_file_during_load_is_rejected(cached_tokenizer):
    config, snapshot, tokenizer, _, factory = cached_tokenizer
    def mutate(path, **kwargs):
        (snapshot / "merges.txt").write_text("changed\n", encoding="utf-8")
        return tokenizer
    factory.from_pretrained = mutate
    with pytest.raises(ValueError, match="changed while"):
        load_pinned_tokenizer(config)
    assert not hasattr(tokenizer, TOKENIZER_PROVENANCE_ATTRIBUTE)


def test_metadata_does_not_infer_revision_without_verified_file_provenance():
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": REVISION})
    metadata = tokenizer_execution_metadata(tokenizer)
    assert metadata["tokenizer_hf_commit_hash"] == REVISION
    assert metadata["tokenizer_revision_loaded"] is None
    assert metadata["tokenizer_vocab_sha256"] is None
    assert metadata["tokenizer_snapshot_files_sha256"] is None
