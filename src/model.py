"""Load the pinned Qwen model on one CUDA device for deterministic evaluation."""

import hashlib
import json
from pathlib import Path
import re


TOKENIZER_PROVENANCE_ATTRIBUTE = "_genai_verified_tokenizer_provenance"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json", "tokenizer.model", "chat_template.jinja", "config.json")


def _file_hashes(snapshot):
    return {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in TOKENIZER_FILES if (snapshot / name).is_file()}


def load_pinned_tokenizer(config: dict):
    """Load only an exact cached snapshot and verify its full token-to-ID map.

    This CPU-only function establishes provenance from the files actually read;
    a missing Hugging Face ``_commit_hash`` is never filled by assumption.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    revision = config["model"]["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Tokenizer requires an exact full 40-character revision")
    snapshot = Path(snapshot_download(repo_id=config["model"]["id"], revision=revision,
                                      local_files_only=True)).resolve()
    if snapshot.name != revision or snapshot.parent.name != "snapshots":
        raise ValueError("Cached tokenizer snapshot directory does not match the pinned revision")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Pinned tokenizer snapshot is missing {name}")
    hashes = _file_hashes(snapshot)
    serialized = json.loads((snapshot / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = serialized.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Expected the pinned tokenizer.json model vocabulary mapping")
    expected = dict(vocab)
    for item in serialized.get("added_tokens", []):
        token, index = item["content"], item["id"]
        if token in expected and expected[token] != index:
            raise ValueError("Conflicting added-token ID in tokenizer snapshot")
        expected[token] = index
    if (any(not isinstance(token, str) or type(index) is not int or index < 0 for token, index in expected.items())
            or len(set(expected.values())) != len(expected)):
        raise ValueError("Tokenizer snapshot has invalid or duplicate token IDs")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tokenizer.get_vocab() != expected:
        raise ValueError("Loaded tokenizer vocabulary/added tokens differ from the pinned snapshot")
    if _file_hashes(snapshot) != hashes:
        raise ValueError("Tokenizer snapshot changed while being loaded")
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    setattr(tokenizer, TOKENIZER_PROVENANCE_ATTRIBUTE, {
        "verified": True, "revision": revision, "repo_id": config["model"]["id"],
        "snapshot_path": str(snapshot), "files_sha256": hashes,
        "vocab_sha256": hashlib.sha256(canonical).hexdigest(), "vocab_size": len(expected),
    })
    return tokenizer


def load_model(config: dict):
    revision = config["model"]["revision"]
    if not revision:
        raise ValueError("model.revision must be pinned before loading the model")
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("this experiment requires a CUDA GPU")
    if config["model"]["dtype"] != "float32" or config["model"]["attn_implementation"] != "eager":
        raise ValueError("v1.2 requires float32 and eager attention")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = load_pinned_tokenizer(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"], revision=revision,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda")
    model.eval()
    expected = config["model"]["expect"]
    for name in ("num_hidden_layers", "intermediate_size", "hidden_act"):
        if getattr(model.config, name) != expected[name]:
            raise ValueError(f"unexpected model {name}: {getattr(model.config, name)}")
    return model, tokenizer
