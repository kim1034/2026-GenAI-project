"""Build deterministic on-disk noise banks before model inference."""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from src.noise import coefficients, control_pool, direction, haar_orthogonal, uniform_base
from src.rng import rng_for


def _spectrum_direction_job(arguments):
    """Independent keyed task; scheduling cannot change a response's RNG stream."""
    condition, d, seed_set, split_key, original_index, response_index, seed = arguments
    return direction(coefficients(condition, d=d, seed_set=seed_set, split_key=split_key,
                                  original_index=original_index, response_index=response_index,
                                  global_seed=seed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bank_path(root: Path, condition: str, basis: str, seed_set: int, split: str,
              row_start: int, n_rows: int) -> Path:
    return root / "bank" / condition / basis / f"s{seed_set}" / f"{split}_rows{row_start}-{row_start + n_rows - 1}.npy"


def build_bank(
    root: Path,
    ids: list[str],
    config: dict,
    *,
    condition: str,
    split: str,
    seed_set: int,
    row_start: int,
    n_rows: int,
    workers: int = 1,
) -> tuple[Path, Path]:
    if condition in ("A", "Ap"):
        raise ValueError("temperature-only conditions have no noise bank")
    if split not in ("dev", "dev_base", "dev_select", "dev_anal", "test") or not ids or n_rows < 1 or row_start < 0:
        raise ValueError("invalid bank split, problem list or response row range")
    if len(ids) != len(set(ids)):
        raise ValueError("bank problem IDs must be unique")
    d = int(config["noise"]["d"])
    seed = int(config["global_seed"])
    basis = "B1" if condition in ("S_H", "T5_H") else "B0"
    source_condition = {"S_H": "S", "T5_H": "T5", "Bc": "B"}.get(condition, condition)
    if workers < 1:
        raise ValueError("workers must be positive")
    sparse_p = config["noise"]["conditions"]["S"].get("p") if source_condition == "S" else None
    ids_digest = hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()
    identity = {
        "condition": condition, "basis": basis, "seed_set": seed_set, "split": split,
        "row_start": row_start, "n_rows": n_rows, "dimension": d, "ids_sha256": ids_digest,
        "global_seed": seed, "sparse_p": sparse_p,
        "control_pool_size": config["noise"]["control_pool_size"] if source_condition == "E" else None,
    }
    path = bank_path(root, condition, basis, seed_set, split, row_start, n_rows)
    manifest_path = path.with_suffix(".json")
    if path.exists() or manifest_path.exists():
        if path.exists() and manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != value for key, value in identity.items()):
                raise RuntimeError("existing bank metadata differs from this request")
            load_bank(path, manifest_path, ids, row_start=row_start, n_rows=n_rows)
            return path, manifest_path
        raise RuntimeError("an incomplete final bank exists; inspect it before proceeding")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    state_path = path.with_suffix(".state.json")
    shape = (len(ids), n_rows, d)
    if partial.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("partial bank metadata differs from this request")
        array = np.lib.format.open_memmap(partial, mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise RuntimeError("partial bank has the wrong shape or dtype")
        start = state["next_problem"]
    else:
        if state_path.exists():
            raise RuntimeError("orphaned bank state exists; inspect it before proceeding")
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        start = 0
        _write_json(state_path, {**identity, "next_problem": start})

    pool = control_pool(d, config["noise"]["control_pool_size"], global_seed=seed) if source_condition == "E" else None
    rotation = haar_orthogonal(d, rng_for(2, 1, global_seed=seed)) if basis == "B1" else None
    split_key = 1 if split == "test" else 0
    support_partial = path.with_name(path.stem + "_support_count.partial.npy")
    support = None
    if source_condition == "S":
        if start and not support_partial.exists():
            raise RuntimeError("missing support-count evidence for partial sparse bank")
        support = np.lib.format.open_memmap(support_partial, mode="r+" if support_partial.exists() else "w+",
                                            dtype=np.int32, shape=(len(ids), n_rows))
    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=workers)) if workers > 1 and source_condition in ("D", "O") else None
        for position in range(start, len(ids)):
            problem_id = ids[position]
            prefix, number = problem_id.split("-", 1)
            if prefix != ("test" if split == "test" else "train"):
                raise ValueError("problem ID prefix does not match the bank split")
            original_index = int(number)
            if executor:
                tasks = [(source_condition, d, seed_set, split_key, original_index, row_start + k, seed) for k in range(n_rows)]
                array[position] = np.stack(list(executor.map(_spectrum_direction_job, tasks)))
            else:
                for response in range(n_rows):
                    k = row_start + response
                    if source_condition == "B":
                        base = uniform_base(d=d, seed_set=seed_set, split_key=split_key,
                                            original_index=original_index, response_index=k, global_seed=seed)
                        if condition == "Bc":
                            base = base - np.float32(0.5)
                    else:
                        coeff = coefficients(source_condition, d=d, seed_set=seed_set,
                                             split_key=split_key, original_index=original_index,
                                             response_index=k, global_seed=seed, pool=pool, sparse_p=sparse_p)
                        if support is not None:
                            support[position, response] = np.count_nonzero(coeff)
                        base = direction(coeff, rotation)
                    array[position, response] = base
            array.flush()
            if support is not None:
                support.flush()
            _write_json(state_path, {**identity, "next_problem": position + 1})
    del array
    os.replace(partial, path)
    state_path.unlink()
    manifest = {**identity, "problem_count": len(ids), "file_sha256": _sha256(path),
                "ids": ids, "dtype": "float32"}
    if support is not None:
        del support
        support_path = path.with_name(path.stem + "_support_count.npy")
        os.replace(support_partial, support_path)
        manifest.update(support_count_file=str(support_path.relative_to(root)), support_count_sha256=_sha256(support_path))
    _write_json(manifest_path, manifest)
    return path, manifest_path


def load_bank(path: Path, manifest_path: Path, ids: list[str], *, row_start: int,
              n_rows: int) -> np.memmap:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["ids"] != ids or manifest["row_start"] != row_start or manifest["n_rows"] != n_rows:
        raise ValueError("bank manifest does not match this run")
    if _sha256(path) != manifest["file_sha256"]:
        raise ValueError("bank SHA-256 does not match its manifest")
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.float32 or array.shape != (len(ids), n_rows, manifest["dimension"]):
        raise ValueError("bank shape or dtype does not match manifest")
    if manifest.get("support_count_sha256"):
        support_path = path.with_name(path.stem + "_support_count.npy")
        if _sha256(support_path) != manifest["support_count_sha256"]:
            raise ValueError("sparse support-count hash mismatch")
        support = np.load(support_path)
        if support.shape != (len(ids), n_rows) or not np.all(support == round(manifest["sparse_p"] * manifest["dimension"])):
            raise ValueError("sparse support count differs from selected sparsity")
    return array
