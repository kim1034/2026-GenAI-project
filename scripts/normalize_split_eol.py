"""Normalize ID-list storage to LF, preserving the exact old bytes and ID sequence."""

import argparse
import hashlib
import json

from scripts.run_experiment import ROOT, json_write


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "data/splits"
    existing_report = ROOT / "reports/split_eol_normalization.json"
    if args.apply and existing_report.exists():
        previous = json.loads(existing_report.read_text(encoding="utf-8"))
        if any(digest((folder / entry["file"]).read_bytes()) != entry["canonical_lf_sha256"] for entry in previous["entries"]):
            raise RuntimeError("Canonical splits changed after the recorded normalization")
        print(json.dumps(previous, indent=2))
        return
    checksums = (folder / "SHA256SUMS").read_bytes()
    recorded = {name: sha for sha, name in (line.split() for line in checksums.decode("ascii").splitlines())}
    preserved = ROOT / "outputs/reproducibility/split_eol_original"
    entries, canonical, seen = [], {}, set()
    for name, expected_count in (("dev_base", 50), ("dev_select", 200), ("dev_anal", 50), ("test", 1319)):
        path = folder / f"{name}.ids"
        original = path.read_bytes()
        ids = original.decode("ascii").splitlines()
        if len(ids) != expected_count or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise RuntimeError("ID counts or split disjointness failed")
        if any(not i.startswith("test-" if name == "test" else "train-") for i in ids):
            raise RuntimeError("Unexpected ID namespace")
        seen.update(ids)
        normalized = ("\n".join(ids) + "\n").encode("ascii")
        assert original.decode("ascii").splitlines() == normalized.decode("ascii").splitlines()
        canonical[path] = normalized
        entries.append({"file": path.name, "count": len(ids), "original_byte_sha256": digest(original),
                        "previous_recorded_sha256": recorded[path.name], "canonical_lf_sha256": digest(normalized),
                        "id_sequence_unchanged": True})
        if args.apply:
            preserved.mkdir(parents=True, exist_ok=True)
            backup = preserved / (path.name + "." + digest(original))
            if backup.exists() and backup.read_bytes() != original:
                raise RuntimeError("Original ID backup conflict")
            if not backup.exists():
                backup.write_bytes(original)
    if args.apply:
        backup = preserved / ("SHA256SUMS." + digest(checksums))
        if not backup.exists():
            backup.write_bytes(checksums)
        for path, content in canonical.items():
            path.write_bytes(content)
            assert digest(path.read_bytes()) == digest(content)
        (folder / "SHA256SUMS").write_bytes("".join(f'{entry["canonical_lf_sha256"]}  {entry["file"]}\n' for entry in entries).encode("ascii"))
        json_write(ROOT / "reports/split_eol_normalization.json", {"scope": "ID_storage_only", "entries": entries,
                   "raw_data_and_model_outputs_unchanged": True, "historical_manifests_unchanged": True})
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
