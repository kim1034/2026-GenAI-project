"""Local SFTP backup worker. Host keys are verified; changed versions are retained."""

import argparse
import hashlib
import json
import os
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import paramiko

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def backup(args, base):
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / ".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(args.host, username=args.user, key_filename=args.key, timeout=30,
                       banner_timeout=30, auth_timeout=30, allow_agent=False, look_for_keys=False)
        command = f"cd {shlex.quote(args.remote_root)} && .venv/bin/python -m scripts.export_inventory"
        if args.full_hash:
            command += " --full-hash"
        _, stdout, stderr = client.exec_command(command, timeout=180)
        payload = stdout.read()
        if stdout.channel.recv_exit_status():
            raise RuntimeError("Remote inventory failed: " + stderr.read().decode(errors="replace")[-2000:])
        inventory = json.loads(payload)
        state_path = base / "verified_files.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        transferred = deferred = 0
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(180)
            for item in inventory["files"]:
                rel = PurePosixPath(item["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError("Unsafe relative backup path")
                destination = base / "latest" / Path(*rel.parts)
                if not destination.resolve().is_relative_to((base / "latest").resolve()):
                    raise RuntimeError("Local backup path escaped destination")
                previous = state.get(item["path"])
                if previous == item["sha256"] and destination.exists() and not args.full_hash:
                    continue
                if destination.exists() and digest(destination) == item["sha256"]:
                    state[item["path"]] = item["sha256"]
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".download")
                sftp.get(args.remote_root.rstrip("/") + "/" + item["path"], str(temporary))
                if digest(temporary) != item["sha256"]:
                    temporary.unlink()
                    deferred += 1
                    continue  # Mutable log changed after inventory; retry next cycle.
                if destination.exists():
                    old_digest = digest(destination)
                    history = base / "versions" / Path(*rel.parts) / old_digest
                    history.parent.mkdir(parents=True, exist_ok=True)
                    if not history.exists():
                        destination.replace(history)
                os.replace(temporary, destination)
                state[item["path"]] = item["sha256"]
                transferred += 1
        save(state_path, state)
        save(base / "last_inventory.json", inventory)
        return {"status": "verified", "inventory_files": len(inventory["files"]), "transferred": transferred,
                "deferred_changed_files": deferred, "git_head": inventory["git_head"]}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--key", required=True)
    parser.add_argument("--remote-root", default="/home/ubuntu/2026-GenAI-project")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--hours", type=float, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    base = ROOT / "outputs/remote_backups/continuous"
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "worker.lock"
    with lock.open("x", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    deadline = time.monotonic() + args.hours * 3600
    try:
        while time.monotonic() < deadline and not (base / "STOP").exists():
            try:
                event = backup(args, base)
            except Exception as error:
                event = {"status": "error", "error": str(error)}
            event["at_utc"] = datetime.now(timezone.utc).isoformat()
            save(base / "status.json", event)
            with (base / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if args.once:
                if event["status"] == "error":
                    raise SystemExit(1)
                return
            for _ in range(max(1, args.interval)):
                if (base / "STOP").exists():
                    return
                time.sleep(1)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
