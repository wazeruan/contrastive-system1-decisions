#!/usr/bin/env python3
"""Small standard-library helper for collision-safe runs and durable receipts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reserve_directory(base: Path) -> Path:
    base = base.expanduser().resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    candidates = [base, *(base.with_name(f"{base.name}-{index:03d}") for index in range(1, 10000))]
    for candidate in candidates:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not reserve a fresh directory below {base.parent}")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_init(args: argparse.Namespace) -> None:
    command = json.loads(args.command_json)
    value = {
        "submission_token": args.token,
        "stage": args.stage,
        "state": "SUBMITTING",
        "submitted_at_utc": args.submitted_at,
        "project_dir": args.project_dir,
        "run_dir": args.run_dir,
        "command": command,
        "log_out": args.log_out,
        "log_err": args.log_err,
        "job_id": None,
        "state_history": [{"state": "SUBMITTING", "at_utc": now_utc()}],
    }
    atomic_json(Path(args.path), value)


def command_update(args: argparse.Namespace) -> None:
    path = Path(args.path)
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = load(path)
        terminal_states = {"SUCCEEDED", "FAILED", "PREEMPTED"}
        if args.if_nonterminal and value.get("state") in terminal_states:
            if args.print_state:
                print(value["state"])
            return
        value["state"] = args.state
        value["updated_at_utc"] = now_utc()
        if args.job_id:
            value["job_id"] = args.job_id
            for key in ("log_out", "log_err"):
                if isinstance(value.get(key), str):
                    value[key] = value[key].replace("%j", args.job_id)
        if args.reason:
            value["reason"] = args.reason
        if args.host:
            value["compute_host"] = args.host
        if args.marker:
            value["application_start"] = load(Path(args.marker))
        if args.details_json:
            value.update(json.loads(args.details_json))
        history = value.setdefault("state_history", [])
        history.append({"state": args.state, "at_utc": now_utc(), "reason": args.reason or None})
        atomic_json(path, value)
        if args.print_state:
            print(value["state"])


def command_reserve(args: argparse.Namespace) -> None:
    print(reserve_directory(Path(args.base)))


def command_within(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    target = Path(args.path).expanduser().resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path must be inside the project directory: {target}")
    print(target)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for name in ("path", "token", "stage", "submitted-at", "project-dir", "run-dir", "command-json", "log-out", "log-err"):
        init.add_argument(f"--{name}", required=True)
    init.set_defaults(function=command_init)

    update = commands.add_parser("update")
    update.add_argument("--path", required=True)
    update.add_argument("--state", required=True)
    update.add_argument("--job-id")
    update.add_argument("--reason")
    update.add_argument("--host")
    update.add_argument("--marker")
    update.add_argument("--details-json")
    update.add_argument("--if-nonterminal", action="store_true")
    update.add_argument("--print-state", action="store_true")
    update.set_defaults(function=command_update)

    reserve = commands.add_parser("reserve")
    reserve.add_argument("--base", required=True)
    reserve.set_defaults(function=command_reserve)

    within = commands.add_parser("within")
    within.add_argument("--root", required=True)
    within.add_argument("--path", required=True)
    within.set_defaults(function=command_within)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.function(args)
    except Exception as error:  # noqa: BLE001 - report a concise CLI failure.
        print(f"receipt helper: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
