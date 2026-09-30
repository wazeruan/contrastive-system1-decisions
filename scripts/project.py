#!/usr/bin/env python3
"""Create, inspect, and summarize durable pipeline runs (stdlib only)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING",
          "SIGNALING", "STAGE_OUT", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "REVOKED",
          "RESV_DEL_HOLD", "SPECIAL_EXIT", "STOPPED", "UPDATE_DB", "EXPEDITING",
          "POWER_UP_NODE", "STARTED", "SUBMITTED", "RUNNING_STARTUP_UNVERIFIED"}
FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "OUT_OF_MEMORY", "NODE_FAIL",
          "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT", "LAUNCH_FAILED",
          "SUBMISSION_REJECTED", "SUBMISSION_STOPPED"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mutate(path: Path, change: Any) -> dict[str, Any]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        value["updated_at_utc"] = now()
        atomic_write(path, value)
    return value


def pipeline_path(project: Path, requested: str | None) -> Path:
    root = project / "runs" / "pipelines"
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_dir():
            candidate = candidate / "pipeline.json"
        elif not candidate.is_absolute() and not candidate.suffix:
            candidate = root / requested / "pipeline.json"
        elif candidate.is_dir():
            candidate = candidate / "pipeline.json"
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"pipeline not found: {requested}")
    runs = sorted(root.glob("*/pipeline.json"), key=lambda p: p.parent.name)
    if not runs:
        raise FileNotFoundError(f"no pipelines found under {root}")
    return runs[-1].resolve()


def command_init(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).expanduser().resolve()
    base = project / "runs" / "pipelines"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pipeline_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    directory = base / pipeline_id
    directory.mkdir()
    source_config = Path(args.config).expanduser().resolve()
    config_bytes = source_config.read_bytes()
    config_value = json.loads(config_bytes)
    config_snapshot = directory / "config.json"
    temporary_config = config_snapshot.with_name(f".{config_snapshot.name}.{os.getpid()}.tmp")
    temporary_config.write_bytes(config_bytes)
    os.replace(temporary_config, config_snapshot)
    resource_profile_path: str | None = None
    resource_profile_name: str | None = None
    resource_profile_sha256: str | None = None
    if args.resource_profile:
        source = Path(args.resource_profile).expanduser().resolve()
        profile_bytes = source.read_bytes()
        profile = json.loads(profile_bytes)
        snapshot = directory / "resource-profile.json"
        temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
        temporary.write_bytes(profile_bytes)
        os.replace(temporary, snapshot)
        resource_profile_path = str(snapshot.resolve())
        resource_profile_name = profile.get("name") if isinstance(profile, dict) else None
        resource_profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "pipeline_id": pipeline_id,
        "project_dir": str(project),
        "account": args.account,
        "config": str(source_config),
        "config_snapshot": str(config_snapshot.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "configured_precision": "BF16 autocast requested" if config_value.get("use_bf16") else "FP32",
        "host_memory": args.memory,
        "resource_profile": resource_profile_path,
        "resource_profile_name": resource_profile_name,
        "resource_profile_sha256": resource_profile_sha256,
        "created_at_utc": now(),
        "updated_at_utc": now(),
        "submission_state": "SUBMITTING",
        "stages": [],
    }
    atomic_write(directory / "pipeline.json", manifest)
    print(f"PIPELINE_ID={pipeline_id}")
    print(f"PIPELINE_FILE={directory / 'pipeline.json'}")
    print(f"PIPELINE_DIR={directory}")
    print(f"CONFIG_SNAPSHOT={config_snapshot.resolve()}")
    if resource_profile_path:
        print(f"RESOURCE_PROFILE={resource_profile_path}")


def receipt_data(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(Path(raw).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def find_receipt(project: Path, job_id: str) -> Path | None:
    for path in (project / "logs" / "submissions").glob("*/receipt.json"):
        data = receipt_data(str(path))
        if str(data.get("job_id")) == job_id:
            return path.resolve()
    return None


def command_register(args: argparse.Namespace) -> None:
    path = Path(args.manifest).resolve()
    receipt = Path(args.receipt).resolve() if args.receipt else None
    if receipt is None and args.job_id and not args.no_receipt_lookup:
        receipt = find_receipt(Path(args.project_dir).resolve(), args.job_id)
    receipt_value = receipt_data(str(receipt) if receipt else None)
    stage: dict[str, Any] = {
        "name": args.stage,
        "job_id": args.job_id or receipt_value.get("job_id"),
        "run_dir": args.run_dir or receipt_value.get("run_dir"),
        "receipt": str(receipt) if receipt else None,
        "stdout": args.stdout or receipt_value.get("log_out"),
        "stderr": args.stderr or receipt_value.get("log_err"),
        "data_path": args.data_path,
        "state": args.state or receipt_value.get("state") or "UNKNOWN/UNVERIFIED",
        "reason": receipt_value.get("reason"),
        "reused": bool(args.reused),
    }

    def change(value: dict[str, Any]) -> None:
        stages = value.setdefault("stages", [])
        # A stage may be registered once as reused and later as newly submitted;
        # update the existing stage in place without losing its earlier evidence.
        old = next((item for item in stages if item.get("name") == args.stage and
                    str(item.get("job_id") or "") == str(stage.get("job_id") or "")), None)
        if old is None:
            stages.append(stage)
        else:
            old.update({key: val for key, val in stage.items() if val is not None})

    mutate(path, change)


def command_finish(args: argparse.Namespace) -> None:
    path = Path(args.manifest).resolve()

    def change(value: dict[str, Any]) -> None:
        value["submission_state"] = "QUEUED" if args.exit_code == 0 else "SUBMISSION_STOPPED"
        value["submission_exit_code"] = args.exit_code
        if args.last_stage:
            value["last_stage"] = args.last_stage

    mutate(path, change)


def scheduler_state(job_id: str, refresh: bool) -> tuple[str, str | None]:
    if not refresh:
        return "", None
    queue_problem: str | None = None
    try:
        queue = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T|%R"],
                               text=True, capture_output=True, timeout=15)
        output = (queue.stdout + queue.stderr).strip()
        if "Invalid job id specified" not in output and queue.returncode == 0:
            if not queue.stdout.strip():
                pass
            else:
                line = queue.stdout.strip().splitlines()[0]
                if "|" in line:
                    state, reason = line.split("|", 1)
                    state = state.strip().upper().split()[0]
                    if state in ACTIVE:
                        return state, reason.strip() or None
                    if state in FAILED:
                        return state, reason.strip() or None
                    if state != "COMPLETED":
                        return "UNKNOWN/UNVERIFIED", f"unrecognized squeue state: {state}"
                else:
                    return "UNKNOWN/UNVERIFIED", f"unrecognized squeue response: {line}"
        elif queue.returncode != 0 and "Invalid job id specified" not in output:
            queue_problem = output
        else:
            queue_problem = None
    except (OSError, subprocess.TimeoutExpired) as error:
        queue_problem = str(error)

    try:
        accounting = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"],
            text=True, capture_output=True, timeout=15,
        )
        if accounting.returncode == 0:
            lines = [line.strip() for line in accounting.stdout.splitlines() if line.strip()]
            if lines:
                state, separator, code = lines[0].partition("|")
                state = state.split("+", 1)[0].strip().upper().split()[0]
                code = code.split("|", 1)[0].strip() if separator else ""
                if state == "COMPLETED" and code == "0:0":
                    return "SUCCEEDED", None
                if state == "COMPLETED" and code != "0:0":
                    return "FAILED", f"COMPLETED with exit {code or 'unknown'}"
                if state in FAILED or state == "COMPLETED":
                    return state, f"exit {code}" if code else None
                if state in ACTIVE:
                    return state, None
        problem = (accounting.stdout + accounting.stderr).strip() or "no sacct record"
    except (OSError, subprocess.TimeoutExpired) as error:
        problem = str(error)
    return "", f"scheduler unavailable: {queue_problem or problem}"


def current_stage(stage: dict[str, Any], refresh: bool) -> dict[str, Any]:
    result = dict(stage)
    receipt = receipt_data(stage.get("receipt"))
    if receipt:
        result["state"] = receipt.get("state", result.get("state", "UNKNOWN/UNVERIFIED"))
        if "reason" in receipt:
            result["reason"] = receipt.get("reason")
        result["run_dir"] = receipt.get("run_dir") or result.get("run_dir")
        result["stdout"] = receipt.get("log_out") or result.get("stdout")
        result["stderr"] = receipt.get("log_err") or result.get("stderr")
    job_id = str(result.get("job_id") or "")
    if job_id and refresh:
        state, reason = scheduler_state(job_id, True)
        if state:
            result["state"] = state
            result["reason"] = reason
        elif reason:
            result["previous_state"] = result.get("state")
            result["state"] = "UNKNOWN/UNVERIFIED"
            result["scheduler_note"] = reason
    return result


def pipeline_state(manifest: dict[str, Any], stages: list[dict[str, Any]]) -> str:
    states = [str(stage.get("state", "UNKNOWN/UNVERIFIED")).upper() for stage in stages]
    if states and all(state == "SUCCEEDED" for state in states):
        return "SUCCEEDED"
    if any(state in FAILED for state in states):
        return "FAILED"
    if manifest.get("submission_state") == "SUBMISSION_STOPPED":
        return "SUBMISSION_STOPPED"
    if any(state == "RUNNING" for state in states):
        return "RUNNING"
    if any(state == "PENDING" for state in states):
        return "PENDING"
    if any(state in ACTIVE for state in states):
        return "ACTIVE"
    if any("UNKNOWN" in state for state in states):
        return "UNKNOWN/UNVERIFIED"
    return str(manifest.get("submission_state", "UNKNOWN"))


def print_status(manifest: dict[str, Any], stages: list[dict[str, Any]]) -> None:
    current = pipeline_state(manifest, stages)
    print(f"Pipeline: {manifest['pipeline_id']}  [{current}]")
    print(f"Created:  {manifest.get('created_at_utc', 'unknown')}  Account: {manifest.get('account', 'unknown')}")
    if manifest.get("resource_profile_name"):
        print(f"GPU:      {manifest['resource_profile_name']} ({manifest.get('resource_profile')})")
    if manifest.get("host_memory"):
        print(f"Host RAM: {manifest['host_memory']} per stage (GPU VRAM is separate)")
    if manifest.get("configured_precision"):
        print(f"Configured precision: {manifest['configured_precision']}")
    if manifest.get("config_snapshot"):
        print(f"Config:   {manifest['config_snapshot']} (sha256 {str(manifest.get('config_sha256', ''))[:12]})")
    print(f"Run:      {Path(manifest['_path']).parent}")
    print("")
    print(f"{'STAGE':<18} {'STATE':<28} {'JOB':<12} OUTPUT / DETAILS")
    for stage in stages:
        name = str(stage.get("name", "?"))
        state = str(stage.get("state", "UNKNOWN/UNVERIFIED"))
        job = str(stage.get("job_id") or "—")
        detail = " | ".join(str(stage[key]) for key in ("run_dir", "reason", "scheduler_note")
                            if stage.get(key))
        print(f"{name:<18} {state:<28} {job:<12} {detail}")
    print(f"\nDetails: {Path(manifest['_path'])}")


def command_status(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).expanduser().resolve()
    path = pipeline_path(project, args.pipeline_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["_path"] = str(path)
    stages = [current_stage(stage, not args.no_refresh) for stage in manifest.get("stages", [])]
    print_status(manifest, stages)
    if args.json:
        print(json.dumps({"pipeline": manifest, "stages": stages}, indent=2))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metric_line(path: Path) -> str:
    value = load_json(path)
    if not isinstance(value, dict):
        return ""
    keys = ("top1_tool_accuracy", "mrr", "nll", "brier", "ece_15_bins", "temperature",
            "forward_precision", "precision",
            "nll_before", "nll_after", "examples", "best_validation_nll", "global_step")
    return ", ".join(f"{key}={value[key]}" for key in keys if key in value)


def command_results(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).expanduser().resolve()
    path = pipeline_path(project, args.pipeline_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stages = [current_stage(stage, not args.no_refresh) for stage in manifest.get("stages", [])]
    manifest["current_state"] = pipeline_state(manifest, stages)
    lines = [f"# Pipeline results: {manifest['pipeline_id']}", "",
             f"- **Created:** {manifest.get('created_at_utc', 'unknown')}",
             f"- **Account:** {manifest.get('account', 'unknown')}",
             f"- **Host RAM:** {manifest.get('host_memory', 'unknown')} per stage",
             f"- **Configured precision:** {manifest.get('configured_precision', 'unknown')}",
             f"- **Config:** `{manifest.get('config_snapshot') or manifest.get('config') or 'unknown'}`",
             f"- **Config SHA-256:** `{manifest.get('config_sha256', 'unknown')}`",
             f"- **GPU profile:** {manifest.get('resource_profile_name') or manifest.get('resource_profile') or 'unknown'}",
             f"- **Current state:** {manifest['current_state']}", "",
             "## Stage status", "", "| Stage | State | Job ID | Output |", "|---|---|---:|---|"]
    for stage in stages:
        output = str(stage.get("run_dir") or "")
        lines.append(f"| {stage.get('name', '')} | {stage.get('state', '')} | "
                     f"{stage.get('job_id') or ''} | `{output}` |")
    lines.extend(["", "## Metrics and artifacts", ""])
    artifacts: list[tuple[str, Path, str]] = []
    for stage in stages:
        run_dir = Path(str(stage.get("run_dir") or "")) if stage.get("run_dir") else None
        if run_dir is None:
            continue
        stage_state = str(stage.get("state", "UNKNOWN/UNVERIFIED")).upper()
        if stage.get("name") == "train":
            for rel in ("status.json", "checkpoints/best.json", "checkpoints/best.pt", "history.jsonl"):
                candidate = run_dir / rel
                if candidate.is_file():
                    artifacts.append((f"Training {rel}", candidate, stage_state))
        elif stage.get("name") in {"calibrate", "evaluate", "evaluate-xlam", "benchmark", "evaluate-bfcl"}:
            for rel in ("calibration.json", "metrics.json"):
                candidate = run_dir / rel
                if candidate.is_file():
                    artifacts.append((f"{stage.get('name')} {rel}", candidate, stage_state))
    if not artifacts:
        lines.append("No result artifacts are present yet; this is normal while jobs are queued or running.")
    for label, artifact, stage_state in artifacts:
        detail = metric_line(artifact) if artifact.suffix == ".json" else ""
        if artifact.name == "history.jsonl":
            records = []
            for row in artifact.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(row))
                except ValueError:
                    continue
            if records:
                last = records[-1]
                validation = last.get("validation", {})
                detail = (f"epochs={len(records)}, last_validation_accuracy={validation.get('accuracy')}, "
                          f"last_validation_nll={validation.get('nll')}, global_step={last.get('global_step')}")
        provisional = "" if stage_state == "SUCCEEDED" else f" [PROVISIONAL: stage {stage_state}]"
        lines.append(f"- **{label}{provisional}:** `{artifact}`" + (f" — {detail}" if detail else ""))
    lines.extend(["", f"Manifest: `{path}`", ""])
    output = path.parent / "RESULTS.md"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output)
    print("\n".join(lines))
    print(f"Saved: {output}")


def command_list(args: argparse.Namespace) -> None:
    root = Path(args.project_dir).expanduser().resolve() / "runs" / "pipelines"
    manifests = sorted(root.glob("*/pipeline.json"), key=lambda p: p.parent.name, reverse=True)
    if not manifests:
        print(f"No pipeline runs in {root}")
        return
    print(f"{'PIPELINE':<30} {'SUBMISSION':<20} {'CREATED':<22} ACCOUNT")
    for path in manifests:
        value = load_json(path) or {}
        print(f"{value.get('pipeline_id', path.parent.name):<30} "
              f"{value.get('submission_state', 'UNKNOWN'):<20} "
              f"{value.get('created_at_utc', ''):<22} {value.get('account', '')}")


def command_check_resume(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"selected config cannot be read: {config_path}")

    model_receipts = []
    for path in (project / "logs" / "submissions").glob("*/receipt.json"):
        value = receipt_data(str(path))
        if str(value.get("job_id")) == args.model_job_id:
            model_receipts.append((path, value))
    if len(model_receipts) != 1:
        raise ValueError(
            f"model job {args.model_job_id} has {len(model_receipts)} matching receipts; "
            "cannot prove its model revision"
        )
    _receipt_path, receipt = model_receipts[0]
    if receipt.get("stage") != "prepare-model":
        raise ValueError(f"job {args.model_job_id} is not a prepare-model submission")
    run_dir = Path(str(receipt.get("run_dir") or ""))
    snapshot = run_dir / "config.json" if receipt.get("run_dir") else None
    if snapshot is not None and snapshot.is_file():
        model_config = load_json(snapshot)
        model_source = snapshot
    else:
        raw_config = receipt.get("command", {}).get("arguments", {}).get("config")
        if not isinstance(raw_config, str) or not raw_config:
            raise ValueError(f"model job {args.model_job_id} receipt has no config path")
        model_source = Path(raw_config).expanduser()
        if not model_source.is_absolute():
            model_source = project / model_source
        model_config = load_json(model_source)
    if not isinstance(model_config, dict):
        raise ValueError(f"model job config cannot be read: {model_source}")
    for key in ("model_name", "model_revision"):
        if model_config.get(key) != config.get(key):
            raise ValueError(
                f"selected config {key} does not match model job {args.model_job_id}; "
                "prepare the matching model revision before resuming"
            )

    related_train_jobs: set[str] = set()
    for path in (project / "runs" / "pipelines").glob("*/pipeline.json"):
        value = load_json(path)
        if not isinstance(value, dict):
            continue
        stages = value.get("stages", [])
        has_preflight = any(
            stage.get("name") == "preflight" and str(stage.get("job_id") or "") == args.preflight_job_id
            for stage in stages if isinstance(stage, dict)
        )
        if not has_preflight:
            continue
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("name") != "train":
                continue
            job_id = str(stage.get("job_id") or "")
            if job_id:
                related_train_jobs.add(job_id)
            elif str(stage.get("state", "")).upper() != "SUBMISSION_REJECTED":
                raise ValueError(f"pipeline {path.parent.name} has a train stage with no verifiable job ID")

    for path in (project / "logs" / "submissions").glob("train-*/receipt.json"):
        value = receipt_data(str(path))
        command = value.get("command", {})
        dependencies = command.get("dependency_resolution", {}).get("requested_afterok_job_id")
        if not dependencies:
            dependencies = command.get("sbatch", {}).get("afterok_job_id")
        dependency_ids = {part for part in str(dependencies or "").split(",") if part}
        if args.preflight_job_id not in dependency_ids:
            continue
        job_id = str(value.get("job_id") or "")
        if job_id:
            related_train_jobs.add(job_id)
        elif str(value.get("state", "")).upper() != "SUBMISSION_REJECTED":
            raise ValueError(f"train receipt {path} references this preflight but has no job ID")

    for job_id in sorted(related_train_jobs, key=int):
        state, reason = scheduler_state(job_id, True)
        if state in ACTIVE:
            raise ValueError(
                f"earlier train job {job_id} linked to preflight {args.preflight_job_id} is {state}"
                + (f" ({reason})" if reason else "")
                + "; wait for it to finish before starting another train"
            )
        if state == "SUCCEEDED":
            raise ValueError(
                f"train job {job_id} linked to preflight {args.preflight_job_id} already succeeded; "
                "reuse its checkpoint/results instead of submitting a duplicate train"
            )
        if state not in FAILED:
            raise ValueError(
                f"earlier train job {job_id} linked to preflight {args.preflight_job_id} is "
                f"{state or 'UNKNOWN/UNVERIFIED'}; do not resubmit until its state is verified"
            )
        print(f"Verified earlier train job {job_id} is terminal ({state}); a retry may proceed.")

    print(f"Verified model job {args.model_job_id} matches selected model name/revision.")
    print(f"No pending, running, successful, or unverifiable train job is linked to preflight {args.preflight_job_id}.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--project-dir", required=True)
    init.add_argument("--account", required=True)
    init.add_argument("--config", required=True)
    init.add_argument("--memory", required=True)
    init.add_argument("--resource-profile")
    init.set_defaults(function=command_init)

    register = commands.add_parser("register")
    register.add_argument("--manifest", required=True)
    register.add_argument("--project-dir", required=True)
    register.add_argument("--stage", required=True)
    register.add_argument("--job-id")
    register.add_argument("--run-dir")
    register.add_argument("--receipt")
    register.add_argument("--stdout")
    register.add_argument("--stderr")
    register.add_argument("--state")
    register.add_argument("--data-path")
    register.add_argument("--reused", action="store_true")
    register.add_argument("--no-receipt-lookup", action="store_true")
    register.set_defaults(function=command_register)

    finish = commands.add_parser("finish")
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--exit-code", required=True, type=int)
    finish.add_argument("--last-stage")
    finish.set_defaults(function=command_finish)

    check_resume = commands.add_parser("check-resume")
    check_resume.add_argument("--project-dir", required=True)
    check_resume.add_argument("--config", required=True)
    check_resume.add_argument("--model-job-id", required=True)
    check_resume.add_argument("--preflight-job-id", required=True)
    check_resume.set_defaults(function=command_check_resume)

    for name, function in (("status", command_status), ("results", command_results), ("list", command_list)):
        item = commands.add_parser(name)
        item.add_argument("--project-dir", default=str(Path(__file__).resolve().parent.parent))
        if name != "list":
            item.add_argument("pipeline_id", nargs="?")
            item.add_argument("--no-refresh", action="store_true")
        if name == "status":
            item.add_argument("--json", action="store_true")
        item.set_defaults(function=function)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.function(args)
    except Exception as error:  # concise CLI diagnostics
        print(f"project status: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
