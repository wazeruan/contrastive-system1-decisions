#!/usr/bin/env python3
"""Lightweight login-node-safe submission and inspection; no model/data work here."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from receipt import atomic_json, reserve_directory, now_utc

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("run", "status", "results"), default="run")
    parser.add_argument("--account")
    parser.add_argument("--models", default="configs/external-models.json")
    parser.add_argument("--datasets", default="configs/external-datasets.json")
    parser.add_argument("--suite")
    parser.add_argument("--reuse-data-dir")
    parser.add_argument("--prepare-job-id")
    parser.add_argument("--memory", default="128G")
    parser.add_argument("--resource-profile", default="configs/resources/nibi-h100-80gb.json")
    args = parser.parse_args()
    pointer = ROOT / "runs/external-evaluation-latest.json"
    if args.command != "run":
        suite = Path(args.suite).resolve() if args.suite else Path(json.loads(pointer.read_text())["suite"])
        manifest = json.loads((suite / "suite.json").read_text())
        if args.command == "results":
            if "evaluation" not in manifest:
                raise SystemExit("Evaluation was not submitted; inspect suite status and receipts.")
            report = Path(manifest["evaluation"]["run_dir"]) / "results.json"
            if not report.is_file():
                raise SystemExit("Results not available; use status. Submission is not completion.")
            result = json.loads(report.read_text())
            receipt = json.loads(Path(manifest["evaluation"]["receipt"]).read_text())
            job = str(receipt.get("job_id", ""))
            accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job, "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=20)
            completed = accounting.returncode == 0 and any(row.split("|")[:3] == [job, "COMPLETED", "0:0"] for row in accounting.stdout.splitlines())
            if result.get("state") != "SUCCEEDED" or receipt["state"] != "SUCCEEDED" or not completed:
                print("PROVISIONAL: evaluation or scheduler completion is unverified; these are not final results.")
            print(report.read_text())
        else:
            print("Suite:", suite)
            print("Submission state:", manifest["state"], "(current stage states below)")
            for stage, entry in manifest.items():
                if isinstance(entry, dict) and "receipt" in entry:
                    receipt = json.loads(Path(entry["receipt"]).read_text())
                    job = receipt.get("job_id")
                    print(stage, "application:", receipt["state"], "job:", job, "receipt:", entry["receipt"])
                    if job:
                        result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", str(job), "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=20)
                        print(result.stdout or result.stderr)
        return
    if not args.account:
        parser.error("--account is required for run")
    source_models = ROOT / args.models
    models = json.loads(source_models.read_text())
    for row in models["models"]:
        for key in ("checkpoint", "calibration"):
            path = (ROOT / row[key]).resolve()
            if not path.is_file():
                parser.error(f"missing {key}: {path}")
            row[key] = str(path)
    datasets_path = (ROOT / args.datasets).resolve()
    datasets = json.loads(datasets_path.read_text())
    xlam_dir = (ROOT / datasets["xlam_dir"]).resolve()
    for split in ("train", "validation", "calibration", "test"):
        if not (xlam_dir / f"{split}.jsonl").is_file():
            parser.error(f"missing xLAM {split} file: {xlam_dir}")
    datasets["xlam_dir"] = str(xlam_dir)
    suite = reserve_directory(ROOT / "runs/external-evaluation")
    atomic_json(suite / "models.json", models)
    atomic_json(suite / "datasets.json", datasets)
    manifest = {"state": "SUBMITTING", "created_at": now_utc(), "account": args.account}
    atomic_json(suite / "suite.json", manifest)
    atomic_json(pointer, {"suite": str(suite)})
    print("Suite:", suite, flush=True)
    def submit(stage, extra):
        command = [str(ROOT / "scripts/submit.sh"), "--stage", stage, "--account", args.account,
                   "--memory", args.memory, *extra]
        if stage == "external-evaluate":
            command += ["--resource-profile", args.resource_profile]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(process.stdout, end="", flush=True)
        print(process.stderr, end="", flush=True)
        fields = dict(re.findall(r"^(JOB_ID|RUN_DIR|RECEIPT)=(.+)$", process.stdout, re.MULTILINE))
        if "RECEIPT" not in fields:
            raise RuntimeError(f"{stage}: missing durable submission receipt")
        record = json.loads(Path(fields["RECEIPT"]).read_text())
        entry = {"job_id": str(record["job_id"]) if record.get("job_id") else None, "run_dir": record["run_dir"], "receipt": fields["RECEIPT"]}
        manifest["prepare" if stage == "external-prepare" else "evaluation"] = entry
        atomic_json(suite / "suite.json", manifest)
        if not str(record.get("job_id", "")).isdigit() or record["state"] not in {"SUBMITTED", "PENDING", "ALLOCATED", "STARTED", "RUNNING", "SUCCEEDED"}:
            raise RuntimeError(f"{stage}: {record['state']}; inspect {fields['RECEIPT']} before retry")
        if process.returncode:
            print(f"{stage}: monitor exited {process.returncode}; accepted durable state {record['state']}; downstream submission will reconcile dependency again.", flush=True)
        return entry
    try:
        if args.reuse_data_dir:
            if not args.prepare_job_id or not args.prepare_job_id.isdigit():
                parser.error("--reuse-data-dir requires numeric --prepare-job-id")
            data_dir = (ROOT / args.reuse_data_dir).resolve()
            prepared = json.loads((data_dir / "manifest.json").read_text())
            if not prepared.get("datasets"):
                raise ValueError("reused preparation has no datasets")
            for row in prepared["datasets"]:
                path = data_dir / row["path"]
                if not path.is_file() or row["examples"] < 1:
                    raise ValueError(f"invalid frozen dataset: {path}")
            accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", args.prepare_job_id, "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=20)
            rows = [row.split("|") for row in accounting.stdout.splitlines()]
            if accounting.returncode or not any(row[:3] == [args.prepare_job_id, "COMPLETED", "0:0"] for row in rows):
                raise ValueError("reused prepare job is not verified COMPLETED|0:0")
            receipts = list((ROOT / "logs/submissions").glob("external-prepare-*/receipt.json"))
            receipt = next((path for path in receipts if str(json.loads(path.read_text()).get("job_id")) == args.prepare_job_id and Path(json.loads(path.read_text())["run_dir"]).resolve() == data_dir), None)
            if receipt is None or json.loads(receipt.read_text())["state"] != "SUCCEEDED":
                raise ValueError("matching successful prepare receipt missing")
            for previous in (ROOT / "runs").glob("external-evaluation*/suite.json"):
                old = json.loads(previous.read_text())
                if previous.parent == suite or not old.get("evaluation"):
                    continue
                if Path(old.get("prepare", {}).get("run_dir", "")).resolve() != data_dir:
                    continue
                prior_job = str(old["evaluation"].get("job_id", ""))
                if not prior_job.isdigit():
                    prior_receipt_path = old["evaluation"].get("receipt")
                    prior_receipt = json.loads(Path(prior_receipt_path).read_text()) if prior_receipt_path else {}
                    if prior_receipt.get("state") == "SUBMISSION_REJECTED" and prior_receipt.get("job_id") is None:
                        continue
                    raise ValueError(f"prior evaluation submission is ambiguous: {previous}")
                queued = subprocess.run(["squeue", "-h", "-j", prior_job, "-o", "%i|%T"], capture_output=True, text=True, timeout=20)
                if queued.returncode == 0 and queued.stdout.strip():
                    raise ValueError(f"prior evaluation remains active: {prior_job}; inspect {previous}")
                accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", prior_job, "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=20)
                states = [row.split("|") for row in accounting.stdout.splitlines() if row.split("|")[0] == prior_job]
                terminal_failures = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
                if accounting.returncode or len(states) != 1 or states[0][1].split()[0].rstrip("+") not in terminal_failures:
                    raise ValueError(f"prior evaluation succeeded or is unverifiable: {prior_job}; inspect {previous}")
            manifest["prepare"] = {"job_id": args.prepare_job_id, "run_dir": str(data_dir), "receipt": str(receipt), "reused": True}
        else:
            manifest["prepare"] = submit("external-prepare", ["--config", str(suite / "datasets.json"), "--output-dir", str(suite / "data")])
        atomic_json(suite / "suite.json", manifest)
        manifest["evaluation"] = submit("external-evaluate", ["--config", str(suite / "models.json"), "--data-dir", manifest["prepare"]["run_dir"], "--output-dir", str(suite / "evaluation"), *([] if args.reuse_data_dir else ["--dependency", manifest["prepare"]["job_id"]])])
        manifest["state"] = "QUEUED"
    except BaseException as error:
        manifest.update(state="STOPPED", reason=str(error))
        raise
    finally:
        atomic_json(suite / "suite.json", manifest)


if __name__ == "__main__":
    main()
