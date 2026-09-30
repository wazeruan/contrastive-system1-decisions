"""Allocation preflight and atomic application-start markers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import torch


def load_gpu_profile(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    contents = profile_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected GPU profile SHA-256 must be a 64-character lowercase hex digest")
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"GPU profile changed after submission: expected SHA-256 {expected_sha256}, got {digest}"
        )
    profile = json.loads(contents)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError(f"unsupported GPU profile format: {profile_path}")
    if not isinstance(profile.get("runtime"), dict) or not isinstance(profile.get("scheduler"), dict):
        raise ValueError(f"GPU profile needs scheduler and runtime objects: {profile_path}")
    scheduler = profile["scheduler"]
    if scheduler.get("gpu_option") not in {"--gpus", "--gres"}:
        raise ValueError("GPU profile gpu_option must be --gpus or --gres")
    if not re.fullmatch(r"[A-Za-z0-9_.:=+-]+", str(scheduler.get("gpu_request", ""))):
        raise ValueError("GPU profile gpu_request has invalid Slurm resource syntax")
    runtime = profile["runtime"]
    for key in ("min_memory_gib", "min_free_memory_gib"):
        value = runtime.get(key, 0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"GPU profile {key} must be a nonnegative number")
    if not isinstance(runtime.get("require_cuda", True), bool):
        raise ValueError("GPU profile require_cuda must be true or false")
    if runtime.get("require_cuda", True) is not True:
        raise ValueError("GPU workload profiles must require CUDA")
    if not isinstance(runtime.get("require_bf16", False), bool):
        raise ValueError("GPU profile require_bf16 must be true or false")
    if not isinstance(runtime.get("device_name_contains", ""), str):
        raise ValueError("GPU profile device_name_contains must be a string")
    profile["_sha256"] = digest
    return profile


def preflight_gpu(
    require_h100: bool = False,
    min_memory_gib: float = 75.0,
    gpu_profile: str | Path | None = None,
    gpu_profile_sha256: str | None = None,
) -> dict[str, Any]:
    profile = load_gpu_profile(gpu_profile, gpu_profile_sha256) if gpu_profile is not None else None
    runtime = profile["runtime"] if profile else {}
    min_memory = float(runtime.get("min_memory_gib", min_memory_gib))
    min_free = float(runtime.get("min_free_memory_gib", 70.0 if require_h100 else 0.0))
    require_cuda = bool(runtime.get("require_cuda", require_h100))
    require_bf16 = bool(runtime.get("require_bf16", require_h100))
    device_name_contains = str(runtime.get("device_name_contains", ""))
    result: dict[str, Any] = {
        "host": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
    }
    if (require_cuda or require_h100) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command inside the requested Slurm allocation")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free_memory, total_memory = torch.cuda.mem_get_info()
        result.update(
            {
                "device": properties.name,
                "memory_gib": total_memory / (1024**3),
                "free_memory_gib": free_memory / (1024**3),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        if profile:
            result["gpu_profile"] = profile.get("name")
            result["gpu_profile_sha256"] = profile["_sha256"]
            if device_name_contains and device_name_contains.casefold() not in properties.name.casefold():
                raise RuntimeError(
                    f"GPU profile {profile.get('name')!r} requires a device name containing "
                    f"{device_name_contains!r}; allocated {properties.name!r}"
                )
            if result["memory_gib"] < min_memory:
                raise RuntimeError(
                    f"GPU profile {profile.get('name')!r} requires at least {min_memory:.1f} GiB "
                    f"got {result['memory_gib']:.2f} GiB"
                )
        elif require_h100:
            if "h100" not in properties.name.casefold():
                raise RuntimeError(f"Expected an H100 allocation, got {properties.name}")
            if result["memory_gib"] < min_memory:
                raise RuntimeError(
                    f"Expected at least {min_memory:.1f} GiB GPU memory, got {result['memory_gib']:.2f} GiB"
                )
        if result["free_memory_gib"] < min_free:
            if profile:
                raise RuntimeError(
                    f"GPU profile {profile.get('name')!r} requires at least {min_free:.1f} GiB "
                    f"free GPU memory; got {result['free_memory_gib']:.2f} GiB"
                )
            raise RuntimeError(
                f"Expected at least {min_free:.1f} GiB free GPU memory before model startup, "
                f"got {result['free_memory_gib']:.2f} GiB"
            )
        if require_bf16 and not result["bf16_supported"]:
            if profile:
                raise RuntimeError(f"GPU profile {profile.get('name')!r} requires BF16 support")
            else:
                raise RuntimeError("The allocated accelerator does not support BF16")
    return result


def write_started_marker(
    path: str | Path,
    run_dir: str | Path,
    workflow: str,
    require_h100: bool = False,
    details: dict[str, Any] | None = None,
    min_memory_gib: float = 75.0,
    gpu_profile: str | Path | None = None,
    gpu_profile_sha256: str | None = None,
) -> dict[str, Any]:
    allocation = preflight_gpu(
        require_h100=require_h100 and gpu_profile is None,
        min_memory_gib=min_memory_gib,
        gpu_profile=gpu_profile,
        gpu_profile_sha256=gpu_profile_sha256,
    )
    marker = {
        "state": "STARTED",
        "workflow": workflow,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(Path(run_dir).resolve()),
        "allocation": allocation,
        "gpu_profile_path": str(Path(gpu_profile).expanduser().resolve()) if gpu_profile else None,
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    if details:
        marker.update(details)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return marker
