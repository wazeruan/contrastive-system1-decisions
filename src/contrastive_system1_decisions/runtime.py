"""Allocation preflight and atomic application-start markers."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch


def preflight_gpu(require_h100: bool = False, min_memory_gib: float = 75.0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
    }
    if require_h100 and not torch.cuda.is_available():
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
        if require_h100:
            if "h100" not in properties.name.casefold():
                raise RuntimeError(f"Expected an H100 allocation, got {properties.name}")
            if result["memory_gib"] < min_memory_gib:
                raise RuntimeError(
                    f"Expected at least {min_memory_gib:.1f} GiB GPU memory, "
                    f"got {result['memory_gib']:.2f} GiB"
                )
            if result["free_memory_gib"] < 70.0:
                raise RuntimeError(
                    f"Expected at least 70 GiB free GPU memory before model startup, "
                    f"got {result['free_memory_gib']:.2f} GiB"
                )
            if not result["bf16_supported"]:
                raise RuntimeError("The allocated accelerator does not support BF16")
    return result


def write_started_marker(
    path: str | Path,
    run_dir: str | Path,
    workflow: str,
    require_h100: bool = False,
    details: dict[str, Any] | None = None,
    min_memory_gib: float = 75.0,
) -> dict[str, Any]:
    allocation = preflight_gpu(require_h100=require_h100, min_memory_gib=min_memory_gib)
    marker = {
        "state": "STARTED",
        "workflow": workflow,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(Path(run_dir).resolve()),
        "allocation": allocation,
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
