"""CPU-side materialization of a pinned model revision for offline GPU jobs."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from transformers import AutoModel, AutoTokenizer

from .runtime import write_started_marker


def prepare_model(config_path: str | Path, manifest_path: str | Path,
                  started_marker: str | Path | None = None,
                  run_dir: str | Path | None = None) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    config_bytes = config_source.read_bytes()
    config = json.loads(config_bytes)
    model_name = str(config["model_name"])
    revision = config.get("model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model_revision must be a full 40-character commit SHA before preparing the cache")
    output_dir = Path(run_dir) if run_dir else Path(manifest_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "config.json"
    config_temporary = config_snapshot.with_name(f".{config_snapshot.name}.{os.getpid()}.tmp")
    config_temporary.write_bytes(config_bytes)
    os.replace(config_temporary, config_snapshot)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if started_marker is not None:
        write_started_marker(
            started_marker,
            output_dir,
            "prepare-model",
            details={"model_name": model_name, "revision": revision,
                     "config_sha256": config_sha256},
        )
    # Load on CPU so downloads and initialization stay outside the timed GPU job.
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32)
    result = {
        "model_name": model_name,
        "revision": revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "hidden_size": int(model.config.hidden_size),
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_source),
        "config_snapshot": str(config_snapshot.resolve()),
        "config_sha256": config_sha256,
        "hf_home": os.environ.get("HF_HOME"),
        "semantics": "cached pinned model/tokenizer for offline GPU allocations",
    }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return result
