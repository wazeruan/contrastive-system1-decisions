#!/usr/bin/env python3
"""Validate a training configuration before creating any Slurm jobs."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def validate(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    required = {
        "model_name", "model_revision", "architecture", "projection_dim", "tau",
        "seed", "epochs", "early_stopping_patience", "batch_size", "eval_batch_size",
        "learning_rate", "weight_decay", "max_grad_norm", "max_query_length",
        "max_action_length", "gradient_checkpointing", "require_h100", "use_bf16",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config missing fields: {', '.join(missing)}")
    if not isinstance(config["model_name"], str) or not config["model_name"].strip():
        raise ValueError("model_name must be a nonempty string")
    if not re.fullmatch(r"[0-9a-f]{40}", str(config["model_revision"])):
        raise ValueError("model_revision must be a full 40-character commit SHA")
    if config["architecture"] not in {"shared_tied", "shared_heads", "separate"}:
        raise ValueError("unsupported architecture")

    for key in ("projection_dim", "epochs", "early_stopping_patience", "batch_size",
                "eval_batch_size", "max_query_length", "max_action_length"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    for key, allow_zero in (("tau", False), ("learning_rate", False),
                            ("weight_decay", True), ("max_grad_norm", False)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{key} must be {'nonnegative' if allow_zero else 'positive'}")

    for key in ("gradient_checkpointing", "require_h100", "use_bf16"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_config.py CONFIG.json")
    try:
        validate(Path(sys.argv[1]))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid training config: {error}") from error


if __name__ == "__main__":
    main()
