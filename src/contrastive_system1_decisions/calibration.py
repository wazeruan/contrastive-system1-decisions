"""Post-hoc candidate-set temperature calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import DecisionExample, read_jsonl
from .model import DualEncoderScorer, load_tokenizer
from .runtime import preflight_gpu, write_started_marker


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[DualEncoderScorer, Any, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = DualEncoderScorer(
        model_name=config["model_name"],
        architecture=config["architecture"],
        projection_dim=int(config["projection_dim"]),
        revision=config.get("model_revision"),
        tau=float(config["tau"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    tokenizer = load_tokenizer(config["model_name"], config.get("model_revision"))
    return model, tokenizer, config


@torch.no_grad()
def collect_logits(
    model: DualEncoderScorer,
    tokenizer: Any,
    examples: list[DecisionExample],
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logits_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and bool(config.get("use_bf16", True))
                     and torch.cuda.is_bf16_supported()),
        ):
            logits = model.score_groups(
                [row.query for row in batch],
                [[candidate.text for candidate in row.candidates] for row in batch],
                tokenizer,
                int(config["max_query_length"]),
                int(config["max_action_length"]),
                device,
            )
        for row, example in zip(logits, batch):
            scores = row.detach().float().cpu()
            if not torch.isfinite(scores).all().item():
                raise FloatingPointError(f"non-finite candidate scores for example {example.example_id}")
            logits_rows.append(scores)
        target_rows.extend(torch.tensor(row.target_weights, dtype=torch.float32) for row in batch)
    return logits_rows, target_rows


def _mean_nll(logits: list[torch.Tensor], targets: list[torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    if not logits or len(logits) != len(targets):
        raise ValueError("calibration requires aligned, nonempty logits and targets")
    if temperature.numel() != 1 or not torch.isfinite(temperature).all().item():
        raise FloatingPointError("calibration temperature is non-finite or not scalar")
    if temperature.item() <= 0:
        raise ValueError("calibration temperature must be positive")

    losses: list[torch.Tensor] = []
    for row, target in zip(logits, targets):
        if row.shape != target.shape:
            raise ValueError("calibration score and target shapes do not match")
        if not torch.isfinite(row).all().item():
            raise FloatingPointError("calibration received non-finite candidate scores")
        if not torch.isfinite(target).all().item():
            raise FloatingPointError("calibration received non-finite target weights")
        loss = -(target * F.log_softmax(row / temperature, dim=0)).sum()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("calibration produced a non-finite NLL")
        losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    if not torch.isfinite(mean_loss).item():
        raise FloatingPointError("calibration produced a non-finite mean NLL")
    return mean_loss


def fit_temperature(checkpoint: str | Path, data_path: str | Path, output_path: str | Path,
                    batch_size: int = 32, started_marker: str | Path | None = None,
                    require_h100: bool = False,
                    gpu_profile: str | Path | None = None,
                    gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    if require_h100 or gpu_profile is not None:
        preflight_gpu(require_h100=require_h100 and gpu_profile is None, gpu_profile=gpu_profile,
                      gpu_profile_sha256=gpu_profile_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(checkpoint, device)
    examples = read_jsonl(data_path)
    if not examples:
        raise ValueError("calibration split is empty")
    if started_marker is not None:
        write_started_marker(
            started_marker,
            Path(output_path).parent,
            "calibrate",
            require_h100=require_h100 and gpu_profile is None,
            details={"checkpoint": str(Path(checkpoint).resolve()), "examples": len(examples)},
            gpu_profile=gpu_profile,
            gpu_profile_sha256=gpu_profile_sha256,
        )
    logits, targets = collect_logits(model, tokenizer, examples, config, device, batch_size)
    initial = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([initial], lr=0.25, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = initial.exp().clamp(min=0.05, max=20.0)
        loss = _mean_nll(logits, targets, temperature)
        loss.backward()
        if initial.grad is None or not torch.isfinite(initial.grad).all().item():
            raise FloatingPointError("calibration produced a non-finite temperature gradient")
        return loss

    before = float(_mean_nll(logits, targets, torch.tensor(1.0)).item())
    optimizer.step(closure)
    if not torch.isfinite(initial.detach()).all().item():
        raise FloatingPointError("temperature optimizer produced a non-finite parameter")
    temperature = float(initial.detach().exp().clamp(min=0.05, max=20.0).item())
    after = float(_mean_nll(logits, targets, torch.tensor(temperature)).item())
    result = {
        "temperature": temperature,
        "examples": len(examples),
        "nll_before": before,
        "nll_after": after,
        "checkpoint": str(Path(checkpoint).resolve()),
        "calibration_data": str(Path(data_path).resolve()),
        "semantics": "P(selected tool | query, supplied candidates)",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return result
