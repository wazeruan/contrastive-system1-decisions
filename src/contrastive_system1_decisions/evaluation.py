"""Ranking and calibration metrics for dynamic candidate sets."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import collect_logits, load_model
from .data import DecisionExample, download_bfcl, load_bfcl_category, read_jsonl, write_jsonl
from .runtime import preflight_gpu, write_started_marker


def _metrics(logits_rows: list[torch.Tensor], examples: list[DecisionExample], temperature: float) -> dict[str, Any]:
    if len(logits_rows) != len(examples) or not examples:
        raise ValueError("evaluation requires aligned, nonempty scores and examples")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("evaluation temperature must be finite and positive")
    top1: list[float] = []
    reciprocal_ranks: list[float] = []
    nlls: list[float] = []
    briers: list[float] = []
    confidences: list[float] = []
    correctness: list[float] = []
    for logits, example in zip(logits_rows, examples):
        target = torch.tensor(example.target_weights, dtype=torch.float32)
        if logits.shape != target.shape:
            raise ValueError(f"{example.example_id}: score and target shapes do not match")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError(f"{example.example_id}: non-finite candidate scores")
        if not torch.isfinite(target).all().item():
            raise FloatingPointError(f"{example.example_id}: non-finite target weights")
        probabilities = torch.softmax(logits.float() / temperature, dim=0)
        predicted = int(torch.argmax(probabilities).item())
        positive = torch.nonzero(target > 0, as_tuple=False).flatten().tolist()
        top1.append(float(target[predicted].item() > 0))
        order = torch.argsort(probabilities, descending=True).tolist()
        rank = min((order.index(index) + 1 for index in positive), default=len(order) + 1)
        reciprocal_ranks.append(1.0 / rank)
        nlls.append(float(-(target * probabilities.clamp_min(1e-12).log()).sum().item()))
        briers.append(float(((probabilities - target) ** 2).sum().item()))
        confidences.append(float(probabilities.max().item()))
        correctness.append(top1[-1])

    bins = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for index in range(len(bins) - 1):
        members = [i for i, value in enumerate(confidences) if bins[index] <= value < bins[index + 1]
                   or (index == len(bins) - 2 and value == 1.0)]
        if members:
            mean_confidence = float(np.mean([confidences[i] for i in members]))
            mean_accuracy = float(np.mean([correctness[i] for i in members]))
            ece += len(members) / len(examples) * abs(mean_confidence - mean_accuracy)
    metrics = {
        "examples": len(examples),
        "top1_tool_accuracy": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "nll": float(np.mean(nlls)),
        "brier": float(np.mean(briers)),
        "ece_15_bins": float(ece),
        "mean_candidate_count": float(np.mean([len(row.candidates) for row in examples])),
        "temperature": temperature,
        "metric_scope": "custom tool-selection slice; does not score generated arguments or claim BFCL leaderboard equivalence",
    }
    if any(not math.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))):
        raise FloatingPointError("evaluation produced a non-finite metric")
    return metrics


def evaluate_jsonl(checkpoint: str | Path, data_path: str | Path, output_path: str | Path,
                   calibration_path: str | Path | None = None, batch_size: int = 32,
                   started_marker: str | Path | None = None,
                   require_h100: bool = False,
                   gpu_profile: str | Path | None = None,
                   gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    if require_h100 or gpu_profile is not None:
        preflight_gpu(require_h100=require_h100 and gpu_profile is None, gpu_profile=gpu_profile,
                      gpu_profile_sha256=gpu_profile_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(checkpoint, device)
    examples = read_jsonl(data_path)
    if started_marker is not None:
        write_started_marker(
            started_marker,
            Path(output_path).parent,
            "evaluate",
            require_h100=require_h100 and gpu_profile is None,
            details={"checkpoint": str(Path(checkpoint).resolve()), "examples": len(examples)},
            gpu_profile=gpu_profile,
            gpu_profile_sha256=gpu_profile_sha256,
        )
    logits, _ = collect_logits(model, tokenizer, examples, config, device, batch_size)
    temperature = 1.0
    if calibration_path:
        temperature = float(json.loads(Path(calibration_path).read_text(encoding="utf-8"))["temperature"])
    result = _metrics(logits, examples, temperature)
    result["checkpoint"] = str(Path(checkpoint).resolve())
    result["data"] = str(Path(data_path).resolve())
    _write_result(result, output_path)
    return result


def prepare_bfcl(output_dir: str | Path, revision: str = "main") -> dict[str, Any]:
    snapshot, resolved_revision = download_bfcl(output_dir, revision)
    counts: dict[str, int] = {}
    for category in ("multiple", "live_multiple"):
        questions = snapshot / f"BFCL_v3_{category}.json"
        answers = snapshot / "possible_answer" / f"BFCL_v3_{category}.json"
        if not questions.is_file() or not answers.is_file():
            raise FileNotFoundError(f"BFCL snapshot is missing {category} question/answer files")
        examples = load_bfcl_category(questions, answers)
        counts[category] = write_jsonl(snapshot / f"{category}.selector.jsonl", examples)
    result = {
        "source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision_requested": revision,
        "resolved_revision": resolved_revision,
        "categories": counts,
        "metric_scope": "tool selection only; do not report as official full-call BFCL score",
    }
    _write_result(result, snapshot / "selector_manifest.json")
    return result


def evaluate_bfcl(checkpoint: str | Path, data_dir: str | Path, category: str,
                  output_path: str | Path, calibration_path: str | Path | None = None,
                  batch_size: int = 32, started_marker: str | Path | None = None,
                  require_h100: bool = False,
                  gpu_profile: str | Path | None = None,
                  gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    if category not in {"multiple", "live_multiple"}:
        raise ValueError("category must be multiple or live_multiple")
    if require_h100 or gpu_profile is not None:
        preflight_gpu(require_h100=require_h100 and gpu_profile is None, gpu_profile=gpu_profile,
                      gpu_profile_sha256=gpu_profile_sha256)
    examples_path = Path(data_dir) / f"{category}.selector.jsonl"
    examples = read_jsonl(examples_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(checkpoint, device)
    if started_marker is not None:
        write_started_marker(
            started_marker,
            Path(output_path).parent,
            "evaluate-bfcl",
            require_h100=require_h100 and gpu_profile is None,
            details={"checkpoint": str(Path(checkpoint).resolve()), "examples": len(examples),
                     "category": category},
            gpu_profile=gpu_profile,
            gpu_profile_sha256=gpu_profile_sha256,
        )
    logits, _ = collect_logits(model, tokenizer, examples, config, device, batch_size)
    temperature = 1.0
    if calibration_path:
        temperature = float(json.loads(Path(calibration_path).read_text(encoding="utf-8"))["temperature"])
    result = _metrics(logits, examples, temperature)
    result.update({"benchmark": "BFCL V3 selector slice", "category": category,
                   "checkpoint": str(Path(checkpoint).resolve()),
                   "data": str(examples_path.resolve())})
    _write_result(result, output_path)
    return result


def _write_result(result: dict[str, Any], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
