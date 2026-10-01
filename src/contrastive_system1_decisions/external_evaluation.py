"""Evaluate frozen external selector slices with existing calibrated checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from .calibration import collect_logits, load_model
from .data import read_jsonl
from .evaluation import _metrics, _write_result
from .model import forward_precision
from .runtime import preflight_gpu, write_started_marker


def _file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def evaluate_external(models_path: str | Path, data_dir: str | Path, output_dir: str | Path,
                      batch_size: int = 4, started_marker: str | Path | None = None,
                      gpu_profile: str | Path | None = None,
                      gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    """One allocation, same frozen data, no calibration refitting or training."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path(models_path).read_text())["models"]
    names = [row["name"] for row in models]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("unsafe model name")
    if not models or len(set(names)) != len(names):
        raise ValueError("model names must be nonempty and unique")
    manifest_path = Path(data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    datasets = manifest["datasets"]
    dataset_names = [row["name"] for row in datasets]
    if len(set(dataset_names)) != len(dataset_names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in dataset_names):
        raise ValueError("dataset names must be unique safe path names")
    if not datasets:
        raise ValueError("external manifest contains no datasets")
    for row in models:
        for key in ("checkpoint", "calibration"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    frozen = []
    for dataset in datasets:
        path = Path(data_dir) / dataset["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dataset["sha256"]:
            raise ValueError(f"frozen dataset hash mismatch: {path}")
        examples = read_jsonl(path)
        if len(examples) != dataset["examples"] or not examples:
            raise ValueError(f"frozen dataset count mismatch: {path}")
        frozen.append((dataset, examples))
    if gpu_profile:
        preflight_gpu(gpu_profile=gpu_profile, gpu_profile_sha256=gpu_profile_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {"state": "RUNNING", "dataset_manifest_sha256":
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "models": [],
        "scope": "custom single-target tool selection; no arguments or official leaderboard scores",
        "calibration": "existing xLAM calibration reused; no external labels used for fitting",
        "dataset_provenance": [{key: value for key, value in row.items() if key != "path"} for row in datasets]}
    try:
        for index, entry in enumerate(models):
            calibration = json.loads(Path(entry["calibration"]).read_text())
            if Path(calibration.get("checkpoint", "")).resolve() != Path(entry["checkpoint"]).resolve():
                raise ValueError("calibration does not belong to selected checkpoint")
            temperature = float(calibration["temperature"])
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("invalid saved calibration temperature")
            model, tokenizer, config = load_model(entry["checkpoint"], device)
            if index == 0 and started_marker:
                write_started_marker(started_marker, root, "external-evaluate", details={"models": names},
                    gpu_profile=gpu_profile, gpu_profile_sha256=gpu_profile_sha256)
            model_result = {"name": entry["name"], "checkpoint": str(Path(entry["checkpoint"]).resolve()),
                "calibration": str(Path(entry["calibration"]).resolve()),
                "forward_precision": forward_precision(config, device), "datasets": [],
                "architecture": config.get("architecture"), "optimizer": config.get("optimizer", "adamw"),
                "checkpoint_sha256": _file_sha256(entry["checkpoint"]),
                "calibration_sha256": hashlib.sha256(Path(entry["calibration"]).read_bytes()).hexdigest()}
            for dataset, examples in frozen:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                logits, _ = collect_logits(model, tokenizer, examples, config, device, batch_size)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                metrics = _metrics(logits, examples, temperature)
                strata = {}
                for count in sorted({len(example.candidates) for example in examples}):
                    indices = [i for i, example in enumerate(examples) if len(example.candidates) == count]
                    strata[str(count)] = _metrics([logits[i] for i in indices], [examples[i] for i in indices], temperature)
                metrics["candidate_count_strata"] = strata
                metrics.update({"dataset": dataset["name"], "data_sha256": dataset["sha256"],
                    "elapsed_seconds": elapsed, "examples_per_second": len(examples) / elapsed,
                    "timing_scope": "tokenization plus forward scoring; excludes checkpoint loading; sequential run order"})
                predictions = [{"example_id": example.example_id, "prediction": int(score.argmax()),
                    "positive_indices": [i for i, value in enumerate(example.target_weights) if value > 0],
                    "scores": score.float().tolist()} for score, example in zip(logits, examples)]
                folder = root / entry["name"] / dataset["name"]
                _write_result(metrics, folder / "metrics.json")
                _write_result({"predictions": predictions}, folder / "predictions.json")
                model_result["datasets"].append(metrics)
            result["models"].append(model_result)
            _write_result(result, root / "results.json")
            del model, tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["paired_comparisons"] = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                for dataset, _ in frozen:
                    a = json.loads((root / left / dataset["name"] / "predictions.json").read_text())["predictions"]
                    b = json.loads((root / right / dataset["name"] / "predictions.json").read_text())["predictions"]
                    if [row["example_id"] for row in a] != [row["example_id"] for row in b]:
                        raise ValueError("paired predictions are not aligned")
                    ac = [row["prediction"] in row["positive_indices"] for row in a]
                    bc = [row["prediction"] in row["positive_indices"] for row in b]
                    result["paired_comparisons"].append({"left": left, "right": right, "dataset": dataset["name"],
                        "left_only_correct": sum(x and not y for x, y in zip(ac, bc)),
                        "right_only_correct": sum(y and not x for x, y in zip(ac, bc)), "examples": len(a)})
        result["state"] = "SUCCEEDED"
        _write_result(result, root / "results.json")
        lines = ["# External tool-selection evaluation", "", result["scope"], "", result["calibration"], "",
                 "| Model | Dataset | Examples | Accuracy | MRR | NLL | ECE |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in result["models"]:
            for metric in row["datasets"]:
                lines.append(f"| {row['name']} | {metric['dataset']} | {metric['examples']} | "
                    f"{metric['top1_tool_accuracy']:.4f} | {metric['mrr']:.4f} | {metric['nll']:.4f} | {metric['ece_15_bins']:.4f} |")
        lines += ["", "## Dataset provenance and limitations", ""]
        for dataset in datasets:
            lines += [f"### {dataset['name']}", "",
                f"Source: {dataset.get('source_url', dataset.get('repository', 'not recorded'))}",
                f"Revision: `{dataset.get('revision', 'not recorded')}`; license: {dataset.get('license', 'not recorded')}",
                f"SHA-256: `{dataset['sha256']}`",
                dataset.get("caveat", "No dataset-specific caveat recorded."), ""]
        temporary = root / "RESULTS.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(root / "RESULTS.md")
    except BaseException as error:
        result.update(state="FAILED", reason=f"{type(error).__name__}: {error}")
        _write_result(result, root / "results.json")
        raise
    return result
