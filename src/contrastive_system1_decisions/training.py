"""Training, checkpointing, and score evaluation for the listwise selector."""

from __future__ import annotations

import json
import hashlib
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import DecisionExample, read_jsonl
from .model import DualEncoderScorer, load_tokenizer
from .runtime import preflight_gpu, write_started_marker


_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; will checkpoint after the current epoch.", flush=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _record_run_manifest(config_path: Path, data_dir: Path, run_dir: Path) -> None:
    """Finalize data fingerprints after any upstream Slurm dependency has completed."""
    manifest_path = run_dir / "run-manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    split_names = ("train.jsonl", "validation.jsonl")
    existing.update(
        {
            "created_at_utc": existing.get(
                "created_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "config_snapshot": str(config_path.resolve()),
            "data_dir": str(data_dir.resolve()),
            "data_manifest": str((data_dir / "manifest.json").resolve())
            if (data_dir / "manifest.json").is_file()
            else None,
            "data_manifest_sha256": hashlib.sha256(
                (data_dir / "manifest.json").read_bytes()
            ).hexdigest()
            if (data_dir / "manifest.json").is_file()
            else None,
            "split_sha256": {
                name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                for name in split_names
            },
        }
    )
    _atomic_json(existing, manifest_path)


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model_name", "model_revision", "architecture", "projection_dim", "tau"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config is missing required keys: {', '.join(missing)}")
    return config


def _make_loader(path: Path, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    examples = read_jsonl(path)
    if not examples:
        raise ValueError(f"dataset split is empty: {path}")
    generator = torch.Generator()
    generator.manual_seed(seed)

    def collate(batch: list[DecisionExample]) -> list[DecisionExample]:
        return batch

    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )


def _loss_and_metrics(
    logits_by_row: list[torch.Tensor],
    batch: list[DecisionExample],
) -> tuple[torch.Tensor, dict[str, float]]:
    row_losses: list[torch.Tensor] = []
    correct = 0
    reciprocal_rank = 0.0
    nll_total = 0.0
    for logits, example in zip(logits_by_row, batch):
        target = torch.tensor(example.target_weights, dtype=logits.dtype, device=logits.device)
        log_probs = F.log_softmax(logits.float(), dim=0)
        row_losses.append(-(target * log_probs).sum())
        correct += int(torch.argmax(logits).item() == int(torch.argmax(target).item()))
        order = torch.argsort(logits, descending=True)
        target_indices = torch.nonzero(target > 0, as_tuple=False).flatten().tolist()
        ranks = [int(torch.nonzero(order == index, as_tuple=False)[0].item()) + 1 for index in target_indices]
        reciprocal_rank += max((1.0 / rank for rank in ranks), default=0.0)
        nll_total += float(-(target * log_probs).sum().detach().cpu().item())
    loss = torch.stack(row_losses).mean()
    if not torch.isfinite(loss).item():
        for logits, example in zip(logits_by_row, batch):
            target = torch.tensor(example.target_weights, dtype=logits.dtype, device=logits.device)
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"non-finite candidate scores for example {example.example_id}")
            if not torch.isfinite(target).all().item():
                raise FloatingPointError(f"non-finite target weights for example {example.example_id}")
            log_probs = F.log_softmax(logits.float(), dim=0)
            if not torch.isfinite(log_probs).all().item():
                raise FloatingPointError(f"non-finite log probabilities for example {example.example_id}")
            row_loss = -(target * log_probs).sum()
            if not torch.isfinite(row_loss).item():
                raise FloatingPointError(f"non-finite contrastive loss for example {example.example_id}")
        raise FloatingPointError("non-finite contrastive loss for the batch")
    count = len(batch)
    metrics = {
        "accuracy": correct / count,
        "mrr": reciprocal_rank / count,
        "nll": nll_total / count,
    }
    return loss, metrics


def _run_epoch(
    model: DualEncoderScorer,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0, "nll": 0.0}
    count = 0
    for batch in loader:
        queries = [example.query for example in batch]
        candidates = [[candidate.text for candidate in example.candidates] for example in batch]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and bool(config.get("use_bf16", True))
                         and torch.cuda.is_bf16_supported()),
            ):
                logits = model.score_groups(
                    queries,
                    candidates,
                    tokenizer,
                    int(config["max_query_length"]),
                    int(config["max_action_length"]),
                    device,
                )
                loss, metrics = _loss_and_metrics(logits, batch)
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["max_grad_norm"])
                )
                if not torch.isfinite(gradient_norm).item():
                    raise FloatingPointError("non-finite gradient norm; optimizer update skipped")
                optimizer.step()
        batch_count = len(batch)
        totals["loss"] += float(loss.detach().cpu().item()) * batch_count
        for name, value in metrics.items():
            totals[name] += value * batch_count
        count += batch_count
    if count == 0:
        raise RuntimeError("loader produced no batches")
    return {name: value / count for name, value in totals.items()} | {"examples": count}


def _save_checkpoint(
    model: DualEncoderScorer,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_validation_nll: float,
    run_dir: Path,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_nll": best_validation_nll,
        "config": config,
    }
    _atomic_torch_save(checkpoint, run_dir / "checkpoints" / "last.pt")
    _atomic_json(
        {key: value for key, value in checkpoint.items() if key not in {"state_dict", "optimizer"}},
        run_dir / "checkpoints" / "last.json",
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")


def train(config_path: str | Path, data_dir: str | Path, run_dir: str | Path, resume: bool = False,
          started_marker: str | Path | None = None,
          gpu_profile: str | Path | None = None,
          gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_dir)
    train_loader = _make_loader(data_root / "train.jsonl", int(config["batch_size"]), True, seed)
    validation_loader = _make_loader(data_root / "validation.jsonl", int(config["eval_batch_size"]), False, seed)
    _record_run_manifest(Path(config_path), data_root, destination)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_h100 = bool(config.get("require_h100", True)) and gpu_profile is None
    if gpu_profile is not None:
        preflight_gpu(gpu_profile=gpu_profile, gpu_profile_sha256=gpu_profile_sha256)
        device = torch.device("cuda")
    elif require_h100:
        _preflight_h100()
        device = torch.device("cuda")

    tokenizer = load_tokenizer(config["model_name"], config.get("model_revision"))
    model = DualEncoderScorer(
        model_name=config["model_name"],
        architecture=config["architecture"],
        projection_dim=int(config["projection_dim"]),
        revision=config.get("model_revision"),
        tau=float(config["tau"]),
    ).to(device)
    if bool(config.get("gradient_checkpointing", False)):
        encoders = {id(model.query_encoder): model.query_encoder, id(model.action_encoder): model.action_encoder}
        for encoder in encoders.values():
            encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    first_epoch = 0
    global_step = 0
    best_validation_nll = float("inf")
    if resume:
        checkpoint_path = destination / "checkpoints" / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("resume config differs from the checkpoint config")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_nll = float(checkpoint["best_validation_nll"])

    if started_marker is not None:
        _write_started_marker(Path(started_marker), destination, model,
                              require_h100=require_h100, gpu_profile=gpu_profile,
                              gpu_profile_sha256=gpu_profile_sha256)

    history_path = destination / "history.jsonl"
    max_epochs = int(config["epochs"])
    patience = int(config.get("early_stopping_patience", 2))
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(first_epoch, max_epochs):
        train_metrics = _run_epoch(model, train_loader, tokenizer, device, config, optimizer)
        global_step += int(np.ceil(train_metrics["examples"] / int(config["batch_size"])))
        validation_metrics = _run_epoch(model, validation_loader, tokenizer, device, config, None)
        validation_nll = float(validation_metrics["nll"])
        if not np.isfinite(validation_nll):
            raise RuntimeError(
                f"validation NLL is non-finite at epoch {epoch + 1}: {validation_nll}"
            )
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": time.time() - start_time,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            _atomic_torch_save(
                {"state_dict": model.state_dict(), "config": config},
                destination / "checkpoints" / "best.pt",
            )
            _atomic_json({"config": config, "best_validation_nll": best_validation_nll,
                          "epoch": epoch + 1}, destination / "checkpoints" / "best.json")
            stale_epochs = 0
        else:
            stale_epochs += 1
        _save_checkpoint(model, tokenizer, optimizer, config, epoch + 1, global_step,
                         best_validation_nll, destination)
        print(json.dumps(record), flush=True)
        if _STOP_REQUESTED:
            _write_status(destination, "PREEMPTED", {"epoch": epoch + 1, "global_step": global_step})
            return {"status": "PREEMPTED", "run_dir": str(destination), "global_step": global_step}
        if stale_epochs >= patience:
            break

    best_checkpoint = destination / "checkpoints" / "best.pt"
    if (not best_checkpoint.is_file() or best_checkpoint.stat().st_size == 0
            or not np.isfinite(best_validation_nll)):
        raise RuntimeError(f"training ended without a valid best checkpoint: {best_checkpoint}")
    _write_status(destination, "SUCCEEDED", {"global_step": global_step, "best_validation_nll": best_validation_nll})
    return {"status": "SUCCEEDED", "run_dir": str(destination), "global_step": global_step,
            "best_validation_nll": best_validation_nll}


def _preflight_h100(min_memory_gib: float = 75.0) -> dict[str, Any]:
    return preflight_gpu(require_h100=True, min_memory_gib=min_memory_gib)


def _write_started_marker(path: Path, run_dir: Path, model: DualEncoderScorer,
                          require_h100: bool = True,
                          gpu_profile: str | Path | None = None,
                          gpu_profile_sha256: str | None = None) -> None:
    marker = write_started_marker(
        path,
        run_dir,
        "train",
        require_h100=require_h100 and gpu_profile is None,
        gpu_profile=gpu_profile,
        gpu_profile_sha256=gpu_profile_sha256,
        details={"architecture": model.architecture, "model_name": model.model_name},
    )
    _write_status(path.parent, "STARTED", marker)


def _write_status(run_dir: Path, state: str, details: dict[str, Any]) -> None:
    _atomic_json({"state": state, "job_id": os.environ.get("SLURM_JOB_ID"),
                  "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  **details}, run_dir / "status.json")
