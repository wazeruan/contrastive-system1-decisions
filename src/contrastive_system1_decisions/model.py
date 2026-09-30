"""Dual-encoder scoring models with controlled parameter sharing."""

from __future__ import annotations

import json
import os
from typing import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


ARCHITECTURES = ("shared_tied", "shared_heads", "separate")


def bf16_autocast_enabled(config: dict[str, object], device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and bool(config.get("use_bf16", True))
        and torch.cuda.is_bf16_supported()
    )


def forward_precision(config: dict[str, object], device: torch.device) -> str:
    return "BF16 autocast" if bf16_autocast_enabled(config, device) else "FP32"


def verify_prepared_model(model_name: str, revision: str | None) -> None:
    """Require the CPU-prepared pinned model snapshot for offline Slurm jobs."""
    offline_values = {"1", "true", "yes"}
    is_offline = any(
        os.environ.get(name, "").strip().casefold() in offline_values
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    if not is_offline:
        return
    manifest_path = Path(
        os.environ.get("CSD_MODEL_CACHE_MANIFEST", Path.cwd() / "models" / "model-cache-manifest.json")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared model manifest is missing: {manifest_path}; run the prepare-model CPU Slurm stage first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_name") != model_name or manifest.get("revision") != revision:
        raise ValueError(
            "prepared model cache does not match the config's model_name/model_revision; "
            "run prepare-model with this config"
        )


class DualEncoderScorer(nn.Module):
    def __init__(
        self,
        model_name: str,
        architecture: str = "shared_heads",
        projection_dim: int = 256,
        revision: str | None = None,
        tau: float = 0.07,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if projection_dim < 1 or tau <= 0:
            raise ValueError("projection_dim and tau must be positive")
        self.model_name = model_name
        self.architecture = architecture
        self.projection_dim = projection_dim
        self.revision = revision
        self.tau = tau

        verify_prepared_model(model_name, revision)
        self.query_encoder = AutoModel.from_pretrained(model_name, revision=revision)
        if architecture == "separate":
            self.action_encoder = AutoModel.from_pretrained(model_name, revision=revision)
        else:
            self.action_encoder = self.query_encoder

        hidden_size = int(self.query_encoder.config.hidden_size)
        self.query_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        if architecture == "shared_tied":
            self.action_projection = self.query_projection
        else:
            self.action_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def encode(
        self,
        texts: Sequence[str],
        tokenizer: AutoTokenizer,
        side: str,
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if side not in {"query", "action"}:
            raise ValueError("side must be query or action")
        encoder = self.query_encoder if side == "query" else self.action_encoder
        projection = self.query_projection if side == "query" else self.action_projection
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        output = encoder(**tokens)
        pooled = self._mean_pool(output.last_hidden_state, tokens["attention_mask"])
        return F.normalize(projection(pooled).float(), p=2, dim=-1)

    def score_groups(
        self,
        query_texts: Sequence[str],
        candidate_texts: Sequence[Sequence[str]],
        tokenizer: AutoTokenizer,
        max_query_length: int,
        max_action_length: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if len(query_texts) != len(candidate_texts):
            raise ValueError("query and candidate batch sizes differ")
        if any(not candidates for candidates in candidate_texts):
            raise ValueError("each query requires at least one candidate")

        query_vectors = self.encode(query_texts, tokenizer, "query", max_query_length, device)
        flat_actions = [action for group in candidate_texts for action in group]
        action_vectors = self.encode(flat_actions, tokenizer, "action", max_action_length, device)
        counts = [len(group) for group in candidate_texts]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return [
            (action_vectors[start:end] @ query_vectors[row]) / self.tau
            for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
        ]


def load_tokenizer(model_name: str, revision: str | None = None) -> AutoTokenizer:
    verify_prepared_model(model_name, revision)
    return AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
