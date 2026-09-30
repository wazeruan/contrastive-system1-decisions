from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from contrastive_system1_decisions import model, model_cache


class TinyTokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        del padding, truncation, return_tensors
        encoded = [
            [ord(character) % 31 + 1 for character in text[:max_length]] or [1]
            for text in texts
        ]
        width = max(map(len, encoded))
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, token_ids in enumerate(encoded):
            input_ids[row, : len(token_ids)] = torch.tensor(token_ids)
            attention_mask[row, : len(token_ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyEncoder(nn.Module):
    def __init__(self, output_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, self.config.hidden_size)
        self.output_dtype = output_dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        hidden = self.embedding(input_ids).to(dtype=self.output_dtype)
        return SimpleNamespace(last_hidden_state=hidden)


class ModelDtypeTests(unittest.TestCase):
    def test_half_hidden_states_work_with_fp32_and_half_projection_weights(self) -> None:
        tokenizer = TinyTokenizer()
        for architecture in model.ARCHITECTURES:
            for hidden_dtype in (torch.float16, torch.bfloat16):
                for projection_dtype in (torch.float32, torch.float16):
                    with self.subTest(
                        architecture=architecture,
                        hidden_dtype=hidden_dtype,
                        projection_dtype=projection_dtype,
                    ):
                        calls: list[dict[str, object]] = []

                        def load_model(*args, **kwargs):
                            del args
                            calls.append(kwargs)
                            return TinyEncoder(hidden_dtype)

                        with patch.object(model.AutoModel, "from_pretrained", side_effect=load_model):
                            scorer = model.DualEncoderScorer(
                                "local/tiny", architecture=architecture, projection_dim=4
                            )

                        expected_loads = 2 if architecture == "separate" else 1
                        self.assertEqual(len(calls), expected_loads)
                        self.assertTrue(
                            all(call.get("torch_dtype") is torch.float32 for call in calls)
                        )
                        self.assertEqual(scorer.query_projection.weight.dtype, torch.float32)
                        scorer.query_projection.to(dtype=projection_dtype)
                        scorer.action_projection.to(dtype=projection_dtype)

                        hidden = torch.tensor(
                            [[[1.0, 2.0], [3.0, 4.0]]], dtype=hidden_dtype
                        )
                        mask = torch.tensor([[1, 1]])
                        pooled = scorer._mean_pool(hidden, mask)
                        self.assertEqual(pooled.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(pooled).all().item())

                        rows = scorer.score_groups(
                            ["query alpha", "query beta"],
                            [["tool one", "tool two"], ["tool three", "tool four", "tool five"]],
                            tokenizer,
                            max_query_length=16,
                            max_action_length=16,
                            device=torch.device("cpu"),
                        )
                        self.assertTrue(all(row.dtype == torch.float32 for row in rows))
                        self.assertTrue(all(torch.isfinite(row).all().item() for row in rows))
                        torch.cat(rows).square().mean().backward()
                        gradients = [parameter.grad for parameter in scorer.parameters()]
                        self.assertTrue(gradients)
                        self.assertTrue(
                            all(
                                gradient is not None and torch.isfinite(gradient).all().item()
                                for gradient in gradients
                            )
                        )

    def test_cpu_model_cache_requests_float32_parameters(self) -> None:
        revision = "a" * 40
        config = {"model_name": "local/tiny", "model_revision": revision}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest_path = root / "model-cache-manifest.json"
            with (
                patch.object(model_cache.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()),
                patch.object(
                    model_cache.AutoModel,
                    "from_pretrained",
                    return_value=TinyEncoder(torch.float32),
                ) as load_model,
            ):
                result = model_cache.prepare_model(config_path, manifest_path)

        self.assertIs(load_model.call_args.kwargs.get("torch_dtype"), torch.float32)
        self.assertEqual(result["parameter_dtype"], "torch.float32")


if __name__ == "__main__":
    unittest.main()
