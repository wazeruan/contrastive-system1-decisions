"""Offline regressions: mocked pretrained loader, real torch Muon updates."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from contrastive_system1_decisions import model
from contrastive_system1_decisions.optimizers import build_optimizer

class PairTokenizer:
    def __init__(self):
        self.calls = []
    def __call__(self, texts, *, text_pair, **kwargs):
        self.calls.append((list(texts), list(text_pair), kwargs))
        ids = torch.tensor([[int(q), int(a)] for q, a in zip(texts, text_pair)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

class PairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(16, 4)
        self.matrix = nn.Linear(4, 4)
    def forward(self, input_ids, attention_mask):
        hidden = self.matrix(self.embedding(input_ids).sum(1)).half()
        return SimpleNamespace(last_hidden_state=hidden[:, None, :])

class CrossMuonTests(unittest.TestCase):
    def make_model(self, chunk=2):
        with patch.object(model.AutoModel, "from_pretrained", return_value=PairEncoder()) as load:
            scorer = model.build_model({"model_name": "tiny", "architecture": "cross_encoder", "pair_batch_size": chunk})
        self.assertIs(load.call_args.kwargs["torch_dtype"], torch.float32)
        return scorer

    def test_pairs_chunk_order_gradients_and_model_restore(self):
        scorer = self.make_model()
        tok = PairTokenizer()
        args = (["1", "2"], [["3", "4", "5"], ["6", "7"]], tok, 8, 8, torch.device("cpu"))
        rows = scorer.score_groups(*args)
        self.assertEqual([len(r) for r in rows], [3, 2])
        self.assertEqual([(q, a) for qs, acts, _ in tok.calls for q, a in zip(qs, acts)], [("1", "3"), ("1", "4"), ("1", "5"), ("2", "6"), ("2", "7")])
        self.assertTrue(all(c[2]["truncation"] == "longest_first" for c in tok.calls))
        clone = self.make_model(20)
        clone.load_state_dict(scorer.state_dict())
        expected = clone.score_groups(*args)
        torch.testing.assert_close(torch.cat(rows), torch.cat(expected))
        loss = sum(torch.nn.functional.cross_entropy(r[None], torch.tensor([0])) for r in rows)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters()))
        self.assertEqual(len(list(scorer.parameters())), len({id(p) for p in scorer.parameters()}))

    def test_calibration_checkpoint_consumer(self):
        from contrastive_system1_decisions import calibration
        scorer = self.make_model()
        config = {"architecture": "cross_encoder", "model_name": "tiny", "pair_batch_size": 2}
        args = (["1"], [["3", "4"]], PairTokenizer(), 8, 8, torch.device("cpu"))
        expected = scorer.score_groups(*args)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            torch.save({"config": config, "state_dict": scorer.state_dict()}, checkpoint)
            with patch.object(model.AutoModel, "from_pretrained", return_value=PairEncoder()), patch.object(calibration, "load_tokenizer", return_value=PairTokenizer()):
                restored, tokenizer, restored_config = calibration.load_model(checkpoint, torch.device("cpu"))
            self.assertIsInstance(restored, model.CrossEncoderScorer)
            self.assertFalse(restored.training)
            self.assertEqual(restored_config, config)
            actual = restored.score_groups(args[0], args[1], tokenizer, 8, 8, torch.device("cpu"))
            torch.testing.assert_close(actual[0], expected[0])

    def test_configs_and_invalid_muon_options(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("validate_config", root / "scripts/validate_config.py")
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        for path in (root / "configs").glob("*.json"):
            validation.validate(path)
        base = json.loads((root / "configs/shared-heads-h100.json").read_text())
        invalid = {"optimizer": "sgd", "muon_learning_rate": float("nan"),
                   "muon_momentum": 1.0, "muon_ns_steps": 0,
                   "pair_batch_size": True, "max_pair_length": 2}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for key, value in invalid.items():
                with self.subTest(key=key):
                    path.write_text(json.dumps({**base, key: value}))
                    with self.assertRaises(ValueError): validation.validate(path)

    def test_real_muon_assignment_update_and_restore(self):
        scorer = self.make_model()
        cfg = {"optimizer": "muon", "learning_rate": 2e-5, "weight_decay": .01}
        opt = build_optimizer(scorer, cfg)
        self.assertIn("query_encoder.matrix.weight", opt.names["muon"])
        self.assertIn("query_encoder.embedding.weight", opt.names["adamw"])
        self.assertIn("scoring_head.weight", opt.names["adamw"])
        assigned = opt.names["muon"] + opt.names["adamw"]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(dict(scorer.named_parameters())))
        for p in scorer.parameters(): p.grad = torch.ones_like(p)
        before = {n: p.detach().clone() for n, p in scorer.named_parameters()}
        opt.step()
        self.assertTrue(all(torch.isfinite(p).all() and not torch.equal(p, before[n]) for n, p in scorer.named_parameters()))
        clone = self.make_model()
        clone.load_state_dict(scorer.state_dict())
        restored = build_optimizer(clone, cfg)
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for network in (scorer, clone):
            for p in network.parameters(): p.grad = torch.ones_like(p)
        opt.step(); restored.step()
        for p, q in zip(scorer.parameters(), clone.parameters()): torch.testing.assert_close(p, q)
        bad = copy.deepcopy(opt.state_dict()); bad["names"]["muon"] = []
        with self.assertRaises(ValueError): restored.load_state_dict(bad)
        self.assertIsInstance(build_optimizer(scorer, {**cfg, "optimizer": "adamw"}), torch.optim.AdamW)

if __name__ == "__main__": unittest.main()
