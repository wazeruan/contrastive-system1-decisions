"""Synthetic, offline adapter/runner regressions; no network or scheduler jobs."""
import importlib.util
import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from contrastive_system1_decisions import external_data as data
from contrastive_system1_decisions import external_evaluation as evaluation
from contrastive_system1_decisions.data import write_jsonl

TOOLS = [{"name": "a", "description": "first"}, {"name": "b", "description": "second"}]
def wrow(query="unique"):
    return {"question": query, "tools": TOOLS, "correct_answer": "tool_call", "answers": {"tool_call": {"name": "a"}}}
def trow(answer="[a(x='value, ()')]"):
    return {"system": "Here is a list of functions in JSON format that you can invoke: " + json.dumps(TOOLS),
            "conversations": [{"from": "user", "value": "unique"}, {"from": "assistant", "value": answer}]}

class ExternalTests(unittest.TestCase):
    def test_adapter_targets_and_rejections(self):
        for example in [data.when2call_example(wrow(), 0), data.toolace_example(trow(), 0)]:
            self.assertEqual([x.candidate_id for x in example.candidates], ["a", "b"])
            self.assertEqual(example.target_weights, [1., 0.])
        for row in [dict(wrow(), correct_answer="no_tool"), dict(wrow(), tools=TOOLS[:1])]:
            with self.assertRaises(ValueError): data.when2call_example(row, 0)
        for answer in ["[]", "[a(), b()]", "natural language", "[a(),]"]:
            with self.assertRaises(ValueError): data.toolace_example(trow(answer), 0)
        self.assertEqual(data.toolace_example(trow("[a(), a() ]"), 0).target_weights, [1., 0.])

    def test_preparation_pins_hashes_excludes_overlap_dedupes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); xlam = root / "xlam"; xlam.mkdir()
            for split in ["train", "validation", "calibration", "test"]:
                write_jsonl(xlam / (split + ".jsonl"), [data.when2call_example(wrow("Existing Query"), 0)])
            config = root / "config.json"; config.write_text(json.dumps({"xlam_dir": str(xlam)}))
            source = root / "source"; source.mkdir()
            for filename in data.SOURCES["when2call"][1]:
                path = source / filename; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(json.dumps(row) for row in [wrow(" existing  QUERY "), wrow("new query"), wrow("NEW QUERY")]))
            toolace = source / "data.json"; toolace.write_text(json.dumps([trow()]))
            pinned = []
            def download(**kwargs):
                pinned.append(kwargs["revision"])
                return str(source / kwargs["filename"])
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", side_effect=download):
                api.return_value.dataset_info.return_value = SimpleNamespace(sha="abc123")
                result = data.prepare_external(root / "out", config)
            self.assertEqual(pinned, ["abc123"] * 3)
            first = result["datasets"][0]
            self.assertEqual(first["examples"], 1)
            self.assertEqual(first["counts"]["exact_xlam_query_overlap"], 2)
            self.assertEqual(first["counts"]["duplicate_query_within_dataset"], 3)
            self.assertEqual(len(result["xlam_sha256"]), 4)
            for row in result["datasets"]:
                self.assertEqual(row["sha256"], hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest())
                self.assertEqual(row["revision"], "abc123")

    def fixtures(self, root):
        sample = data.when2call_example(wrow(), 0)
        dataset = root / "sample.jsonl"
        write_jsonl(dataset, [sample])
        manifest = {"datasets": [{"name": "sample", "path": str(dataset), "examples": 1,
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        rows = []
        for name, temp in [("shared", 1.), ("muon", 2.), ("adamw", 3.)]:
            checkpoint = root / (name + ".pt"); checkpoint.write_bytes(b"stub")
            calibration = root / (name + ".json"); calibration.write_text(json.dumps({"temperature": temp, "checkpoint": str(checkpoint)}))
            rows.append({"name": name, "checkpoint": str(checkpoint), "calibration": str(calibration)})
        models = root / "models.json"; models.write_text(json.dumps({"models": rows}))
        return models, sample

    def test_all_models_identical_data_saved_temperatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, sample = self.fixtures(root)
            observed = []
            def logits(model, tokenizer, examples, config, device, batch):
                observed.append([x.example_id for x in examples])
                return [torch.tensor([2., 0.])], None
            with patch.object(evaluation, "load_model", return_value=(object(), object(), {"use_bf16": False})), patch.object(evaluation, "collect_logits", side_effect=logits), patch.object(torch.cuda, "is_available", return_value=False):
                result = evaluation.evaluate_external(models, root, root / "out")
            self.assertEqual(result["state"], "SUCCEEDED")
            self.assertEqual(observed, [[sample.example_id]] * 3)
            for name, temperature in [("shared", 1.), ("muon", 2.), ("adamw", 3.)]:
                metric = json.loads((root / "out" / name / "sample" / "metrics.json").read_text())
                self.assertEqual(metric["temperature"], temperature)
                self.assertTrue((root / "out" / name / "sample" / "predictions.json").is_file())
            self.assertTrue((root / "out" / "RESULTS.md").is_file())

    def test_integrity_checked_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, _ = self.fixtures(root)
            (root / "sample.jsonl").write_text("tampered")
            with patch.object(evaluation, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    evaluation.evaluate_external(models, root, root / "out")
                load.assert_not_called()

    def test_calibration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, _ = self.fixtures(root)
            (root / "shared.json").write_text(json.dumps({"temperature": 1., "checkpoint": "/wrong/best.pt"}))
            with patch.object(evaluation, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "calibration"):
                    evaluation.evaluate_external(models, root, root / "out")
                load.assert_not_called()

    def test_launcher_fake_submissions_cpu_and_gpu_resources(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("external_launcher_test", scripts / "external_suite.py")
            launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, _ = self.fixtures(root)
            configs = root / "datasets.json"; configs.write_text(json.dumps({"xlam_dir": str(root)}))
            for split in ["train", "validation", "calibration", "test"]:
                (root / (split + ".jsonl")).write_text("")
            submitted = []
            def fake_run(command, **kwargs):
                submitted.append(command)
                stage = command[command.index("--stage") + 1]
                output = command[command.index("--output-dir") + 1]
                receipt = root / (stage + "-receipt.json")
                receipt.write_text(json.dumps({"job_id": "123", "state": "SUBMITTED", "run_dir": output}))
                return SimpleNamespace(returncode=0, stdout=f"JOB_ID=123\nRUN_DIR={output}\nRECEIPT={receipt}\n", stderr="")
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", side_effect=fake_run), patch.object(sys, "argv", ["external_suite", "run", "--account", "test", "--models", str(models), "--datasets", str(configs)]):
                launcher.main()
            self.assertEqual(len(submitted), 2)
            self.assertNotIn("--resource-profile", submitted[0])
            self.assertIn("--resource-profile", submitted[1])
            self.assertEqual(submitted[1][submitted[1].index("--dependency") + 1], "123")
            pointer = json.loads((root / "runs/external-evaluation-latest.json").read_text())
            manifest = json.loads((Path(pointer["suite"]) / "suite.json").read_text())
            self.assertEqual(manifest["state"], "QUEUED")

    def test_launcher_reuse_requires_verified_accounting(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("external_reuse_test", scripts / "external_suite.py")
            launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, _ = self.fixtures(root)
            configs = root / "datasets.json"; configs.write_text(json.dumps({"xlam_dir": str(root)}))
            for split in ["train", "validation", "calibration", "test"]:
                (root / (split + ".jsonl")).write_text("")
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123|FAILED|1:0\n", stderr="")) as run, patch.object(sys, "argv", ["external_suite", "run", "--account", "test", "--models", str(models), "--datasets", str(configs), "--reuse-data-dir", str(root), "--prepare-job-id", "123"]):
                with self.assertRaisesRegex(ValueError, "COMPLETED"):
                    launcher.main()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][0], "sacct")
            pointer = json.loads((root / "runs/external-evaluation-latest.json").read_text())
            manifest = json.loads((Path(pointer["suite"]) / "suite.json").read_text())
            self.assertEqual(manifest["state"], "STOPPED")
            receipt_dir = root / "logs/submissions/external-prepare-test"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "receipt.json").write_text(json.dumps({"state": "SUCCEEDED", "job_id": "123", "run_dir": str(root)}))
            commands = []
            def recovered(command, **kwargs):
                commands.append(command)
                if command[0] == "sacct":
                    return SimpleNamespace(returncode=0, stdout="123|COMPLETED|0:0\n", stderr="")
                receipt = root / "eval-receipt.json"
                output = command[command.index("--output-dir") + 1]
                receipt.write_text(json.dumps({"state": "SUBMITTED", "job_id": "456", "run_dir": output}))
                return SimpleNamespace(returncode=0, stdout=f"JOB_ID=456\nRUN_DIR={output}\nRECEIPT={receipt}\n", stderr="")
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", side_effect=recovered), patch.object(sys, "argv", ["external_suite", "run", "--account", "test", "--models", str(models), "--datasets", str(configs), "--reuse-data-dir", str(root), "--prepare-job-id", "123"]):
                launcher.main()
            self.assertEqual(len(commands), 2)
            self.assertNotIn("--dependency", commands[1])
            self.assertEqual(commands[1][commands[1].index("--stage") + 1], "external-evaluate")
            def active(command, **kwargs):
                if command[0] == "squeue":
                    return SimpleNamespace(returncode=0, stdout="456|RUNNING\n", stderr="")
                self.assertEqual(command[0], "sacct")
                return SimpleNamespace(returncode=0, stdout="123|COMPLETED|0:0\n", stderr="")
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", side_effect=active) as run, patch.object(sys, "argv", ["external_suite", "run", "--account", "test", "--models", str(models), "--datasets", str(configs), "--reuse-data-dir", str(root), "--prepare-job-id", "123"]):
                with self.assertRaisesRegex(ValueError, "remains active"):
                    launcher.main()
                self.assertEqual(run.call_count, 2)

    def test_rejected_null_submission_retries_but_unknown_blocks(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("external_null_job_test", scripts / "external_suite.py")
            launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        finally:
            sys.path.pop(0)
        for state in ["SUBMISSION_REJECTED", "UNKNOWN/UNVERIFIED"]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); models, _ = self.fixtures(root)
                configs = root / "datasets.json"; configs.write_text(json.dumps({"xlam_dir": str(root)}))
                for split in ["train", "validation", "calibration", "test"]:
                    (root / (split + ".jsonl")).write_text("")
                receipt_dir = root / "logs/submissions/external-prepare-test"; receipt_dir.mkdir(parents=True)
                (receipt_dir / "receipt.json").write_text(json.dumps({"state": "SUCCEEDED", "job_id": "123", "run_dir": str(root)}))
                old = root / "runs/external-evaluation-old"; old.mkdir(parents=True)
                rejection = old / "receipt.json"; rejection.write_text(json.dumps({"state": state, "job_id": None}))
                (old / "suite.json").write_text(json.dumps({"prepare": {"run_dir": str(root)}, "evaluation": {"job_id": None, "receipt": str(rejection)}}))
                submitted = []
                def fake(command, **kwargs):
                    if command[0] == "sacct":
                        return SimpleNamespace(returncode=0, stdout="123|COMPLETED|0:0\n", stderr="")
                    submitted.append(command)
                    receipt = root / "new-receipt.json"
                    output = command[command.index("--output-dir") + 1]
                    receipt.write_text(json.dumps({"state": "SUBMITTED", "job_id": "789", "run_dir": output}))
                    return SimpleNamespace(returncode=0, stdout=f"JOB_ID=789\nRUN_DIR={output}\nRECEIPT={receipt}\n", stderr="")
                with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", side_effect=fake), patch.object(sys, "argv", ["external_suite", "run", "--account", "test", "--models", str(models), "--datasets", str(configs), "--reuse-data-dir", str(root), "--prepare-job-id", "123"]):
                    if state == "SUBMISSION_REJECTED":
                        launcher.main()
                        self.assertEqual(len(submitted), 1)
                    else:
                        with self.assertRaisesRegex(ValueError, "ambiguous"):
                            launcher.main()
                        self.assertEqual(submitted, [])

    def test_nonfinite_logits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models, _ = self.fixtures(root)
            with patch.object(evaluation, "load_model", return_value=(object(), object(), {"use_bf16": False})), patch.object(evaluation, "collect_logits", return_value=([torch.tensor([float('nan'), 0.])], None)), patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaises((FloatingPointError, ValueError)):
                    evaluation.evaluate_external(models, root, root / "out")
            self.assertEqual(json.loads((root / "out/results.json").read_text())["state"], "FAILED")

if __name__ == '__main__': unittest.main()
