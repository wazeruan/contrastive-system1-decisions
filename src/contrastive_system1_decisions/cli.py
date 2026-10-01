"""Command-line entry points for preparation, training, calibration, and scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .calibration import fit_temperature
from .data import download_and_prepare_xlam, prepare_xlam_records
from .evaluation import evaluate_bfcl, evaluate_jsonl, prepare_bfcl
from .model_cache import prepare_model
from .runtime import preflight_gpu, write_started_marker
from .training import train


def _write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _preflight(require_h100: bool, min_memory_gib: float,
               gpu_profile: str | Path | None = None,
               gpu_profile_sha256: str | None = None) -> dict[str, Any]:
    return preflight_gpu(
        require_h100=require_h100 and gpu_profile is None,
        min_memory_gib=min_memory_gib,
        gpu_profile=gpu_profile,
        gpu_profile_sha256=gpu_profile_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csd", description="Contrastive structured-action selection research prototype")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-xlam", help="download and normalize gated APIGen/xLAM data")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed/xlam"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--input-json", type=Path, help="optional local APIGen JSON file")
    prepare.add_argument("--bm25-negatives", type=int, default=0,
                         help="append this many BM25-retrieved weak negatives per training example")
    prepare.add_argument("--started-marker", type=Path)

    prepare_benchmark = commands.add_parser("prepare-bfcl", help="download and normalize BFCL selector categories")
    prepare_benchmark.add_argument("--output-dir", type=Path, default=Path("data/benchmark/bfcl"))
    prepare_benchmark.add_argument("--revision", default="main")
    prepare_benchmark.add_argument("--started-marker", type=Path)

    cache = commands.add_parser("prepare-model", help="download a pinned model/tokenizer in a CPU allocation")
    cache.add_argument("--config", type=Path, required=True)
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--started-marker", type=Path)
    cache.add_argument("--run-dir", type=Path)

    train_command = commands.add_parser("train", help="train a candidate-set contrastive selector")
    train_command.add_argument("--config", type=Path, required=True)
    train_command.add_argument("--data-dir", type=Path, required=True)
    train_command.add_argument("--run-dir", type=Path, required=True)
    train_command.add_argument("--resume", action="store_true")
    train_command.add_argument("--started-marker", type=Path)
    train_command.add_argument("--gpu-profile", type=Path)
    train_command.add_argument("--gpu-profile-sha256")

    calibrate = commands.add_parser("calibrate", help="fit one temperature on the calibration split")
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--data", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--batch-size", type=int, default=32)
    calibrate.add_argument("--require-h100", action="store_true")
    calibrate.add_argument("--started-marker", type=Path)
    calibrate.add_argument("--gpu-profile", type=Path)
    calibrate.add_argument("--gpu-profile-sha256")

    evaluate = commands.add_parser("evaluate", help="score a prepared JSONL selector split")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--require-h100", action="store_true")
    evaluate.add_argument("--started-marker", type=Path)
    evaluate.add_argument("--gpu-profile", type=Path)
    evaluate.add_argument("--gpu-profile-sha256")

    benchmark = commands.add_parser("evaluate-bfcl", help="score the custom BFCL tool-selection slice")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--data-dir", type=Path, required=True)
    benchmark.add_argument("--category", choices=("multiple", "live_multiple"), required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--calibration", type=Path)
    benchmark.add_argument("--batch-size", type=int, default=32)
    benchmark.add_argument("--require-h100", action="store_true")
    benchmark.add_argument("--started-marker", type=Path)
    benchmark.add_argument("--gpu-profile", type=Path)
    benchmark.add_argument("--gpu-profile-sha256")

    external_prepare = commands.add_parser("external-prepare", help="freeze external selector datasets")
    external_prepare.add_argument("--config", type=Path, required=True)
    external_prepare.add_argument("--output-dir", type=Path, required=True)
    external_prepare.add_argument("--started-marker", type=Path)
    external_evaluate = commands.add_parser("external-evaluate", help="compare all checkpoints on frozen external data")
    external_evaluate.add_argument("--models", type=Path, required=True)
    external_evaluate.add_argument("--data-dir", type=Path, required=True)
    external_evaluate.add_argument("--output-dir", type=Path, required=True)
    external_evaluate.add_argument("--batch-size", type=int, default=4)
    external_evaluate.add_argument("--started-marker", type=Path)
    external_evaluate.add_argument("--gpu-profile", type=Path)
    external_evaluate.add_argument("--gpu-profile-sha256")

    preflight = commands.add_parser("preflight", help="check the allocated accelerator")
    preflight.add_argument("--require-h100", action="store_true")
    preflight.add_argument("--min-memory-gib", type=float, default=75.0)
    preflight.add_argument("--gpu-profile", type=Path)
    preflight.add_argument("--gpu-profile-sha256")
    preflight.add_argument("--write-marker", type=Path)
    preflight.add_argument("--run-dir", type=Path)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-xlam":
        if args.bm25_negatives < 0:
            raise ValueError("--bm25-negatives must be zero or greater")
        if args.started_marker:
            write_started_marker(args.started_marker, args.output_dir, "prepare-xlam")
        if args.input_json:
            raw = json.loads(args.input_json.read_text(encoding="utf-8"))
            records = raw if isinstance(raw, list) else list(raw.values())
            result = prepare_xlam_records(records, args.output_dir, args.seed, args.bm25_negatives)
        else:
            result = download_and_prepare_xlam(args.output_dir, args.seed, args.bm25_negatives)
    elif args.command == "prepare-bfcl":
        if args.started_marker:
            write_started_marker(args.started_marker, args.output_dir, "prepare-bfcl")
        result = prepare_bfcl(args.output_dir, args.revision)
    elif args.command == "prepare-model":
        result = prepare_model(args.config, args.manifest, args.started_marker, args.run_dir)
    elif args.command == "train":
        result = train(args.config, args.data_dir, args.run_dir, args.resume, args.started_marker,
                       args.gpu_profile, args.gpu_profile_sha256)
    elif args.command == "calibrate":
        result = fit_temperature(args.checkpoint, args.data, args.output, args.batch_size,
                                 args.started_marker, args.require_h100, args.gpu_profile,
                                 args.gpu_profile_sha256)
    elif args.command == "evaluate":
        result = evaluate_jsonl(args.checkpoint, args.data, args.output, args.calibration, args.batch_size,
                                args.started_marker, args.require_h100, args.gpu_profile,
                                args.gpu_profile_sha256)
    elif args.command == "evaluate-bfcl":
        result = evaluate_bfcl(
            args.checkpoint, args.data_dir, args.category, args.output, args.calibration, args.batch_size,
            args.started_marker, args.require_h100, args.gpu_profile, args.gpu_profile_sha256
        )
    elif args.command == "external-prepare":
        from .external_data import prepare_external
        if args.started_marker:
            write_started_marker(args.started_marker, args.output_dir, "external-prepare")
        result = prepare_external(args.output_dir, args.config)
    elif args.command == "external-evaluate":
        from .external_evaluation import evaluate_external
        result = evaluate_external(args.models, args.data_dir, args.output_dir, args.batch_size,
                                   args.started_marker, args.gpu_profile, args.gpu_profile_sha256)
    elif args.command == "preflight":
        result = _preflight(args.require_h100, args.min_memory_gib, args.gpu_profile,
                            args.gpu_profile_sha256)
        if args.write_marker:
            result = write_started_marker(
                args.write_marker, args.run_dir or args.write_marker.parent, "preflight",
                require_h100=args.require_h100 and args.gpu_profile is None,
                min_memory_gib=args.min_memory_gib, gpu_profile=args.gpu_profile,
                gpu_profile_sha256=args.gpu_profile_sha256,
            )
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
