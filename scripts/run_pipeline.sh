#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ACCOUNT="" PARTITION="" GPU_RESOURCE="" ENV_SCRIPT=""
CONFIG="" CONFIG_SET=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_pipeline.sh --account ACCOUNT [options]

Submits one baseline dependency chain:
  setup -> xLAM/BFCL/model preparation -> H100 preflight -> train
        -> calibration -> xLAM test evaluation and BFCL live_multiple evaluation

HF_TOKEN must be set in the environment for the gated xLAM download. It is
forwarded only to that preparation stage. The command queues jobs with afterok
dependencies; Slurm runs each stage only after its required predecessors succeed.

Options:
  --account ACCOUNT       Required Slurm account
  --project-dir PATH      Project checkout (defaults to this checkout)
  --config PATH           Training config (defaults to shared-heads-h100.json)
  --env-script PATH       Shared cluster environment setup script
  --partition NAME        Optional partition override for all stages
  --gpu-resource SPEC     Optional full GRES value for GPU stages
  --help                  Show this help

The default config is the shared-heads seed-42 baseline. The pipeline submits
training, calibration, xLAM test evaluation, and the BFCL live_multiple slice.
USAGE
}

die() { echo "run_pipeline: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --account|--project-dir|--config|--env-script|--partition|--gpu-resource)
      (($# >= 2)) || die "$1 needs a value"
      key="$1"; value="$2"; shift 2
      case "$key" in
        --account) ACCOUNT="$value" ;;
        --project-dir) PROJECT_DIR="$value" ;;
        --config) CONFIG="$value"; CONFIG_SET=1 ;;
        --env-script) ENV_SCRIPT="$value" ;;
        --partition) PARTITION="$value" ;;
        --gpu-resource) GPU_RESOURCE="$value" ;;
      esac
      ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$ACCOUNT" && "$ACCOUNT" =~ ^[[:alnum:]_.-]+$ ]] || die "valid --account is required"
[[ -z "$PARTITION" || "$PARTITION" =~ ^[[:alnum:]_.-]+$ ]] || die "invalid --partition"
[[ -z "$GPU_RESOURCE" || "$GPU_RESOURCE" =~ ^[[:alnum:]_.:=+-]+$ ]] || die "invalid GPU resource syntax"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN must be set in the environment for gated xLAM data preparation"

PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd -P)" || die "project directory does not exist"
[[ -f "$PROJECT_DIR/scripts/submit.sh" && -f "$PROJECT_DIR/pyproject.toml" ]] || die "not a project checkout"
if [[ "$CONFIG_SET" == 0 ]]; then
  CONFIG="$PROJECT_DIR/configs/shared-heads-h100.json"
elif [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_DIR/$CONFIG"
fi
[[ -f "$CONFIG" ]] || die "training config does not exist: $CONFIG"
if [[ -n "$ENV_SCRIPT" ]]; then
  [[ "$ENV_SCRIPT" == /* ]] || ENV_SCRIPT="$PROJECT_DIR/$ENV_SCRIPT"
  [[ -r "$ENV_SCRIPT" ]] || die "environment script is not readable: $ENV_SCRIPT"
fi

SUBMIT="$PROJECT_DIR/scripts/submit.sh"
base_args=(--project-dir "$PROJECT_DIR" --account "$ACCOUNT")
if [[ -n "$ENV_SCRIPT" ]]; then base_args+=(--env-script "$ENV_SCRIPT"); fi
if [[ -n "$PARTITION" ]]; then base_args+=(--partition "$PARTITION"); fi

LAST_JOB_ID="" LAST_RUN_DIR="" LAST_RECEIPT="" UNVERIFIED_STAGES=""
submit_stage() {
  local label="$1" output rc line receipt_state=""
  shift
  echo "Submitting pipeline stage: $label"
  if output="$("$@" 2>&1)"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$output"
  LAST_JOB_ID="" LAST_RUN_DIR="" LAST_RECEIPT=""
  while IFS= read -r line; do
    case "$line" in
      JOB_ID=*) LAST_JOB_ID="${line#JOB_ID=}" ;;
      RUN_DIR=*) LAST_RUN_DIR="${line#RUN_DIR=}" ;;
      RECEIPT=*) LAST_RECEIPT="${line#RECEIPT=}" ;;
    esac
  done <<< "$output"
  if (( rc != 0 )); then
    if [[ -n "$LAST_RECEIPT" && -f "$LAST_RECEIPT" ]]; then
      receipt_state="$(python3 - "$LAST_RECEIPT" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))
except (OSError, ValueError):
    pass
PY
)"
    fi
    if [[ "$receipt_state" == RUNNING_STARTUP_UNVERIFIED ]]; then
      [[ "$LAST_JOB_ID" =~ ^[0-9]+$ && -n "$LAST_RECEIPT" ]] || die "$label startup is unverified and its submission details are missing"
      UNVERIFIED_STAGES+="${UNVERIFIED_STAGES:+; }$label (job $LAST_JOB_ID; $LAST_RECEIPT)"
      echo "$label is confirmed RUNNING, but application startup is unverified. Later jobs will depend on its successful completion." >&2
    else
      echo "Pipeline stopped at $label; inspect the stage receipt before resubmitting." >&2
      return "$rc"
    fi
  fi
  [[ "$LAST_JOB_ID" =~ ^[0-9]+$ ]] || die "$label returned no numeric JOB_ID"
  [[ -n "$LAST_RUN_DIR" && -n "$LAST_RECEIPT" ]] || die "$label did not report its run directory and receipt"
  printf 'PIPELINE_STAGE=%s JOB_ID=%s RUN_DIR=%s RECEIPT=%s\n' \
    "$label" "$LAST_JOB_ID" "$LAST_RUN_DIR" "$LAST_RECEIPT"
}

submit_gpu_stage() {
  local label="$1"
  shift
  if [[ -n "$GPU_RESOURCE" ]]; then
    submit_stage "$label" "$@" --gpu-resource "$GPU_RESOURCE"
  else
    submit_stage "$label" "$@"
  fi
}

echo "Queueing the baseline pipeline under account $ACCOUNT."

submit_stage setup "$SUBMIT" "${base_args[@]}" --stage setup
SETUP_JOB_ID="$LAST_JOB_ID"

# This is the only stage that needs the gated dataset credential.
submit_stage prepare-xlam "$SUBMIT" "${base_args[@]}" --stage prepare-xlam \
  --dependency "$SETUP_JOB_ID" --bm25-negatives 0
XLAM_JOB_ID="$LAST_JOB_ID" XLAM_DIR="$LAST_RUN_DIR"
unset HF_TOKEN

submit_stage prepare-bfcl "$SUBMIT" "${base_args[@]}" --stage prepare-bfcl \
  --dependency "$SETUP_JOB_ID"
BFCL_JOB_ID="$LAST_JOB_ID" BFCL_DIR="$LAST_RUN_DIR"

submit_stage prepare-model "$SUBMIT" "${base_args[@]}" --stage prepare-model \
  --config "$CONFIG" --dependency "$SETUP_JOB_ID"
MODEL_JOB_ID="$LAST_JOB_ID"

submit_gpu_stage preflight "$SUBMIT" "${base_args[@]}" \
  --stage preflight --dependency "$MODEL_JOB_ID"
PREFLIGHT_JOB_ID="$LAST_JOB_ID"

TRAIN_DEPENDENCIES="$XLAM_JOB_ID,$MODEL_JOB_ID,$PREFLIGHT_JOB_ID"
submit_gpu_stage train "$SUBMIT" "${base_args[@]}" --stage train \
  --config "$CONFIG" --data-dir "$XLAM_DIR" --dependency "$TRAIN_DEPENDENCIES"
TRAIN_JOB_ID="$LAST_JOB_ID" TRAIN_DIR="$LAST_RUN_DIR"

submit_gpu_stage calibrate "$SUBMIT" "${base_args[@]}" --stage calibrate \
  --checkpoint "$TRAIN_DIR/checkpoints/best.pt" --data "$XLAM_DIR/calibration.jsonl" \
  --dependency "$TRAIN_JOB_ID"
CALIBRATION_JOB_ID="$LAST_JOB_ID" CALIBRATION_DIR="$LAST_RUN_DIR"

submit_gpu_stage evaluate-xlam "$SUBMIT" "${base_args[@]}" --stage evaluate \
  --checkpoint "$TRAIN_DIR/checkpoints/best.pt" --data "$XLAM_DIR/test.jsonl" \
  --calibration "$CALIBRATION_DIR/calibration.json" --dependency "$CALIBRATION_JOB_ID"
XLAM_EVALUATION_JOB_ID="$LAST_JOB_ID" XLAM_EVALUATION_DIR="$LAST_RUN_DIR"

BENCHMARK_DEPENDENCIES="$BFCL_JOB_ID,$CALIBRATION_JOB_ID"
submit_gpu_stage evaluate-bfcl "$SUBMIT" "${base_args[@]}" --stage benchmark \
  --checkpoint "$TRAIN_DIR/checkpoints/best.pt" --data-dir "$BFCL_DIR" \
  --category live_multiple --calibration "$CALIBRATION_DIR/calibration.json" \
  --dependency "$BENCHMARK_DEPENDENCIES"
BFCL_EVALUATION_JOB_ID="$LAST_JOB_ID" BFCL_EVALUATION_DIR="$LAST_RUN_DIR"

cat <<SUMMARY

Pipeline queued with afterok dependencies. Jobs run only after their required predecessors succeed.
  setup:          $SETUP_JOB_ID
  xLAM download:  $XLAM_JOB_ID  ($XLAM_DIR)
  BFCL download:  $BFCL_JOB_ID  ($BFCL_DIR)
  model download: $MODEL_JOB_ID
  H100 preflight: $PREFLIGHT_JOB_ID
  training:       $TRAIN_JOB_ID  ($TRAIN_DIR)
  calibration:    $CALIBRATION_JOB_ID  ($CALIBRATION_DIR)
  xLAM evaluation:$XLAM_EVALUATION_JOB_ID  ($XLAM_EVALUATION_DIR)
  BFCL evaluation:$BFCL_EVALUATION_JOB_ID  ($BFCL_EVALUATION_DIR)
SUMMARY

if [[ -n "$UNVERIFIED_STAGES" ]]; then
  echo "Startup remains unverified for: $UNVERIFIED_STAGES" >&2
  echo "Inspect those receipts and Slurm state; dependent jobs are held by afterok until success." >&2
fi
