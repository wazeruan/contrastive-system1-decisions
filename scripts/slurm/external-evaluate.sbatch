#!/usr/bin/env bash
#SBATCH --job-name=csd-external-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
MODELS="$5"
DATA_DIR="$6"
GPU_PROFILE="$7"
GPU_PROFILE_SHA256="$8"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize external-evaluate "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT_DIR"
job_run uv run --project "$PROJECT_DIR" --no-sync csd external-evaluate \
  --models "$MODELS" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --started-marker "$RECEIPT_PATH.started.json" --gpu-profile "$GPU_PROFILE" \
  --gpu-profile-sha256 "$GPU_PROFILE_SHA256"
