#!/usr/bin/env bash
#SBATCH --job-name=csd-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
GPU_PROFILE="$5"
GPU_PROFILE_SHA256="$6"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize preflight "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
job_run uv run --project "$PROJECT_DIR" --no-sync csd preflight \
  --gpu-profile "$GPU_PROFILE" --gpu-profile-sha256 "$GPU_PROFILE_SHA256" \
  --write-marker "$RECEIPT_PATH.started.json" --run-dir "$RUN_DIR"
