#!/usr/bin/env bash
#SBATCH --job-name=csd-calibrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
CHECKPOINT="$5"
DATA_PATH="$6"
GPU_PROFILE="$7"
GPU_PROFILE_SHA256="$8"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize calibrate "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
[[ -s "$CHECKPOINT" ]] || { echo "required training checkpoint is missing or empty: $CHECKPOINT" >&2; exit 66; }
job_run uv run --project "$PROJECT_DIR" --no-sync csd calibrate \
  --checkpoint "$CHECKPOINT" --data "$DATA_PATH" --output "$RUN_DIR/calibration.json" \
  --started-marker "$RECEIPT_PATH.started.json" --gpu-profile "$GPU_PROFILE" \
  --gpu-profile-sha256 "$GPU_PROFILE_SHA256"
