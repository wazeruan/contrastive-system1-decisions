#!/usr/bin/env bash
#SBATCH --job-name=csd-model-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
CONFIG="$5"
MODEL_MANIFEST="${6:-$PROJECT_DIR/models/model-cache-manifest.json}"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize prepare-model "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
job_run uv run --project "$PROJECT_DIR" --no-sync csd prepare-model \
  --config "$CONFIG" --manifest "$MODEL_MANIFEST" \
  --started-marker "$RECEIPT_PATH.started.json" --run-dir "$RUN_DIR"
