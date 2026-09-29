#!/usr/bin/env bash
#SBATCH --job-name=csd-bfcl-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
ARTIFACT_DIR="$5"
STARTED_MARKER="$RECEIPT_PATH.started.json"
REVISION="$6"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize prepare-bfcl "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
job_run uv run --project "$PROJECT_DIR" --no-sync csd prepare-bfcl \
  --output-dir "$ARTIFACT_DIR" --revision "$REVISION" --started-marker "$STARTED_MARKER"
