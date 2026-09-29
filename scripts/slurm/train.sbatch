#!/usr/bin/env bash
#SBATCH --job-name=csd-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
CONFIG="$5"
DATA_DIR="$6"
RESUME="$7"
STARTED_MARKER="$RECEIPT_PATH.started.json"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize train "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
command=(uv run --project "$PROJECT_DIR" --no-sync csd train
  --config "$CONFIG" --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --started-marker "$STARTED_MARKER")
if [[ "$RESUME" == "1" ]]; then command+=(--resume); fi
job_run "${command[@]}"
