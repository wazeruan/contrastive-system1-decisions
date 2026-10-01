#!/usr/bin/env bash
#SBATCH --job-name=csd-external-prep
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
CONFIG="$5"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize external-prepare "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
cd "$PROJECT_DIR"
job_run uv run --project "$PROJECT_DIR" --no-sync csd external-prepare \
  --config "$CONFIG" --output-dir "$RUN_DIR" --started-marker "$RECEIPT_PATH.started.json"
