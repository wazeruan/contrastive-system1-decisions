#!/usr/bin/env bash
#SBATCH --job-name=csd-xlam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -Eeuo pipefail
PROJECT_DIR="$1"
RECEIPT_PATH="$2"
SUBMISSION_TOKEN="$3"
RUN_DIR="$4"
ARTIFACT_DIR="$5"
STARTED_MARKER="$RECEIPT_PATH.started.json"
BM25_NEGATIVES="$6"
INPUT_JSON="${7:-}"
source "$PROJECT_DIR/scripts/job_common.sh"
job_initialize prepare-xlam "$PROJECT_DIR" "$RECEIPT_PATH" "$RUN_DIR"
command -v uv >/dev/null 2>&1 || { echo "uv is not available on this cluster" >&2; exit 127; }
command=(uv run --project "$PROJECT_DIR" --no-sync csd prepare-xlam
  --output-dir "$ARTIFACT_DIR" --bm25-negatives "$BM25_NEGATIVES" --started-marker "$STARTED_MARKER")
if [[ -n "$INPUT_JSON" ]]; then command+=(--input-json "$INPUT_JSON"); fi
job_run "${command[@]}"
