#!/usr/bin/env bash

set -Eeuo pipefail

APP_PID=""
APP_STAGE=""
# Batch entry points set this before sourcing the shared functions.
RECEIPT_PATH="${RECEIPT_PATH:-}"
RUN_PATH=""
PROJECT_PATH=""
SIGNAL_SENT=""

receipt_update() {
  local state="$1"
  shift
  python3 "$PROJECT_PATH/scripts/receipt.py" update \
    --path "$RECEIPT_PATH" --state "$state" --job-id "${SLURM_JOB_ID:-unknown}" "$@"
}

job_initialize() {
  APP_STAGE="$1"
  PROJECT_PATH="$2"
  RECEIPT_PATH="$3"
  RUN_PATH="$4"
  if [[ -n "${CSD_ENV_SCRIPT:-}" ]]; then
    [[ -r "$CSD_ENV_SCRIPT" ]] || { echo "cluster environment script is not readable: $CSD_ENV_SCRIPT" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$CSD_ENV_SCRIPT"
  fi
  if [[ "$APP_STAGE" != "prepare-xlam" ]]; then unset HF_TOKEN; fi
  export PYTHONUNBUFFERED=1
  export HF_HOME="${HF_HOME:-$PROJECT_PATH/hf_cache}"
  export CSD_MODEL_CACHE_MANIFEST="${CSD_MODEL_CACHE_MANIFEST:-$PROJECT_PATH/models/model-cache-manifest.json}"
  [[ "$CSD_MODEL_CACHE_MANIFEST" == /* ]] || CSD_MODEL_CACHE_MANIFEST="$PROJECT_PATH/$CSD_MODEL_CACHE_MANIFEST"
  export CSD_MODEL_CACHE_MANIFEST
  case "$APP_STAGE" in
    train|calibrate|evaluate|evaluate-bfcl)
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      ;;
  esac
  receipt_update ALLOCATED --host "$(hostname)" \
    --details-json "{\"compute_allocated_at_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
  trap 'job_exit $?' EXIT
  trap 'forward_signal TERM' TERM
  trap 'forward_signal INT' INT
  trap 'forward_signal USR1' USR1
}

forward_signal() {
  local signal_name="$1"
  SIGNAL_SENT="$signal_name"
  if [[ -n "$APP_PID" ]]; then
    kill -s "$signal_name" "$APP_PID" 2>/dev/null || true
  fi
}

job_exit() {
  local exit_code="$1"
  trap - EXIT
  if (( exit_code == 0 )); then
    if [[ ! -f "$RECEIPT_PATH.started.json" ]]; then
      receipt_update FAILED --reason "application exited successfully without a valid STARTED marker"
      exit 70
    fi
    local app_state=""
    if [[ -f "$RUN_PATH/status.json" ]]; then
      app_state="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("state", "") if str(d.get("job_id", "")) == sys.argv[2] else "")' "$RUN_PATH/status.json" "${SLURM_JOB_ID:-}" 2>/dev/null || true)"
    fi
    if [[ "$app_state" == "PREEMPTED" ]]; then
      receipt_update PREEMPTED --reason "training checkpointed after signal ${SIGNAL_SENT:-USR1}"
      exit_code=75
    else
      receipt_update SUCCEEDED
    fi
  else
    receipt_update FAILED --reason "batch process exited with code $exit_code${SIGNAL_SENT:+ after $SIGNAL_SENT}"
  fi
  exit "$exit_code"
}

job_run() {
  local mark_on_launch=0
  local mark_workflow=""
  if [[ "${1:-}" == "--mark-on-launch" ]]; then
    mark_on_launch=1
    mark_workflow="$2"
    shift 2
  fi
  local marker="$RECEIPT_PATH.started.json"
  if ! command -v srun >/dev/null 2>&1; then
    echo "srun is required inside this Slurm allocation" >&2
    return 127
  fi
  if [[ "$mark_on_launch" == 1 ]]; then
    CSD_MARKER_PATH="$marker" CSD_RUN_PATH="$RUN_PATH" \
      CSD_MARKER_WORKFLOW="$mark_workflow" CSD_PROJECT_PATH="$PROJECT_PATH" \
      srun --ntasks=1 --unbuffered --export=ALL bash -c '
        source "$CSD_PROJECT_PATH/scripts/job_common.sh"
        command -v "$1" >/dev/null 2>&1 || {
          echo "application executable is unavailable on compute node: $1" >&2
          exit 127
        }
        job_write_marker "$CSD_MARKER_WORKFLOW" "$CSD_MARKER_PATH" "$CSD_RUN_PATH"
        exec "$@"
      ' csd-launch "$@" &
  else
    srun --ntasks=1 --unbuffered "$@" &
  fi
  APP_PID=$!
  local marked=0
  local start_epoch
  start_epoch="$(date +%s)"
  while [[ -r "/proc/$APP_PID/stat" ]] && [[ "$(awk '{print $3}' "/proc/$APP_PID/stat")" != "Z" ]]; do
    if [[ -f "$marker" && "$marked" -eq 0 ]]; then
      receipt_update STARTED --marker "$marker" --host "$(hostname)"
      marked=1
    fi
    sleep 2 || true
    if (( marked == 0 )) && (( $(date +%s) - start_epoch > 900 )); then
      receipt_update ALLOCATED_STARTUP --reason "application marker not written within 15 minutes"
      marked=2
    fi
  done
  local exit_code=0
  wait "$APP_PID" || exit_code=$?
  APP_PID=""
  if [[ -f "$marker" && "$marked" -eq 0 ]]; then
    receipt_update STARTED --marker "$marker" --host "$(hostname)"
    marked=1
  fi
  return "$exit_code"
}

job_write_marker() {
  local workflow="$1"
  local marker="${2:-$RECEIPT_PATH.started.json}"
  local run_path="${3:-$RUN_PATH}"
  python3 - "$marker" "$run_path" "$workflow" "${SLURM_JOB_ID:-unknown}" "$(hostname)" <<'PY'
import json, os, sys, time
from pathlib import Path
path, run_dir, workflow, job_id, host = sys.argv[1:]
destination = Path(path)
destination.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "state": "STARTED",
    "workflow": workflow,
    "job_id": job_id,
    "host": host,
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_dir": str(Path(run_dir).resolve()),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
}
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, destination)
PY
}
