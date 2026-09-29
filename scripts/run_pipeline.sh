#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ACCOUNT="" PARTITION="" GPU_RESOURCE="" ENV_SCRIPT=""
CONFIG="" CONFIG_SET=0
CONTINUE_AFTER_XLAM=0 EXISTING_SETUP_JOB_ID="" EXISTING_XLAM_JOB_ID="" EXISTING_XLAM_DIR=""
EXISTING_BFCL_JOB_ID="" EXISTING_BFCL_DIR=""
RESUME_AFTER_PREFLIGHT=0 EXISTING_MODEL_JOB_ID="" EXISTING_PREFLIGHT_JOB_ID=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_pipeline.sh --account ACCOUNT [options]

Submits one baseline dependency chain:
  setup -> xLAM/BFCL/model preparation -> H100 preflight -> train
        -> calibration -> xLAM test evaluation and BFCL live_multiple evaluation

HF_TOKEN must be set in the environment for the gated xLAM download. It is
forwarded only to that preparation stage. The command queues jobs with afterok
dependencies; Slurm runs each stage only after its required predecessors succeed.

To continue after an already completed xLAM preparation, pass
--continue-after-xlam, --setup-job-id, --xlam-job-id, and --xlam-dir. The script
validates the prepared files, refreshes the Python environment, and resumes with
BFCL/model preparation. Add --bfcl-job-id and --bfcl-dir to reuse completed BFCL data.
To resume after model preparation and H100 preflight, add --resume-after-preflight,
--model-job-id, and --preflight-job-id. This mode also requires the completed
BFCL data so no earlier pipeline stage needs to be submitted again. Use it only
when no training job was accepted in the previous attempt.

Options:
  --account ACCOUNT       Required Slurm account
  --project-dir PATH      Project checkout (defaults to this checkout)
  --config PATH           Training config (defaults to shared-heads-h100.json)
  --env-script PATH       Shared cluster environment setup script
  --partition NAME        Optional partition override for all stages
  --gpu-resource SPEC     Optional full GRES value for GPU stages
  --continue-after-xlam   Reuse an existing prepared xLAM dataset and jobs
  --setup-job-id ID       Successful setup job required by --continue-after-xlam
  --xlam-job-id ID        Successful xLAM preparation job required by --continue-after-xlam
  --xlam-dir PATH         Existing xLAM dataset directory required by --continue-after-xlam
  --bfcl-job-id ID        Optional successful BFCL preparation job to reuse
  --bfcl-dir PATH         Existing BFCL data directory required with --bfcl-job-id
  --resume-after-preflight Reuse existing model/preflight jobs and resume at training
  --model-job-id ID       Existing active or successful model preparation job
  --preflight-job-id ID   Existing active or successful H100 preflight job
  --help                  Show this help

The default config is the shared-heads seed-42 baseline. The pipeline submits
training, calibration, xLAM test evaluation, and the BFCL live_multiple slice.
USAGE
}

die() { echo "run_pipeline: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --account|--project-dir|--config|--env-script|--partition|--gpu-resource|--setup-job-id|--xlam-job-id|--xlam-dir|--bfcl-job-id|--bfcl-dir|--model-job-id|--preflight-job-id)
      (($# >= 2)) || die "$1 needs a value"
      key="$1"; value="$2"; shift 2
      case "$key" in
        --account) ACCOUNT="$value" ;;
        --project-dir) PROJECT_DIR="$value" ;;
        --config) CONFIG="$value"; CONFIG_SET=1 ;;
        --env-script) ENV_SCRIPT="$value" ;;
        --partition) PARTITION="$value" ;;
        --gpu-resource) GPU_RESOURCE="$value" ;;
        --setup-job-id) EXISTING_SETUP_JOB_ID="$value" ;;
        --xlam-job-id) EXISTING_XLAM_JOB_ID="$value" ;;
        --xlam-dir) EXISTING_XLAM_DIR="$value" ;;
        --bfcl-job-id) EXISTING_BFCL_JOB_ID="$value" ;;
        --bfcl-dir) EXISTING_BFCL_DIR="$value" ;;
        --model-job-id) EXISTING_MODEL_JOB_ID="$value" ;;
        --preflight-job-id) EXISTING_PREFLIGHT_JOB_ID="$value" ;;
      esac
      ;;
    --continue-after-xlam) CONTINUE_AFTER_XLAM=1; shift ;;
    --resume-after-preflight) RESUME_AFTER_PREFLIGHT=1; CONTINUE_AFTER_XLAM=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$ACCOUNT" && "$ACCOUNT" =~ ^[[:alnum:]_.-]+$ ]] || die "valid --account is required"
[[ -z "$PARTITION" || "$PARTITION" =~ ^[[:alnum:]_.-]+$ ]] || die "invalid --partition"
[[ -z "$GPU_RESOURCE" || "$GPU_RESOURCE" =~ ^[[:alnum:]_.:=+-]+$ ]] || die "invalid GPU resource syntax"
if [[ "$CONTINUE_AFTER_XLAM" == 0 ]]; then
  [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN must be set in the environment for gated xLAM data preparation"
  [[ -z "$EXISTING_SETUP_JOB_ID$EXISTING_XLAM_JOB_ID$EXISTING_XLAM_DIR$EXISTING_BFCL_JOB_ID$EXISTING_BFCL_DIR" ]] || \
    die "existing job/data options require --continue-after-xlam"
else
  [[ "$EXISTING_SETUP_JOB_ID" =~ ^[0-9]+$ ]] || die "--continue-after-xlam requires numeric --setup-job-id"
  [[ "$EXISTING_XLAM_JOB_ID" =~ ^[0-9]+$ ]] || die "--continue-after-xlam requires numeric --xlam-job-id"
  [[ -n "$EXISTING_XLAM_DIR" ]] || die "--continue-after-xlam requires --xlam-dir"
  [[ -z "$EXISTING_BFCL_JOB_ID" && -z "$EXISTING_BFCL_DIR" || \
     "$EXISTING_BFCL_JOB_ID" =~ ^[0-9]+$ && -n "$EXISTING_BFCL_DIR" ]] || \
    die "provide both --bfcl-job-id (numeric) and --bfcl-dir, or neither"
  unset HF_TOKEN
fi
if [[ "$RESUME_AFTER_PREFLIGHT" == 1 ]]; then
  [[ "$EXISTING_MODEL_JOB_ID" =~ ^[0-9]+$ ]] || die "--resume-after-preflight requires numeric --model-job-id"
  [[ "$EXISTING_PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]] || die "--resume-after-preflight requires numeric --preflight-job-id"
  [[ "$EXISTING_BFCL_JOB_ID" =~ ^[0-9]+$ && -n "$EXISTING_BFCL_DIR" ]] || \
    die "--resume-after-preflight requires --bfcl-job-id and --bfcl-dir so benchmark data can be reused"
elif [[ -n "$EXISTING_MODEL_JOB_ID$EXISTING_PREFLIGHT_JOB_ID" ]]; then
  die "--model-job-id and --preflight-job-id require --resume-after-preflight"
fi

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
SETUP_DEPENDENCY="" XLAM_DEPENDENCY="" BFCL_DEPENDENCY="" MODEL_DEPENDENCY="" PREFLIGHT_DEPENDENCY=""
BFCL_JOB_ID="" BFCL_DIR=""
# Keep each startup observation bounded so the full afterok graph is submitted
# promptly. Slurm may reject dependencies on successful jobs after MinJobAge.
PIPELINE_STARTUP_TIMEOUT=30
base_args+=(--startup-timeout-seconds "$PIPELINE_STARTUP_TIMEOUT")
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
      echo "Pipeline stopped at $label; no later stage was submitted. Resolve the error above before retrying." >&2
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

verify_completed_job() {
  local label="$1" job_id="$2" output compact state exit_code
  if ! output="$(sacct --parsable2 -n -X -j "$job_id" --format=State,ExitCode 2>&1)"; then
    die "could not verify $label job $job_id with sacct: $output"
  fi
  compact="$(printf '%s\n' "$output" | head -n 1 | tr -d '[:space:]')"
  IFS='|' read -r state exit_code <<< "$compact"
  state="${state%%+}"
  [[ "$state" == COMPLETED && "$exit_code" == "0:0" ]] || \
    die "$label job $job_id is not confirmed successful (sacct: ${compact:-no record})"
  echo "Verified $label job $job_id completed successfully."
}

verify_active_or_completed_job() {
  local label="$1" job_id="$2" queue_output acct_output compact state state_field exit_code queue_rc=0
  queue_output="$(squeue -h -j "$job_id" -o '%T|%R' 2>&1)" || queue_rc=$?
  case "$queue_output" in
    "Invalid job id specified"|"slurm_load_jobs error: Invalid job id specified")
      # Nibi may reject a completed job ID after it leaves the controller's
      # active-job table. Fall through to accounting to determine its outcome.
      queue_output=""
      ;;
    *)
      (( queue_rc == 0 )) || die "could not verify $label job $job_id with squeue: $queue_output"
      if [[ -n "$queue_output" ]]; then
        [[ "$queue_output" != *$'\n'* && "$queue_output" == *"|"* && "${queue_output#*|}" != *"|"* ]] || \
          die "unexpected squeue response for $label job $job_id: $queue_output"
        state_field="${queue_output%%|*}"
        if [[ "$state_field" =~ ^([[:alnum:]_.-]+[[:space:]]+)?([[:alnum:]_]+)$ ]]; then
          state="${BASH_REMATCH[2]}"
        else
          die "unexpected squeue state field for $label job $job_id: $queue_output"
        fi
        case "$state" in
          PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|RESIZING|SIGNALING|STAGE_OUT|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|REVOKED|RESV_DEL_HOLD|SPECIAL_EXIT|STOPPED|UPDATE_DB|EXPEDITING|LAUNCH_FAILED|RECONFIG_FAIL|POWER_UP_NODE)
            ;;
          *)
            die "unexpected squeue state for $label job $job_id: $queue_output"
            ;;
        esac
      fi
      ;;
  esac
  if [[ -n "$queue_output" ]]; then
    JOB_DISPOSITION=ACTIVE
    echo "Verified $label job $job_id is still queued or running ($queue_output)."
    return
  fi
  if ! acct_output="$(sacct --parsable2 -n -X -j "$job_id" --format=State,ExitCode 2>&1)"; then
    die "could not verify $label job $job_id with sacct: $acct_output"
  fi
  compact="$(printf '%s\n' "$acct_output" | head -n 1 | tr -d '[:space:]')"
  IFS='|' read -r state exit_code <<< "$compact"
  state="${state%%+}"
  if [[ "$state" == COMPLETED && "$exit_code" == "0:0" ]]; then
    JOB_DISPOSITION=COMPLETED
    echo "Verified $label job $job_id completed successfully."
    return
  fi
  die "$label job $job_id is neither active nor successfully completed (sacct: ${compact:-no record})"
}

echo "Queueing the baseline pipeline under account $ACCOUNT."

if [[ "$CONTINUE_AFTER_XLAM" == 1 ]]; then
  SETUP_JOB_ID="$EXISTING_SETUP_JOB_ID"
  XLAM_JOB_ID="$EXISTING_XLAM_JOB_ID"
  if [[ "$EXISTING_XLAM_DIR" == /* ]]; then
    XLAM_DIR="$EXISTING_XLAM_DIR"
  else
    XLAM_DIR="$PROJECT_DIR/$EXISTING_XLAM_DIR"
  fi
  [[ -d "$XLAM_DIR" ]] || die "xLAM data directory does not exist: $XLAM_DIR"
  python3 - "$XLAM_DIR" <<'PY'
import json, sys
import math
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"xLAM manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("dataset") != "Salesforce/xlam-function-calling-60k":
    raise SystemExit("xLAM manifest identifies an unexpected dataset")
for split in ("train", "validation", "calibration", "test"):
    path = root / f"{split}.jsonl"
    expected = manifest.get("counts", {}).get(split)
    if not path.is_file() or not isinstance(expected, int) or expected <= 0:
        raise SystemExit(f"xLAM {split} split is missing or has an invalid manifest count")
    with path.open(encoding="utf-8") as stream:
        actual = 0
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidates = row["candidates"]
                weights = [float(value) for value in row["target_weights"]]
                candidate_ids = [candidate["candidate_id"] for candidate in candidates]
                valid = (
                    isinstance(row, dict)
                    and isinstance(row.get("query"), str) and bool(row["query"].strip())
                    and isinstance(row.get("example_id"), str) and bool(row["example_id"])
                    and isinstance(row.get("group_id"), str) and bool(row["group_id"])
                    and isinstance(candidates, list) and len(candidates) >= 2
                    and len(candidate_ids) == len(set(candidate_ids))
                    and all(
                        set(candidate) == {"candidate_id", "text"}
                        and isinstance(candidate["text"], str)
                        and candidate["text"]
                        for candidate in candidates
                    )
                    and len(weights) == len(candidates)
                    and all(math.isfinite(weight) and weight >= 0 for weight in weights)
                    and any(weight > 0 for weight in weights)
                    and abs(sum(weights) - 1.0) <= 1e-5
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                raise SystemExit(f"invalid xLAM {split} JSONL record at line {line_number}")
            actual += 1
    if actual != expected:
        raise SystemExit(f"xLAM {split} count mismatch: manifest={expected}, file={actual}")
PY
  verify_completed_job setup "$SETUP_JOB_ID"
  verify_completed_job xLAM "$XLAM_JOB_ID"
  if [[ -n "$EXISTING_BFCL_JOB_ID" ]]; then
    BFCL_JOB_ID="$EXISTING_BFCL_JOB_ID"
    if [[ "$EXISTING_BFCL_DIR" == /* ]]; then
      BFCL_DIR="$EXISTING_BFCL_DIR"
    else
      BFCL_DIR="$PROJECT_DIR/$EXISTING_BFCL_DIR"
    fi
    [[ -d "$BFCL_DIR" ]] || die "BFCL data directory does not exist: $BFCL_DIR"
    verify_completed_job BFCL "$BFCL_JOB_ID"
    python3 - "$BFCL_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "selector_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"BFCL selector manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("source") != "gorilla-llm/Berkeley-Function-Calling-Leaderboard":
    raise SystemExit("BFCL selector manifest identifies an unexpected source")
for category in ("multiple", "live_multiple"):
    path = root / f"{category}.selector.jsonl"
    expected = manifest.get("categories", {}).get(category)
    if not path.is_file() or not isinstance(expected, int) or expected <= 0:
        raise SystemExit(f"BFCL {category} selector data is missing or has an invalid count")
    with path.open(encoding="utf-8") as stream:
        actual = sum(1 for line in stream if line.strip())
    if actual != expected:
        raise SystemExit(f"BFCL {category} count mismatch: manifest={expected}, file={actual}")
PY
    echo "Reusing BFCL data $BFCL_DIR from job $BFCL_JOB_ID."
  fi
  if [[ "$RESUME_AFTER_PREFLIGHT" == 0 ]]; then
    echo "Reusing prepared xLAM data $XLAM_DIR. Refreshing the environment from the current lockfile."
    submit_stage setup "$SUBMIT" "${base_args[@]}" --stage setup
    SETUP_JOB_ID="$LAST_JOB_ID"
    SETUP_DEPENDENCY="$SETUP_JOB_ID"
  fi
else
  submit_stage setup "$SUBMIT" "${base_args[@]}" --stage setup
  SETUP_JOB_ID="$LAST_JOB_ID"
  SETUP_DEPENDENCY="$SETUP_JOB_ID"

  # This is the only stage that needs the gated dataset credential.
  submit_stage prepare-xlam "$SUBMIT" "${base_args[@]}" --stage prepare-xlam \
    --dependency "$SETUP_JOB_ID" --bm25-negatives 0
  XLAM_JOB_ID="$LAST_JOB_ID" XLAM_DIR="$LAST_RUN_DIR"
  XLAM_DEPENDENCY="$XLAM_JOB_ID"
  unset HF_TOKEN
fi

if [[ "$RESUME_AFTER_PREFLIGHT" == 1 ]]; then
  verify_active_or_completed_job model "$EXISTING_MODEL_JOB_ID"
  MODEL_JOB_ID="$EXISTING_MODEL_JOB_ID"
  MODEL_JOB_DISPOSITION="$JOB_DISPOSITION"
  if [[ "$MODEL_JOB_DISPOSITION" == ACTIVE ]]; then
    MODEL_DEPENDENCY="$MODEL_JOB_ID"
  fi
  verify_active_or_completed_job preflight "$EXISTING_PREFLIGHT_JOB_ID"
  PREFLIGHT_JOB_ID="$EXISTING_PREFLIGHT_JOB_ID"
  if [[ "$JOB_DISPOSITION" == ACTIVE ]]; then
    PREFLIGHT_DEPENDENCY="$PREFLIGHT_JOB_ID"
  elif [[ "$MODEL_JOB_DISPOSITION" != COMPLETED ]]; then
    die "preflight job $PREFLIGHT_JOB_ID completed before model job $MODEL_JOB_ID was confirmed successful"
  fi
  echo "Reusing model job $MODEL_JOB_ID and preflight job $PREFLIGHT_JOB_ID; resuming at training."
else
  if [[ -z "$BFCL_JOB_ID" ]]; then
    if [[ -n "$SETUP_DEPENDENCY" ]]; then
      submit_stage prepare-bfcl "$SUBMIT" "${base_args[@]}" --stage prepare-bfcl \
        --dependency "$SETUP_DEPENDENCY"
    else
      submit_stage prepare-bfcl "$SUBMIT" "${base_args[@]}" --stage prepare-bfcl
    fi
    BFCL_JOB_ID="$LAST_JOB_ID" BFCL_DIR="$LAST_RUN_DIR"
    BFCL_DEPENDENCY="$BFCL_JOB_ID"
  fi

  if [[ -n "$SETUP_DEPENDENCY" ]]; then
    submit_stage prepare-model "$SUBMIT" "${base_args[@]}" --stage prepare-model \
      --config "$CONFIG" --dependency "$SETUP_DEPENDENCY"
  else
    submit_stage prepare-model "$SUBMIT" "${base_args[@]}" --stage prepare-model \
      --config "$CONFIG"
  fi
  MODEL_JOB_ID="$LAST_JOB_ID"

  submit_gpu_stage preflight "$SUBMIT" "${base_args[@]}" \
    --stage preflight --dependency "$MODEL_JOB_ID"
  PREFLIGHT_JOB_ID="$LAST_JOB_ID"
  PREFLIGHT_DEPENDENCY="$PREFLIGHT_JOB_ID"
fi

# In the normal pipeline, preflight carries the model-success dependency. On
# resume, an active model job is also included as a direct gate; completed model
# jobs are omitted so an old ID cannot expire while preflight waits in the queue.
TRAIN_DEPENDENCIES="$PREFLIGHT_DEPENDENCY"
[[ -z "$MODEL_DEPENDENCY" ]] || TRAIN_DEPENDENCIES="${TRAIN_DEPENDENCIES:+$TRAIN_DEPENDENCIES,}$MODEL_DEPENDENCY"
[[ -z "$XLAM_DEPENDENCY" ]] || TRAIN_DEPENDENCIES="$XLAM_DEPENDENCY,$TRAIN_DEPENDENCIES"
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

BENCHMARK_DEPENDENCIES="$CALIBRATION_JOB_ID"
[[ -z "$BFCL_DEPENDENCY" ]] || BENCHMARK_DEPENDENCIES="$BFCL_DEPENDENCY,$BENCHMARK_DEPENDENCIES"
submit_gpu_stage evaluate-bfcl "$SUBMIT" "${base_args[@]}" --stage benchmark \
  --checkpoint "$TRAIN_DIR/checkpoints/best.pt" --data-dir "$BFCL_DIR" \
  --category live_multiple --calibration "$CALIBRATION_DIR/calibration.json" \
  --dependency "$BENCHMARK_DEPENDENCIES"
BFCL_EVALUATION_JOB_ID="$LAST_JOB_ID" BFCL_EVALUATION_DIR="$LAST_RUN_DIR"

cat <<SUMMARY

Pipeline queued with afterok dependencies. Jobs run only after their required predecessors succeed.
  setup:          $SETUP_JOB_ID
  xLAM data:      $XLAM_JOB_ID  ($XLAM_DIR)
  BFCL data:      $BFCL_JOB_ID  ($BFCL_DIR)
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
