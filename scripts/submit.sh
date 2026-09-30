#!/usr/bin/env bash

set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAGE="" ACCOUNT="" PARTITION="" GPU_RESOURCE="" RESOURCE_PROFILE="" GPU_COUNT=0 CONFIG="" DATA_DIR="" INPUT_JSON=""
OUTPUT_DIR="" OUTPUT_DIR_SET=0 CHECKPOINT="" DATA_PATH="" CALIBRATION="" CATEGORY="" REVISION="main"
ENV_SCRIPT=""
BM25_NEGATIVES=0 RESUME=0 CPUS="" MEMORY="" WALL_TIME="" STARTUP_TIMEOUT=180 MIN_FREE_GIB=15
DEPENDENCY=""

usage() {
  cat <<'USAGE'
Usage: scripts/submit.sh --stage STAGE --account ACCOUNT [options]

Stages: setup, prepare-xlam, prepare-bfcl, prepare-model, preflight, train, calibrate, evaluate, benchmark

Common: --project-dir PATH --env-script PATH --partition NAME --resource-profile PATH
        --gpu-resource SPEC (legacy --gres override)
        --cpus-per-task N --memory SIZE --time D-HH:MM:SS
        --output-dir PATH --dependency JOB_ID --startup-timeout-seconds N --min-free-gib N --help
        GPU stages use the profile's exact scheduler request and runtime checks. The default is
        configs/resources/nibi-h100-80gb.json; select another profile for a different Alliance GPU.

Stage options:
  prepare-xlam:  [--input-json PATH] [--bm25-negatives N]
  prepare-bfcl:  [--revision REF]
  prepare-model: --config PATH (downloads pinned weights in a CPU allocation)
  preflight:     checks the allocated GPU against the selected resource profile
  train:         --config PATH --data-dir PATH [--resume]
  calibrate:     --checkpoint PATH --data PATH
  evaluate:      --checkpoint PATH --data PATH [--calibration PATH]
  benchmark:     --checkpoint PATH --data-dir PATH --category multiple|live_multiple [--calibration PATH]

Submissions write timestamped logs and a durable receipt below logs/. After submission inspect with:
  squeue -j JOB_ID; scontrol show job JOB_ID
  sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
USAGE
}
die() { echo "submit: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --stage|--project-dir|--env-script|--account|--partition|--gpu-resource|--resource-profile|--config|--data-dir|--input-json|--output-dir|--checkpoint|--data|--calibration|--category|--revision|--bm25-negatives|--cpus-per-task|--memory|--time|--startup-timeout-seconds|--min-free-gib|--dependency)
      (($# >= 2)) || die "$1 needs a value"
      key="$1"; value="$2"; shift 2
      case "$key" in
        --stage) STAGE="$value" ;; --project-dir) PROJECT_DIR="$value" ;; --env-script) ENV_SCRIPT="$value" ;;
        --account) ACCOUNT="$value" ;; --partition) PARTITION="$value" ;;
        --gpu-resource) GPU_RESOURCE="$value" ;; --resource-profile) RESOURCE_PROFILE="$value" ;; --config) CONFIG="$value" ;;
        --data-dir) DATA_DIR="$value" ;; --input-json) INPUT_JSON="$value" ;;
        --output-dir) OUTPUT_DIR="$value"; OUTPUT_DIR_SET=1 ;; --checkpoint) CHECKPOINT="$value" ;;
        --data) DATA_PATH="$value" ;; --calibration) CALIBRATION="$value" ;;
        --category) CATEGORY="$value" ;; --revision) REVISION="$value" ;;
        --bm25-negatives) BM25_NEGATIVES="$value" ;; --cpus-per-task) CPUS="$value" ;;
        --memory) MEMORY="$value" ;; --time) WALL_TIME="$value" ;;
        --startup-timeout-seconds) STARTUP_TIMEOUT="$value" ;; --min-free-gib) MIN_FREE_GIB="$value" ;;
        --dependency) DEPENDENCY="$value" ;;
      esac
      ;;
    --resume) RESUME=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd -P)" || die "project directory does not exist"
cd "$PROJECT_DIR"
if [[ -n "$ENV_SCRIPT" ]]; then
  [[ "$ENV_SCRIPT" == /* ]] || ENV_SCRIPT="$PROJECT_DIR/$ENV_SCRIPT"
  [[ -r "$ENV_SCRIPT" ]] || die "environment script is not readable: $ENV_SCRIPT"
  # shellcheck disable=SC1090
  source "$ENV_SCRIPT"
  export CSD_ENV_SCRIPT="$ENV_SCRIPT"
fi
MODEL_MANIFEST="${CSD_MODEL_CACHE_MANIFEST:-$PROJECT_DIR/models/model-cache-manifest.json}"
[[ "$MODEL_MANIFEST" == /* ]] || MODEL_MANIFEST="$PROJECT_DIR/$MODEL_MANIFEST"
export CSD_MODEL_CACHE_MANIFEST="$MODEL_MANIFEST"

[[ -n "$STAGE" ]] || die "--stage is required"
[[ -n "$ACCOUNT" && "$ACCOUNT" =~ ^[[:alnum:]_.-]+$ ]] || die "valid --account is required"
[[ -z "$PARTITION" || "$PARTITION" =~ ^[[:alnum:]_.-]+$ ]] || die "invalid --partition"
[[ "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] && (( STARTUP_TIMEOUT >= 10 && STARTUP_TIMEOUT <= 1800 )) || die "startup timeout must be 10..1800 seconds"
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] && (( MIN_FREE_GIB >= 1 && MIN_FREE_GIB <= 100000 )) || die "minimum free space must be 1..100000 GiB"
[[ "$BM25_NEGATIVES" =~ ^[0-9]+$ ]] && (( BM25_NEGATIVES <= 100 )) || die "BM25 negative count must be 0..100"
[[ -z "$DEPENDENCY" || "$DEPENDENCY" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--dependency needs numeric Slurm job IDs separated by commas"
REQUESTED_DEPENDENCY="$DEPENDENCY"

case "$STAGE" in
  setup)
    [[ -z "$GPU_RESOURCE" && "$RESUME" == 0 ]] || die "setup does not accept GPU or resume options"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-16G}"; WALL_TIME="${WALL_TIME:-02:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/operations/setup" ;;
  prepare-xlam)
    [[ -z "$GPU_RESOURCE" && "$RESUME" == 0 ]] || die "prepare-xlam is a CPU stage and cannot resume"
    [[ -z "$INPUT_JSON" || -f "$INPUT_JSON" ]] || die "input JSON does not exist: $INPUT_JSON"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-24G}"; WALL_TIME="${WALL_TIME:-02:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/data/processed/xlam" ;;
  prepare-bfcl)
    [[ -z "$GPU_RESOURCE" && "$RESUME" == 0 ]] || die "prepare-bfcl is a CPU stage and cannot resume"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-12G}"; WALL_TIME="${WALL_TIME:-01:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/data/benchmark/bfcl" ;;
  prepare-model)
    [[ -z "$GPU_RESOURCE" && "$RESUME" == 0 ]] || die "prepare-model is a CPU stage and cannot resume"
    [[ -n "$CONFIG" && -f "$CONFIG" ]] || die "prepare-model needs an existing --config"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-24G}"; WALL_TIME="${WALL_TIME:-01:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/model-cache-$(basename "${CONFIG%.json}")" ;;
  preflight)
    [[ "$RESUME" == 0 ]] || die "preflight cannot resume"
    GPU_COUNT=1
    CPUS="${CPUS:-2}"; MEMORY="${MEMORY:-8G}"; WALL_TIME="${WALL_TIME:-00:10:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/preflight" ;;
  train)
    GPU_COUNT=1
    [[ "$RESUME" == 0 || -z "$DEPENDENCY" ]] || die "resume cannot use --dependency"
    [[ -n "$CONFIG" && -f "$CONFIG" ]] || die "train needs an existing --config"
    [[ -n "$DATA_DIR" && ( -d "$DATA_DIR" || -n "$DEPENDENCY" ) ]] || die "train needs an existing --data-dir, or --dependency"
    if [[ -z "$DEPENDENCY" ]]; then
      for split in train validation; do [[ -f "$DATA_DIR/$split.jsonl" ]] || die "missing $DATA_DIR/$split.jsonl"; done
    fi
    CPUS="${CPUS:-8}"; MEMORY="${MEMORY:-64G}"; WALL_TIME="${WALL_TIME:-12:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/$(basename "${CONFIG%.json}")-seed42" ;;
  calibrate)
    GPU_COUNT=1
    [[ "$RESUME" == 0 ]] || die "calibrate does not support resume"
    [[ -n "$CHECKPOINT" && ( -f "$CHECKPOINT" || -n "$DEPENDENCY" ) ]] || die "calibrate needs an existing --checkpoint, or --dependency"
    [[ -n "$DATA_PATH" && ( -f "$DATA_PATH" || -n "$DEPENDENCY" ) ]] || die "calibrate needs an existing --data file, or --dependency"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-32G}"; WALL_TIME="${WALL_TIME:-03:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/calibration" ;;
  evaluate)
    GPU_COUNT=1
    [[ "$RESUME" == 0 ]] || die "evaluate does not support resume"
    [[ -n "$CHECKPOINT" && ( -f "$CHECKPOINT" || -n "$DEPENDENCY" ) ]] || die "evaluate needs an existing --checkpoint, or --dependency"
    [[ -n "$DATA_PATH" && ( -f "$DATA_PATH" || -n "$DEPENDENCY" ) ]] || die "evaluate needs an existing --data file, or --dependency"
    [[ -z "$CALIBRATION" || -f "$CALIBRATION" || -n "$DEPENDENCY" ]] || die "calibration file does not exist, or use --dependency"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-32G}"; WALL_TIME="${WALL_TIME:-03:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/evaluation" ;;
  benchmark)
    GPU_COUNT=1
    [[ "$RESUME" == 0 ]] || die "benchmark does not support resume"
    [[ -n "$CHECKPOINT" && ( -f "$CHECKPOINT" || -n "$DEPENDENCY" ) ]] || die "benchmark needs an existing --checkpoint, or --dependency"
    [[ -n "$DATA_DIR" && ( -d "$DATA_DIR" || -n "$DEPENDENCY" ) ]] || die "benchmark needs an existing --data-dir, or --dependency"
    [[ "$CATEGORY" == multiple || "$CATEGORY" == live_multiple ]] || die "category must be multiple or live_multiple"
    [[ -f "$DATA_DIR/$CATEGORY.selector.jsonl" || -n "$DEPENDENCY" ]] || die "missing selector file for $CATEGORY, or use --dependency"
    [[ -z "$CALIBRATION" || -f "$CALIBRATION" || -n "$DEPENDENCY" ]] || die "calibration file does not exist, or use --dependency"
    CPUS="${CPUS:-4}"; MEMORY="${MEMORY:-32G}"; WALL_TIME="${WALL_TIME:-03:00:00}"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$PROJECT_DIR/runs/bfcl-$CATEGORY" ;;
  *) die "unknown stage: $STAGE" ;;
esac

PROFILE_NAME="" PROFILE_GPU_OPTION="" PROFILE_GPU_REQUEST="" PROFILE_GPU_PARTITION="" GPU_PROFILE_SHA256=""
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TOKEN="$(python3 -c 'import uuid; print(uuid.uuid4())')"
SUBMISSION_DIR="$PROJECT_DIR/logs/submissions/$STAGE-$STAMP-$TOKEN"
mkdir -p "$SUBMISSION_DIR"
if (( GPU_COUNT > 0 )); then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-$PROJECT_DIR/configs/resources/nibi-h100-80gb.json}"
  [[ "$RESOURCE_PROFILE" == /* ]] || RESOURCE_PROFILE="$PROJECT_DIR/$RESOURCE_PROFILE"
  [[ -r "$RESOURCE_PROFILE" ]] || die "GPU resource profile is not readable: $RESOURCE_PROFILE"
  RESOURCE_PROFILE_SNAPSHOT="$SUBMISSION_DIR/resource-profile.json"
  if ! profile_output="$(python3 - "$RESOURCE_PROFILE" "$RESOURCE_PROFILE_SNAPSHOT" <<'PY'
import hashlib, json, math, os, re, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
contents = source.read_bytes()
profile = json.loads(contents)
if not isinstance(profile, dict) or profile.get("schema_version") != 1:
    raise SystemExit("resource profile schema_version must be 1")
name = profile.get("name")
scheduler = profile.get("scheduler")
runtime = profile.get("runtime")
if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", name):
    raise SystemExit("resource profile name must be a short, single-line label")
if not isinstance(scheduler, dict) or not isinstance(runtime, dict):
    raise SystemExit("resource profile needs scheduler and runtime objects")
option = scheduler.get("gpu_option")
request = scheduler.get("gpu_request")
partition = scheduler.get("partition") or ""
if option not in {"--gpus", "--gres"}: raise SystemExit("gpu_option must be --gpus or --gres")
if not isinstance(request, str) or not re.fullmatch(r"[A-Za-z0-9_.:=+-]+", request):
    raise SystemExit("gpu_request has invalid Slurm resource syntax")
if partition and not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
    raise SystemExit("profile partition has invalid syntax")
for key in ("min_memory_gib", "min_free_memory_gib"):
    value = runtime.get(key, 0)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SystemExit(f"runtime.{key} must be a finite nonnegative number")
if not isinstance(runtime.get("require_cuda", True), bool): raise SystemExit("runtime.require_cuda must be boolean")
if runtime.get("require_cuda", True) is not True: raise SystemExit("GPU stages require runtime.require_cuda=true")
if not isinstance(runtime.get("require_bf16", False), bool): raise SystemExit("runtime.require_bf16 must be boolean")
if not isinstance(runtime.get("device_name_contains", ""), str):
    raise SystemExit("runtime.device_name_contains must be a string")
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
temporary.write_bytes(contents)
os.replace(temporary, destination)
print(name)
print(option)
print(request)
print(partition)
print(hashlib.sha256(contents).hexdigest())
PY
)"; then
    die "invalid GPU resource profile $RESOURCE_PROFILE: $profile_output"
  fi
  mapfile -t profile_fields <<< "$profile_output"
  (( ${#profile_fields[@]} >= 3 )) || die "could not parse GPU resource profile: $RESOURCE_PROFILE"
  PROFILE_NAME="${profile_fields[0]}"
  PROFILE_GPU_OPTION="${profile_fields[1]}"
  PROFILE_GPU_REQUEST="${profile_fields[2]}"
  PROFILE_GPU_PARTITION="${profile_fields[3]:-}"
  GPU_PROFILE_SHA256="${profile_fields[4]:-}"
  [[ "$GPU_PROFILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "could not calculate the GPU resource profile hash"
  RESOURCE_PROFILE="$RESOURCE_PROFILE_SNAPSHOT"
elif [[ -n "$RESOURCE_PROFILE" ]]; then
  die "--resource-profile applies only to GPU stages"
fi

# The gated xLAM dataset is the only current stage that needs a Hugging Face
# credential. Do not place it in unrelated Slurm job environments.
if [[ "$STAGE" != "prepare-xlam" ]]; then unset HF_TOKEN; fi

[[ "$CPUS" =~ ^[0-9]+$ ]] && (( CPUS >= 1 && CPUS <= 128 )) || die "cpus-per-task must be 1..128"
[[ "$MEMORY" =~ ^[0-9]+([KMGTP])?$ ]] || die "memory must look like 32G or 64000M"
[[ "$WALL_TIME" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || die "time must be HH:MM:SS or D-HH:MM:SS"
if [[ -n "$GPU_RESOURCE" ]]; then [[ "$GPU_RESOURCE" =~ ^[[:alnum:]_.:=+-]+$ ]] || die "invalid GPU resource syntax"; fi

[[ -f "$PROJECT_DIR/pyproject.toml" && -f "$PROJECT_DIR/scripts/receipt.py" ]] || die "not a project checkout"
for executable in sbatch squeue scontrol sacct srun python3 uv; do
  if [[ "$executable" == uv && -n "$ENV_SCRIPT" ]]; then continue; fi
  command -v "$executable" >/dev/null 2>&1 || die "required executable is unavailable: $executable"
done
if [[ -n "$ENV_SCRIPT" ]]; then command -v uv >/dev/null 2>&1 || die "uv is unavailable after loading --env-script"; fi
[[ -w "$PROJECT_DIR" ]] || die "project directory is not writable"

if [[ "$STAGE" == train || "$STAGE" == prepare-model ]]; then
  python3 - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"model_name", "model_revision", "architecture", "projection_dim", "tau", "epochs", "batch_size"}
missing = required - config.keys()
if missing: raise SystemExit(f"config missing fields: {', '.join(sorted(missing))}")
if config["architecture"] not in {"shared_tied", "shared_heads", "separate"}: raise SystemExit("unsupported architecture")
if not isinstance(config.get("require_h100", True), bool): raise SystemExit("require_h100 must be boolean")
import re
if not re.fullmatch(r"[0-9a-f]{40}", str(config["model_revision"])): raise SystemExit("model_revision must be a full 40-character commit SHA")
if config["epochs"] < 1 or config["batch_size"] < 1 or config["tau"] <= 0: raise SystemExit("invalid training values")
PY
fi
if [[ "$STAGE" == train || "$STAGE" == calibrate || "$STAGE" == evaluate || "$STAGE" == benchmark ]]; then
  [[ -f "$MODEL_MANIFEST" || -n "$DEPENDENCY" ]] || die "pinned model cache is missing; run prepare-model or pass its job ID with --dependency"
  # A dependent prepare-model stage may replace a stale manifest before train starts.
  if [[ "$STAGE" == train && -f "$MODEL_MANIFEST" && -z "$DEPENDENCY" ]]; then
    python3 - "$CONFIG" "$MODEL_MANIFEST" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
if config.get("model_name") != manifest.get("model_name") or config.get("model_revision") != manifest.get("revision"):
    raise SystemExit("prepared model cache does not match this training config")
PY
  fi
fi
python3 - "$PROJECT_DIR" "$MIN_FREE_GIB" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / (1024**3)
required = float(sys.argv[2])
if free < required: raise SystemExit(f"only {free:.1f} GiB available; {required:.1f} GiB required")
PY

if [[ "$STAGE" == train && "$RESUME" == 1 ]]; then
  [[ "$OUTPUT_DIR_SET" == 1 && -d "$OUTPUT_DIR" ]] || die "resume requires an explicit existing --output-dir"
  RUN_DIR="$(python3 "$PROJECT_DIR/scripts/receipt.py" within --root "$PROJECT_DIR" --path "$OUTPUT_DIR")"
  [[ -f "$RUN_DIR/checkpoints/last.pt" && -f "$RUN_DIR/run-manifest.json" && -f "$RUN_DIR/config.json" ]] || die "resume run lacks checkpoint, manifest, or config snapshot"
  cmp -s "$CONFIG" "$RUN_DIR/config.json" || die "resume config differs from saved config"
  python3 - "$RUN_DIR" "$DATA_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
run_dir, data_dir = map(Path, sys.argv[1:])
manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
if manifest.get("data_dir") != str(data_dir.resolve()):
    raise SystemExit("resume data directory differs from the original run")
for name in ("train.jsonl", "validation.jsonl"):
    path = data_dir / name
    if not path.is_file():
        raise SystemExit(f"resume input is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest.get("split_sha256", {}).get(name):
        raise SystemExit(f"resume input changed since the original run: {name}")
PY
else
  OUTPUT_DIR="$(python3 "$PROJECT_DIR/scripts/receipt.py" within --root "$PROJECT_DIR" --path "$OUTPUT_DIR")"
  RUN_DIR="$(python3 "$PROJECT_DIR/scripts/receipt.py" reserve --base "$OUTPUT_DIR")"
fi

if [[ "$STAGE" == train && "$RESUME" == 0 ]]; then
  cp "$CONFIG" "$RUN_DIR/config.json.tmp"
  mv "$RUN_DIR/config.json.tmp" "$RUN_DIR/config.json"
  python3 - "$RUN_DIR" "$DATA_DIR" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
run_dir, data_dir = map(Path, sys.argv[1:])
config_path = run_dir / "config.json"
manifest = {
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "config_snapshot": str(config_path.resolve()),
    "data_dir": str(data_dir.resolve()),
    "data_manifest": str((data_dir / "manifest.json").resolve()) if (data_dir / "manifest.json").is_file() else None,
    "data_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest() if (data_dir / "manifest.json").is_file() else None,
    "split_sha256": {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("train.jsonl", "validation.jsonl")
        if (data_dir / name).is_file()
    },
}
temporary = run_dir / "run-manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary.replace(run_dir / "run-manifest.json")
PY
  CONFIG="$RUN_DIR/config.json"
fi

RECEIPT="$SUBMISSION_DIR/receipt.json"
LOG_OUT="$SUBMISSION_DIR/${STAGE}-${STAMP}-%j.out"
LOG_ERR="$SUBMISSION_DIR/${STAGE}-${STAMP}-%j.err"
case "$STAGE" in
  setup) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/setup.sbatch" ;;
  prepare-xlam) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/prepare_xlam.sbatch" ;;
  prepare-bfcl) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/prepare_bfcl.sbatch" ;;
  prepare-model) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/prepare_model.sbatch" ;;
  preflight) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/preflight.sbatch" ;;
  train) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/train.sbatch" ;;
  calibrate) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/calibrate.sbatch" ;;
  evaluate) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/evaluate.sbatch" ;;
  benchmark) SBATCH_SCRIPT="$PROJECT_DIR/scripts/slurm/benchmark.sbatch" ;;
esac
[[ -f "$SBATCH_SCRIPT" ]] || die "missing Slurm entry point: $SBATCH_SCRIPT"

resolve_dependencies() {
  [[ -n "$DEPENDENCY" ]] || return 0
  local -a requested_ids active_ids
  local job_id queue_output queue_rc account_output account_rc account_line state exit_code active_count=0
  IFS=',' read -r -a requested_ids <<< "$DEPENDENCY"
  active_ids=()
  for job_id in "${requested_ids[@]}"; do
    queue_rc=0
    queue_output="$(squeue -h -j "$job_id" -o '%T|%R' 2>&1)" || queue_rc=$?
    case "$queue_output" in
      *"Invalid job id specified"*) queue_output="" ;;
      *) (( queue_rc == 0 )) || die "could not verify dependency job $job_id with squeue: $queue_output" ;;
    esac
    if [[ -n "$queue_output" ]]; then
      [[ "$queue_output" != *$'\n'* && "$queue_output" == *"|"* && "${queue_output#*|}" != *"|"* ]] || \
        die "unexpected squeue response for dependency job $job_id: $queue_output"
      state="${queue_output%%|*}"
      state="${state%%+*}"
      state="${state%% *}"
      case "$state" in
        PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|RESIZING|SIGNALING|STAGE_OUT|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|RESV_DEL_HOLD|STOPPED|UPDATE_DB|EXPEDITING|POWER_UP_NODE)
          active_ids+=("$job_id")
          active_count=$((active_count + 1))
          continue
          ;;
        COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT|LAUNCH_FAILED|RECONFIG_FAIL|COMPLETING*)
          # Confirm terminal states and exit codes through accounting below.
          ;;
        *) die "unrecognized squeue state for dependency job $job_id: $queue_output" ;;
      esac
    fi

    account_rc=0
    account_output="$(sacct --parsable2 -n -X -j "$job_id" --format=State,ExitCode 2>&1)" || account_rc=$?
    (( account_rc == 0 )) || die "could not verify dependency job $job_id with sacct: $account_output"
    account_line="$(printf '%s\n' "$account_output" | head -n 1 | tr -d '[:space:]')"
    [[ -n "$account_line" && "$account_line" == *"|"* ]] || \
      die "dependency job $job_id has no verifiable Slurm accounting record"
    IFS='|' read -r state exit_code <<< "$account_line"
    state="${state%%+*}"
    case "$state" in
      COMPLETED)
        [[ "$exit_code" == "0:0" ]] || die "dependency job $job_id is COMPLETED with nonzero exit code $exit_code"
        echo "Dependency job $job_id already completed successfully; removing its stale afterok reference."
        ;;
      PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|RESIZING|SIGNALING|STAGE_OUT|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|RESV_DEL_HOLD|STOPPED|UPDATE_DB|EXPEDITING|POWER_UP_NODE)
        active_ids+=("$job_id")
        active_count=$((active_count + 1))
        ;;
      *) die "dependency job $job_id is not successful or active (sacct: $account_line)" ;;
    esac
  done
  if (( active_count == 0 )); then
    DEPENDENCY=""
  else
    local IFS=,
    DEPENDENCY="${active_ids[*]}"
  fi
}

resolve_dependencies

COMMAND_JSON="$(python3 - "$STAGE" "$PROJECT_DIR" "$ACCOUNT" "$PARTITION" "$GPU_RESOURCE" "$GPU_COUNT" "$CPUS" "$MEMORY" "$WALL_TIME" "$DEPENDENCY" "$REQUESTED_DEPENDENCY" "$CONFIG" "$DATA_DIR" "$INPUT_JSON" "$RUN_DIR" "$CHECKPOINT" "$DATA_PATH" "$CATEGORY" "$CALIBRATION" "$BM25_NEGATIVES" "$REVISION" "$RESUME" "$ENV_SCRIPT" "$MODEL_MANIFEST" "$RESOURCE_PROFILE" "$PROFILE_NAME" "$PROFILE_GPU_OPTION" "$PROFILE_GPU_REQUEST" "$PROFILE_GPU_PARTITION" "$GPU_PROFILE_SHA256" <<'PY'
import json, sys
(stage, project, account, partition, gpu, gpu_count, cpus, memory, wall, dependency, requested_dependency, config,
 data_dir, input_json, output_dir, checkpoint, data, category, calibration,
 bm25, revision, resume, environment_script, model_manifest, resource_profile, profile_name,
 profile_gpu_option, profile_gpu_request, profile_partition, profile_sha256) = sys.argv[1:]
gpu_count = int(gpu_count)
effective_gpu_option = "--gres" if gpu else profile_gpu_option
effective_gpu_request = gpu or profile_gpu_request
print(json.dumps({
    "stage": stage,
    "project_dir": project,
    "environment_script": environment_script or None,
    "model_cache_manifest": model_manifest,
    "resource_profile": resource_profile or None,
    "resource_profile_name": profile_name or None,
    "resource_profile_sha256": profile_sha256 or None,
    "sbatch": {"account": account, "partition": partition or profile_partition or None,
               "gpu_option": effective_gpu_option or None,
               "gpu_request": effective_gpu_request or None,
               "gpu_resource": gpu or None,
               "cpus_per_task": cpus, "memory": memory, "time": wall,
               "afterok_job_id": dependency or None},
    "dependency_resolution": {"requested_afterok_job_id": requested_dependency or None,
                              "submitted_afterok_job_id": dependency or None},
    "arguments": {"config": config or None, "data_dir": data_dir or None,
                  "input_json": input_json or None, "output_dir": output_dir,
                  "checkpoint": checkpoint or None, "data": data or None,
                  "category": category or None, "calibration": calibration or None,
                  "bm25_negatives": int(bm25), "revision": revision,
                  "resume": bool(int(resume))},
}))
PY
)"
python3 "$PROJECT_DIR/scripts/receipt.py" init --path "$RECEIPT" --token "$TOKEN" --stage "$STAGE" \
  --submitted-at "$STAMP" --project-dir "$PROJECT_DIR" --run-dir "$RUN_DIR" \
  --command-json "$COMMAND_JSON" --log-out "$LOG_OUT" --log-err "$LOG_ERR"

sbatch_args=(--parsable --comment "csd:$TOKEN" --account "$ACCOUNT"
  --cpus-per-task "$CPUS" --mem "$MEMORY" --time "$WALL_TIME" --chdir "$PROJECT_DIR"
  --output "$LOG_OUT" --error "$LOG_ERR" --export=ALL)
EFFECTIVE_PARTITION="$PARTITION"
if [[ -z "$EFFECTIVE_PARTITION" && "$GPU_COUNT" -gt 0 ]]; then EFFECTIVE_PARTITION="$PROFILE_GPU_PARTITION"; fi
[[ -z "$EFFECTIVE_PARTITION" ]] || sbatch_args+=(--partition "$EFFECTIVE_PARTITION")
if [[ -n "$GPU_RESOURCE" ]]; then
  sbatch_args+=(--gres="$GPU_RESOURCE")
elif (( GPU_COUNT > 0 )); then
  sbatch_args+=("$PROFILE_GPU_OPTION=$PROFILE_GPU_REQUEST")
fi
[[ -z "$DEPENDENCY" ]] || sbatch_args+=(--dependency="afterok:${DEPENDENCY//,/:}")
[[ "$STAGE" != train ]] || sbatch_args+=(--signal=B:USR1@300)
case "$STAGE" in
  setup) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR") ;;
  prepare-xlam) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$RUN_DIR" "$BM25_NEGATIVES" "$INPUT_JSON") ;;
  prepare-bfcl) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$RUN_DIR" "$REVISION") ;;
  prepare-model) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$CONFIG" "$MODEL_MANIFEST") ;;
  preflight) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$RESOURCE_PROFILE" "$GPU_PROFILE_SHA256") ;;
  train) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$CONFIG" "$DATA_DIR" "$RESUME" "$RESOURCE_PROFILE" "$GPU_PROFILE_SHA256") ;;
  calibrate) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$CHECKPOINT" "$DATA_PATH" "$RESOURCE_PROFILE" "$GPU_PROFILE_SHA256") ;;
  evaluate) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$CHECKPOINT" "$DATA_PATH" "$CALIBRATION" "$RESOURCE_PROFILE" "$GPU_PROFILE_SHA256") ;;
  benchmark) stage_args=("$PROJECT_DIR" "$RECEIPT" "$TOKEN" "$RUN_DIR" "$CHECKPOINT" "$DATA_DIR" "$CATEGORY" "$CALIBRATION" "$RESOURCE_PROFILE" "$GPU_PROFILE_SHA256") ;;
esac

echo "Submitting $STAGE; resolved output: $RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "RECEIPT=$RECEIPT"
echo "LOG_OUT=$LOG_OUT"
echo "LOG_ERR=$LOG_ERR"
sbatch_rc=0
if sbatch_output="$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "${stage_args[@]}" 2>&1)"; then
  :
else
  sbatch_rc=$?
fi
JOB_ID=""
while IFS= read -r response_line; do
  if [[ "$response_line" =~ ^([0-9]+)(;[[:alnum:]_.-]+)?$ ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  elif [[ -n "$response_line" ]]; then
    echo "sbatch response: $response_line" >&2
  fi
done <<< "$sbatch_output"
if (( sbatch_rc != 0 )); then
  if [[ -n "$JOB_ID" ]]; then
    python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state UNKNOWN/UNVERIFIED \
      --job-id "$JOB_ID" --reason "sbatch exited $sbatch_rc with response: $sbatch_output" --if-nonterminal
    echo "UNKNOWN/UNVERIFIED; possible job ID $JOB_ID. Do not resubmit blindly. Receipt: $RECEIPT" >&2
  else
    if [[ "$sbatch_output" == *"Batch job submission failed:"* ]]; then
      python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state SUBMISSION_REJECTED \
        --reason "$sbatch_output" --if-nonterminal
      echo "Slurm rejected $STAGE; no job was queued. Reason: $sbatch_output" >&2
      echo "Diagnostic record: $RECEIPT" >&2
    else
      python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state UNKNOWN/UNVERIFIED \
        --reason "sbatch returned an ambiguous failure: $sbatch_output" --if-nonterminal
      echo "UNKNOWN/UNVERIFIED. Do not resubmit blindly. Diagnostic record: $RECEIPT" >&2
    fi
  fi
  exit 1
fi
[[ -n "$JOB_ID" ]] || {
  python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state UNKNOWN/UNVERIFIED \
    --reason "unrecognized sbatch response: $sbatch_output" --if-nonterminal
  echo "UNKNOWN/UNVERIFIED. Do not resubmit blindly. Receipt: $RECEIPT" >&2
  exit 1
}
python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state SUBMITTED --job-id "$JOB_ID" --if-nonterminal
LOG_OUT="${LOG_OUT//%j/$JOB_ID}"; LOG_ERR="${LOG_ERR//%j/$JOB_ID}"
echo "JOB_ID=$JOB_ID"
START_MARKER="$RECEIPT.started.json"

show_failure_logs() {
  echo "Slurm logs: stdout=$LOG_OUT stderr=$LOG_ERR" >&2
  if [[ -f "$LOG_ERR" ]]; then
    echo "Last 80 stderr lines:" >&2
    tail -n 80 "$LOG_ERR" >&2
  fi
  if [[ -f "$LOG_OUT" ]]; then
    echo "Last 80 stdout lines:" >&2
    tail -n 80 "$LOG_OUT" >&2
  fi
}

receipt_confirms_started_success() {
  python3 - "$RECEIPT" "$JOB_ID" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
started = value.get("application_start") if isinstance(value, dict) else None
ok = (
    isinstance(value, dict)
    and value.get("state") == "SUCCEEDED"
    and isinstance(started, dict)
    and started.get("state") == "STARTED"
    and str(started.get("job_id")) == sys.argv[2]
)
raise SystemExit(0 if ok else 1)
PY
}

running_seen=0 query_failures=0
deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
while (( $(date +%s) < deadline )); do
  if queue_output="$(squeue -h -j "$JOB_ID" -o '%T|%R' 2>&1)"; then
    query_failures=0
    if [[ -n "$queue_output" ]]; then
      queue_state="${queue_output%%|*}"; queue_reason="${queue_output#*|}"
      [[ "$queue_state" != RUNNING ]] || running_seen=1
      if [[ -f "$START_MARKER" && "$running_seen" == 1 ]]; then
        started_state="$(python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state STARTED \
          --job-id "$JOB_ID" --marker "$START_MARKER" --if-nonterminal --print-state)"
        case "$started_state" in
          SUCCEEDED) echo "Job completed during startup monitoring; receipt: $RECEIPT"; exit 0 ;;
          FAILED|PREEMPTED)
            echo "Job ended during startup monitoring with state $started_state; receipt: $RECEIPT" >&2
            show_failure_logs
            exit 1 ;;
          *) echo "APPLICATION STARTED. Receipt: $RECEIPT"; exit 0 ;;
        esac
      fi
      if [[ "$queue_state" == PENDING || "$queue_state" == CONFIGURING ]]; then
        python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state PENDING --job-id "$JOB_ID" --reason "$queue_reason" --if-nonterminal
      fi
    else
      acct_output=""
      if acct_output="$(sacct --parsable2 -n -X -j "$JOB_ID" --format=State,ExitCode 2>&1)" && [[ -n "$acct_output" ]]; then
        acct_line="$(printf '%s\n' "$acct_output" | head -n 1 | tr -d '[:space:]')"
        acct_state="${acct_line%%|*}"; acct_state="${acct_state%%+}"
        case "$acct_state" in
          COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
            # Slurm accounting can publish a very short job's terminal state
            # before the shared STARTED marker and final receipt become visible
            # to this login process. Give the batch EXIT trap a bounded moment
            # to persist its marker-backed success before classifying it.
            if [[ ! -f "$START_MARKER" && "$acct_state" == COMPLETED && "${acct_line#*|}" == "0:0" ]]; then
              for _ in {1..10}; do
                [[ -f "$START_MARKER" ]] && break
                receipt_confirms_started_success && break
                sleep 1
              done
            fi
            if [[ -f "$START_MARKER" ]]; then
              scheduler_state=FAILED
              if [[ "$acct_state" == COMPLETED && "${acct_line#*|}" == "0:0" ]]; then scheduler_state=SUCCEEDED; fi
              if [[ "$acct_state" == PREEMPTED ]]; then scheduler_state=PREEMPTED; fi
              receipt_state="$(python3 - "$RECEIPT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))
PY
)"
              final_state="$scheduler_state"
              if [[ "$scheduler_state" == FAILED && "$receipt_state" == PREEMPTED ]]; then final_state=PREEMPTED; fi
              if [[ "$receipt_state" != "$final_state" ]]; then
                python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state "$final_state" \
                  --job-id "$JOB_ID" --marker "$START_MARKER" --reason "terminal Slurm state $acct_line"
              fi
              if [[ "$final_state" == SUCCEEDED ]]; then
                echo "Job completed during startup monitoring; state: $final_state; Slurm state: $acct_line. Receipt: $RECEIPT"
                exit 0
              fi
              echo "Job ended during startup monitoring; state: $final_state; Slurm state: $acct_line. Receipt: $RECEIPT" >&2
              show_failure_logs
              exit 1
            fi
            terminal_receipt_state="$(python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" \
              --state FAILED --job-id "$JOB_ID" --reason "terminal Slurm state $acct_line before STARTED" \
              --if-nonterminal --print-state)"
            if [[ "$terminal_receipt_state" == SUCCEEDED ]] && receipt_confirms_started_success; then
              echo "Job completed during startup monitoring; state: SUCCEEDED; Slurm state: $acct_line. Receipt: $RECEIPT"
              exit 0
            fi
            echo "Failed before application startup ($acct_line). Receipt: $RECEIPT" >&2
            show_failure_logs
            exit 1 ;;
        esac
      else
        ((query_failures+=1))
      fi
    fi
  else
    ((query_failures+=1))
  fi
  if (( query_failures >= 5 )); then
    python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state UNKNOWN/UNVERIFIED --job-id "$JOB_ID" --reason "scheduler queries failed repeatedly; do not resubmit blindly" --if-nonterminal
    echo "UNKNOWN/UNVERIFIED. Do not resubmit blindly. Receipt: $RECEIPT; inspect with squeue -j $JOB_ID and sacct -j $JOB_ID" >&2
    exit 1
  fi
  sleep 5
done

queue_output="$(squeue -h -j "$JOB_ID" -o '%T|%R' 2>/dev/null || true)"
if [[ -n "$queue_output" ]]; then
  queue_state="${queue_output%%|*}"; queue_reason="${queue_output#*|}"
  if [[ "$queue_state" == PENDING || "$queue_state" == CONFIGURING ]]; then
    python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state PENDING --job-id "$JOB_ID" --reason "$queue_reason" --if-nonterminal
    echo "Still $queue_state ($queue_reason); queued job was left in place. Receipt: $RECEIPT"; exit 0
  fi
  python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state RUNNING_STARTUP_UNVERIFIED --job-id "$JOB_ID" --reason "running without a valid STARTED marker before timeout" --if-nonterminal
  echo "Job is running but application startup is unverified. Receipt: $RECEIPT" >&2; exit 1
fi
python3 "$PROJECT_DIR/scripts/receipt.py" update --path "$RECEIPT" --state UNKNOWN/UNVERIFIED --job-id "$JOB_ID" --reason "bounded monitor ended without scheduler or application evidence; do not resubmit blindly" --if-nonterminal
echo "UNKNOWN/UNVERIFIED. Do not resubmit blindly. Inspect job $JOB_ID, receipt $RECEIPT, logs $LOG_OUT and $LOG_ERR" >&2
exit 1
