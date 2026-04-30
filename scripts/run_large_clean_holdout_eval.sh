#!/usr/bin/env bash
set -euo pipefail

# Build a fresh Lost-in-Conversation math/actions holdout and evaluate base Qwen
# plus the current dense/RLRF checkpoint. The script is idempotent: if a stage
# completed and wrote its metrics file, rerunning this script skips that stage.
#
# Recommended:
#   tmux new -s large-holdout
#   ./scripts/run_large_clean_holdout_eval.sh
#
# On a local Mac, tmux survives terminal disconnects but does not prevent sleep.
# Use caffeinate or a remote machine for unattended runs.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "TINKER_API_KEY is not set. Export it in your shell; do not put it in the repo." >&2
    exit 1
fi

RUN_PREFIX="${RUN_PREFIX:-lost-math-actions-tools-clean-holdout}"
SOURCE="${SOURCE:-microsoft/lost_in_conversation}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
HOLDOUT_SEED="${HOLDOUT_SEED:-101}"
MAX_HOLDOUT_RECORDS="${MAX_HOLDOUT_RECORDS:-0}"
HOLDOUT_DIR="${HOLDOUT_DIR:-$PROJECT_ROOT/_logs/holdouts/$RUN_PREFIX}"
EVAL_LOG_DIR="${TINKER_EVAL_LOG_DIR:-$PROJECT_ROOT/_logs/tinker_eval}"

REFERENCE_SPLIT_DIR="${REFERENCE_SPLIT_DIR:-$PROJECT_ROOT/datasets/sharded_multiturn/lost_math_actions_tools_200}"
EXCLUDE_CURRENT_TEST="${EXCLUDE_CURRENT_TEST:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
RENDERER_NAME="${RENDERER_NAME:-qwen3_disable_thinking}"
DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-tinker://f0428179-06ff-51f6-8bf1-edb9baf7e298:train:0/sampler_weights/lost-math-actions-tools-dense-rlrf-shuffle7-60-final-sampler}"

MAX_TURNS="${MAX_TURNS:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SHARDED_PROMPT_STYLE="${SHARDED_PROMPT_STYLE:-tool_schema}"
SHARDED_ALLOW_UNTAGGED_FINAL="${SHARDED_ALLOW_UNTAGGED_FINAL:-1}"

RESUME="${RESUME:-1}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-300}"
RETRY_FOREVER="${RETRY_FOREVER:-0}"

mkdir -p "$HOLDOUT_DIR" "$EVAL_LOG_DIR" "$PROJECT_ROOT/_logs/large_holdout_eval"
DRIVER_LOG="$PROJECT_ROOT/_logs/large_holdout_eval/$RUN_PREFIX-driver.log"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"
}

stage_done() {
    local metrics_path="$1"
    [ "$RESUME" = "1" ] && [ -s "$metrics_path" ]
}

run_with_retries() {
    local label="$1"
    shift
    local attempt=1
    while true; do
        log "Starting stage: $label (attempt $attempt)"
        if "$@" 2>&1 | tee -a "$DRIVER_LOG"; then
            log "Completed stage: $label"
            return 0
        fi
        log "Stage failed: $label"
        if [ "$RETRY_FOREVER" != "1" ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
            log "Giving up on stage after $attempt attempts: $label"
            return 1
        fi
        attempt=$((attempt + 1))
        log "Sleeping $RETRY_SLEEP_SECONDS seconds before retrying $label"
        sleep "$RETRY_SLEEP_SECONDS"
    done
}

build_holdout() {
    local args=(
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_clean_holdout.py"
        --source "$SOURCE"
        --split "$SOURCE_SPLIT"
        --output-dir "$HOLDOUT_DIR"
        --seed "$HOLDOUT_SEED"
        --max-records "$MAX_HOLDOUT_RECORDS"
        --task math
        --task actions
        --exclude-json "$REFERENCE_SPLIT_DIR/train.json"
    )
    if [ "$EXCLUDE_CURRENT_TEST" = "1" ] || [ "$EXCLUDE_CURRENT_TEST" = "true" ]; then
        args+=(--exclude-json "$REFERENCE_SPLIT_DIR/test.json")
    fi
    "${args[@]}"
}

run_eval() {
    local run_name="$1"
    local model_path="${2:-}"
    local env_args=(
        "PYTHON_BIN=$PYTHON_BIN"
        "TINKER_EVAL_LOG_DIR=$EVAL_LOG_DIR"
        "DATA_PATH=$HOLDOUT_DIR"
        "SPLIT=test"
        "MODEL_NAME=$MODEL_NAME"
        "RENDERER_NAME=$RENDERER_NAME"
        "SHARDED_PROMPT_STYLE=$SHARDED_PROMPT_STYLE"
        "SHARDED_ALLOW_UNTAGGED_FINAL=$SHARDED_ALLOW_UNTAGGED_FINAL"
        "MAX_TURNS=$MAX_TURNS"
        "MAX_TOKENS=$MAX_TOKENS"
        "TEMPERATURE=$TEMPERATURE"
        "BATCH_SIZE=$BATCH_SIZE"
        "NUM_SAMPLES=$NUM_SAMPLES"
        "MAX_EXAMPLES=$MAX_EXAMPLES"
    )
    if [ -n "$model_path" ]; then
        env_args+=("MODEL_PATH=$model_path")
    else
        env_args+=("MODEL_PATH=")
    fi
    env "${env_args[@]}" "$PROJECT_ROOT/run_tinker_eval.sh" "$run_name"
}

BASE_RUN="${RUN_PREFIX}-base-512"
DENSE_RUN="${RUN_PREFIX}-dense-rlrf-shuffle7-60-512"
BASE_METRICS="$EVAL_LOG_DIR/$BASE_RUN-metrics.json"
DENSE_METRICS="$EVAL_LOG_DIR/$DENSE_RUN-metrics.json"

log "Large clean holdout eval started"
log "Holdout dir: $HOLDOUT_DIR"
log "Run prefix: $RUN_PREFIX"
log "Exclude current test: $EXCLUDE_CURRENT_TEST"

if [ "$RESUME" = "1" ] && [ -s "$HOLDOUT_DIR/test.json" ] && [ -s "$HOLDOUT_DIR/holdout_summary.json" ]; then
    log "Skipping holdout build; existing holdout found."
else
    run_with_retries "build-holdout" build_holdout
fi

log "Holdout summary:"
"$PYTHON_BIN" - <<'PY' "$HOLDOUT_DIR/holdout_summary.json" | tee -a "$DRIVER_LOG"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(path.read_text(encoding="utf-8").strip())
PY

if stage_done "$BASE_METRICS"; then
    log "Skipping base eval; metrics already exist: $BASE_METRICS"
else
    run_with_retries "base-eval" run_eval "$BASE_RUN" ""
fi

if stage_done "$DENSE_METRICS"; then
    log "Skipping dense eval; metrics already exist: $DENSE_METRICS"
else
    run_with_retries "dense-eval" run_eval "$DENSE_RUN" "$DENSE_MODEL_PATH"
fi

log "Final metrics:"
"$PYTHON_BIN" - <<'PY' "$BASE_METRICS" "$DENSE_METRICS" | tee -a "$DRIVER_LOG"
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.exists():
        print(f"MISSING {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{payload['run_name']}: reward_mean={payload['reward_mean']:.4f} "
        f"format_error_rate={payload['format_error_rate']:.4f} "
        f"examples={payload['examples']}"
    )
PY

log "Large clean holdout eval finished"
