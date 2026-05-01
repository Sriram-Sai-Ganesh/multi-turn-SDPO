#!/usr/bin/env bash
set -euo pipefail

# Train dense/RLRF under the clarification-gated reveal policy, then evaluate
# the resulting sampler on the strict math-only clean holdout.

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
    echo "TINKER_API_KEY is not set. Run this from a shell that has the Tinker API environment." >&2
    exit 1
fi

TRAIN_RUN="${TRAIN_RUN:-lost-math-actions-tools-dense-rlrf-clarifyonly-shuffle7-60}"
EVAL_RUN="${EVAL_RUN:-lost-math-tools-clean-holdout-dense-rlrf-clarifyonly-trained-shuffle7-60-512}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-datasets/sharded_multiturn/lost_math_actions_tools_200}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$PROJECT_ROOT/_logs/holdouts/lost-math-tools-clean-holdout}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
RENDERER_NAME="${RENDERER_NAME:-qwen3_disable_thinking}"
SHARDED_PROMPT_STYLE="${SHARDED_PROMPT_STYLE:-tool_schema}"
SHARDED_REVEAL_POLICY="${SHARDED_REVEAL_POLICY:-clarify_only}"
SHARDED_ALLOW_UNTAGGED_FINAL="${SHARDED_ALLOW_UNTAGGED_FINAL:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-7}"
MAX_STEPS="${MAX_STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_TURNS="${MAX_TURNS:-0}"
TRAIN_MAX_TOKENS="${TRAIN_MAX_TOKENS:-512}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"

LOG_DIR="${STRICT_RUN_LOG_DIR:-$PROJECT_ROOT/_logs/strict_math_dense_train_eval}"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/$TRAIN_RUN-train.log"
EVAL_LOG="$LOG_DIR/$EVAL_RUN-eval.log"
SAMPLER_PATH_FILE="$LOG_DIR/$TRAIN_RUN-sampler-path.txt"
EVAL_METRICS="$PROJECT_ROOT/_logs/tinker_eval/$EVAL_RUN-metrics.json"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

parse_sampler_path() {
    "$PYTHON_BIN" - "$TRAIN_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(r"Saved final Tinker sampler weights: .*?path='([^']+)'", text)
if not matches:
    matches = re.findall(r"path='(tinker://[^']+/sampler_weights/[^']+)'", text)
if not matches:
    raise SystemExit("Could not find final sampler path in training log.")
print(matches[-1])
PY
}

log "Strict clarification-gated dense train/eval started"
log "Training run: $TRAIN_RUN"
log "Eval run: $EVAL_RUN"
log "Train data: $TRAIN_DATA_PATH"
log "Eval data: $EVAL_DATA_PATH"
log "Reveal policy: $SHARDED_REVEAL_POLICY"

if [ -s "$SAMPLER_PATH_FILE" ]; then
    SAMPLER_PATH="$(cat "$SAMPLER_PATH_FILE")"
    log "Skipping training; existing sampler path found: $SAMPLER_PATH"
else
    env \
        PYTHON_BIN="$PYTHON_BIN" \
        DATA_PATH="$TRAIN_DATA_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        RENDERER_NAME="$RENDERER_NAME" \
        SHARDED_REWARD_MODE=dense \
        SHARDED_PROMPT_STYLE="$SHARDED_PROMPT_STYLE" \
        SHARDED_REVEAL_POLICY="$SHARDED_REVEAL_POLICY" \
        SHARDED_ALLOW_UNTAGGED_FINAL="$SHARDED_ALLOW_UNTAGGED_FINAL" \
        SHUFFLE_SEED="$SHUFFLE_SEED" \
        MAX_STEPS="$MAX_STEPS" \
        BATCH_SIZE="$BATCH_SIZE" \
        ROLLOUT_N="$ROLLOUT_N" \
        MAX_TURNS="$MAX_TURNS" \
        MAX_TOKENS="$TRAIN_MAX_TOKENS" \
        TEMPERATURE=1.0 \
        "$PROJECT_ROOT/run_tinker_grpo.sh" "$TRAIN_RUN" 2>&1 | tee "$TRAIN_LOG"
    SAMPLER_PATH="$(parse_sampler_path)"
    printf '%s\n' "$SAMPLER_PATH" > "$SAMPLER_PATH_FILE"
    log "Captured sampler path: $SAMPLER_PATH"
fi

if [ -s "$EVAL_METRICS" ]; then
    log "Skipping eval; metrics already exist: $EVAL_METRICS"
else
    env \
        PYTHON_BIN="$PYTHON_BIN" \
        DATA_PATH="$EVAL_DATA_PATH" \
        SPLIT=test \
        MODEL_PATH="$SAMPLER_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        RENDERER_NAME="$RENDERER_NAME" \
        SHARDED_PROMPT_STYLE="$SHARDED_PROMPT_STYLE" \
        SHARDED_REVEAL_POLICY="$SHARDED_REVEAL_POLICY" \
        SHARDED_ALLOW_UNTAGGED_FINAL="$SHARDED_ALLOW_UNTAGGED_FINAL" \
        MAX_TURNS="$MAX_TURNS" \
        MAX_TOKENS="$EVAL_MAX_TOKENS" \
        TEMPERATURE="$TEMPERATURE" \
        BATCH_SIZE=1 \
        NUM_SAMPLES=1 \
        "$PROJECT_ROOT/run_tinker_eval.sh" "$EVAL_RUN" 2>&1 | tee "$EVAL_LOG"
fi

log "Final eval metrics:"
"$PYTHON_BIN" - <<'PY' "$EVAL_METRICS"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
print(
    f"{metrics['run_name']}: reward_mean={metrics['reward_mean']:.4f} "
    f"format_error_rate={metrics['format_error_rate']:.4f} examples={metrics['examples']}"
)
PY
log "Strict clarification-gated dense train/eval finished"
