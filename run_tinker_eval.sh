#!/bin/bash
set -euo pipefail

# Usage: ./run_tinker_eval.sh [run_name]
#
# Required:
#   export TINKER_API_KEY=...
#
# Common overrides:
#   MODEL_NAME, MODEL_PATH, DATA_PATH, SPLIT, BATCH_SIZE, NUM_SAMPLES, MAX_TURNS,
#   MAX_EXAMPLES, MAX_TOKENS, TEMPERATURE, RENDERER_NAME, TINKER_BASE_URL

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "TINKER_API_KEY is not set. Export it in your shell; do not put it in the repo."
    exit 1
fi

if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "$PROJECT_ROOT/css/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/css/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

RUN_NAME="${1:-${RUN_NAME:-tinker-eval}}"
export RUN_NAME

DATA_PATH="${DATA_PATH:-datasets/tooluse}"
SPLIT="${SPLIT:-test}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_TURNS="${MAX_TURNS:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RENDERER_NAME="${RENDERER_NAME:-}"
TINKER_EVAL_LOG_DIR="${TINKER_EVAL_LOG_DIR:-$PROJECT_ROOT/_logs/tinker_eval}"

echo "----------------------------------------------------------------"
echo "Starting Tinker Evaluation"
echo "Run: $RUN_NAME"
echo "Data: $DATA_PATH ($SPLIT)"
if [ -n "$MODEL_PATH" ]; then
    echo "Model path: $MODEL_PATH"
else
    echo "Model: $MODEL_NAME"
fi
if [ -n "$RENDERER_NAME" ]; then
    echo "Renderer: $RENDERER_NAME"
fi
echo "Python: $PYTHON_BIN"
echo "Batch size: $BATCH_SIZE, num_samples: $NUM_SAMPLES, max_examples: $MAX_EXAMPLES, max_turns: $MAX_TURNS"
echo "----------------------------------------------------------------"

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tinker_eval.py"
    --run-name "$RUN_NAME"
    --data-path "$DATA_PATH"
    --split "$SPLIT"
    --model-name "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --num-samples "$NUM_SAMPLES"
    --max-examples "$MAX_EXAMPLES"
    --max-turns "$MAX_TURNS"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --log-dir "$TINKER_EVAL_LOG_DIR"
)

if [ -n "$MODEL_PATH" ]; then
    CMD+=(--model-path "$MODEL_PATH")
fi

if [ -n "$RENDERER_NAME" ]; then
    CMD+=(--renderer-name "$RENDERER_NAME")
fi

"${CMD[@]}"
