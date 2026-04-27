#!/bin/bash
set -euo pipefail

# Usage: ./run_tinker_grpo.sh [run_name]
#
# Required:
#   export TINKER_API_KEY=...
#
# Common overrides:
#   MODEL_NAME, DATA_PATH, BATCH_SIZE, ROLLOUT_N, MAX_STEPS,
#   MAX_TOKENS, TEMPERATURE, LR, LORA_RANK, RENDERER_NAME, TINKER_BASE_URL

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

RUN_NAME="${1:-${RUN_NAME:-tinker-grpo-smoke}}"
export RUN_NAME

DATA_PATH="${DATA_PATH:-datasets/tooluse}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
RENDERER_NAME="${RENDERER_NAME:-}"
TINKER_LOG_DIR="${TINKER_LOG_DIR:-$PROJECT_ROOT/_logs/tinker_grpo}"

echo "----------------------------------------------------------------"
echo "Starting Tinker GRPO Training"
echo "Run: $RUN_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
if [ -n "$RENDERER_NAME" ]; then
    echo "Renderer: $RENDERER_NAME"
fi
echo "Python: $PYTHON_BIN"
echo "Batch size: $BATCH_SIZE, rollout_n: $ROLLOUT_N, max_steps: $MAX_STEPS"
echo "----------------------------------------------------------------"

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tinker_grpo.py"
    --run-name "$RUN_NAME"
    --data-path "$DATA_PATH"
    --model-name "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --rollout-n "$ROLLOUT_N"
    --max-steps "$MAX_STEPS"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --learning-rate "$LR"
    --lora-rank "$LORA_RANK"
    --log-dir "$TINKER_LOG_DIR"
)

if [ -n "$RENDERER_NAME" ]; then
    CMD+=(--renderer-name "$RENDERER_NAME")
fi

"${CMD[@]}"
