#!/bin/bash
set -euo pipefail

# Usage: ./run_tinker_sft.sh [run_name]
#
# Required:
#   export TINKER_API_KEY=...
#
# Common overrides:
#   MODEL_NAME, DATA_PATH, BATCH_SIZE, MAX_STEPS, LR, LORA_RANK,
#   RENDERER_NAME, TINKER_BASE_URL, SHUFFLE_SEED, SHARDED_PROMPT_STYLE,
#   SHARDED_ALLOW_UNTAGGED_FINAL

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

RUN_NAME="${1:-${RUN_NAME:-tinker-sft-smoke}}"
export RUN_NAME

DATA_PATH="${DATA_PATH:-datasets/tooluse}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:--1}"
SHARDED_PROMPT_STYLE="${SHARDED_PROMPT_STYLE:-default}"
SHARDED_ALLOW_UNTAGGED_FINAL="${SHARDED_ALLOW_UNTAGGED_FINAL:-0}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
RENDERER_NAME="${RENDERER_NAME:-}"
TINKER_SFT_LOG_DIR="${TINKER_SFT_LOG_DIR:-$PROJECT_ROOT/_logs/tinker_sft}"

echo "----------------------------------------------------------------"
echo "Starting Tinker SFT Training"
echo "Run: $RUN_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
if [ -n "$RENDERER_NAME" ]; then
    echo "Renderer: $RENDERER_NAME"
fi
echo "Python: $PYTHON_BIN"
echo "Batch size: $BATCH_SIZE, max_steps: $MAX_STEPS"
echo "Sharded prompt style: $SHARDED_PROMPT_STYLE, allow untagged final: $SHARDED_ALLOW_UNTAGGED_FINAL"
if [ "$SHUFFLE_SEED" -ge 0 ]; then
    echo "Shuffle seed: $SHUFFLE_SEED"
fi
echo "----------------------------------------------------------------"

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tinker_sft.py"
    --run-name "$RUN_NAME"
    --data-path "$DATA_PATH"
    --model-name "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --shuffle-seed "$SHUFFLE_SEED"
    --sharded-prompt-style "$SHARDED_PROMPT_STYLE"
    --learning-rate "$LR"
    --lora-rank "$LORA_RANK"
    --log-dir "$TINKER_SFT_LOG_DIR"
)

if [ "$SHARDED_ALLOW_UNTAGGED_FINAL" = "1" ] || [ "$SHARDED_ALLOW_UNTAGGED_FINAL" = "true" ]; then
    CMD+=(--sharded-allow-untagged-final)
fi

if [ -n "$RENDERER_NAME" ]; then
    CMD+=(--renderer-name "$RENDERER_NAME")
fi

"${CMD[@]}"
