#!/bin/bash
set -euo pipefail

# Usage: ./run_tinker_grpo.sh [run_name]
#
# Required:
#   export TINKER_API_KEY=...
#
# Common overrides:
#   MODEL_NAME, DATA_PATH, BATCH_SIZE, ROLLOUT_N, MAX_STEPS, MAX_TURNS,
#   MAX_TOKENS, TEMPERATURE, LR, LORA_RANK, RENDERER_NAME, TINKER_BASE_URL,
#   SHUFFLE_SEED, SHARDED_REWARD_MODE, SDPO_DISTILL_WEIGHT, SDPO_TOPK,
#   SDPO_SKIP_FIRST_N_TOKENS, SDPO_MAX_TEACHER_TOKENS,
#   SDPO_TEACHER_TEMPERATURE, SDPO_DISTILL_ON, SHARDED_PROMPT_STYLE,
#   SHARDED_REVEAL_POLICY, SHARDED_ALLOW_UNTAGGED_FINAL, SDPO_TEACHER_PROMPT_STYLE

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
MAX_TURNS="${MAX_TURNS:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SHUFFLE_SEED="${SHUFFLE_SEED:--1}"
SHARDED_REWARD_MODE="${SHARDED_REWARD_MODE:-sparse}"
SHARDED_PROMPT_STYLE="${SHARDED_PROMPT_STYLE:-default}"
SHARDED_REVEAL_POLICY="${SHARDED_REVEAL_POLICY:-always}"
SHARDED_ALLOW_UNTAGGED_FINAL="${SHARDED_ALLOW_UNTAGGED_FINAL:-0}"
SDPO_TEACHER_PROMPT_STYLE="${SDPO_TEACHER_PROMPT_STYLE:-minimal_teacher}"
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
echo "Batch size: $BATCH_SIZE, rollout_n: $ROLLOUT_N, max_steps: $MAX_STEPS, max_turns: $MAX_TURNS"
echo "Sharded reward mode: $SHARDED_REWARD_MODE"
echo "Sharded prompt style: $SHARDED_PROMPT_STYLE, reveal policy: $SHARDED_REVEAL_POLICY, allow untagged final: $SHARDED_ALLOW_UNTAGGED_FINAL"
if [ "$SHARDED_REWARD_MODE" = "sdpo" ]; then
    echo "SDPO distill weight: ${SDPO_DISTILL_WEIGHT:-0.1}, topk: ${SDPO_TOPK:-20}"
    echo "SDPO skip first tokens: ${SDPO_SKIP_FIRST_N_TOKENS:-3}, teacher max tokens: ${SDPO_MAX_TEACHER_TOKENS:-256}"
    echo "SDPO teacher temperature: ${SDPO_TEACHER_TEMPERATURE:-0.0}, distill on: ${SDPO_DISTILL_ON:-failed}, teacher prompt style: ${SDPO_TEACHER_PROMPT_STYLE:-minimal_teacher}"
fi
if [ "$SHUFFLE_SEED" -ge 0 ]; then
    echo "Shuffle seed: $SHUFFLE_SEED"
fi
echo "----------------------------------------------------------------"

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tinker_grpo.py"
    --run-name "$RUN_NAME"
    --data-path "$DATA_PATH"
    --model-name "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --rollout-n "$ROLLOUT_N"
    --max-steps "$MAX_STEPS"
    --max-turns "$MAX_TURNS"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --shuffle-seed "$SHUFFLE_SEED"
    --sharded-reward-mode "$SHARDED_REWARD_MODE"
    --sharded-prompt-style "$SHARDED_PROMPT_STYLE"
    --sharded-reveal-policy "$SHARDED_REVEAL_POLICY"
    --sdpo-teacher-prompt-style "$SDPO_TEACHER_PROMPT_STYLE"
    --learning-rate "$LR"
    --lora-rank "$LORA_RANK"
    --log-dir "$TINKER_LOG_DIR"
)

if [ "$SHARDED_ALLOW_UNTAGGED_FINAL" = "1" ] || [ "$SHARDED_ALLOW_UNTAGGED_FINAL" = "true" ]; then
    CMD+=(--sharded-allow-untagged-final)
fi

if [ -n "$RENDERER_NAME" ]; then
    CMD+=(--renderer-name "$RENDERER_NAME")
fi

"${CMD[@]}"
