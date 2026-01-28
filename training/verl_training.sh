#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="${PROJECT_ROOT:-$( cd "$SCRIPT_DIR/.." && pwd )}"

unset VLLM_ATTENTION_BACKEND
if [ -n "${VLLM_USE_V1:-}" ]; then
    export VLLM_USE_V1
fi
export PYTHONBUFFERED=1
export PYTHONUNBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

export WANDB_ENTITY="${WANDB_ENTITY:-sample-efficient-rlvr}" # team
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DEFAULT_HF_HOME="${DEFAULT_HF_HOME:-/scratch/jeisner1/ssaigan1/.cache/huggingface}"
if [ -z "${HF_HOME:-}" ] && [ -d "$DEFAULT_HF_HOME/hub" ]; then
    export HF_HOME="$DEFAULT_HF_HOME"
else
    export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/css/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python)"
fi

export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Arguments: $@"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
