#!/bin/bash
set -euo pipefail

# Usage: ./run_local_grpo.sh [experiment_name_suffix]
#
# Environment overrides:
#   MODEL_PATH, DATA_PATH, TRAIN_BATCH_SIZE, ROLLOUT_BATCH_SIZE,
#   MINI_BATCH_SIZE, N_GPUS_PER_NODE, ROLLOUT_TP_SIZE, ROLLOUT_BACKEND,
#   MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, TOTAL_TRAINING_STEPS

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script is located.
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/css/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Could not find executable uv venv Python at $PYTHON_BIN"
    exit 1
fi

export PATH="$PROJECT_ROOT/css/bin:$PATH"
export USER="${USER:-$(whoami)}"

DEFAULT_HF_HOME="${DEFAULT_HF_HOME:-/scratch/jeisner1/ssaigan1/.cache/huggingface}"
if [ -z "${HF_HOME:-}" ] && [ -d "$DEFAULT_HF_HOME/hub" ]; then
    export HF_HOME="$DEFAULT_HF_HOME"
else
    export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$PROJECT_ROOT/_logs"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true

CONFIG_NAME="${CONFIG_NAME:-baseline_grpo}"

# Default to ToolUse dataset
DATA_PATH="${DATA_PATH:-datasets/tooluse}"

# Conservative single-node defaults. Override these env vars for full sweeps.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-hf}"
if [ -z "${ROLLOUT_TP_SIZE:-}" ]; then
    if [ "$ROLLOUT_BACKEND" = "hf" ]; then
        ROLLOUT_TP_SIZE=1
    elif [ "$N_GPUS_PER_NODE" -ge 2 ]; then
        ROLLOUT_TP_SIZE=2
    else
        ROLLOUT_TP_SIZE=1
    fi
fi
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
VAL_ROLLOUT_BATCH_SIZE="${VAL_ROLLOUT_BATCH_SIZE:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-False}"
PREPROCESS_NUM_PROC="${PREPROCESS_NUM_PROC:-1}"

MIN_VALID_MINI_BATCH_SIZE=$(( (N_GPUS_PER_NODE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
if [ "$MINI_BATCH_SIZE" -lt "$MIN_VALID_MINI_BATCH_SIZE" ]; then
    echo "Increasing MINI_BATCH_SIZE from $MINI_BATCH_SIZE to $MIN_VALID_MINI_BATCH_SIZE so FSDP minibatch normalization stays positive."
    MINI_BATCH_SIZE="$MIN_VALID_MINI_BATCH_SIZE"
fi
if [ "$TRAIN_BATCH_SIZE" -lt "$MINI_BATCH_SIZE" ]; then
    echo "Increasing TRAIN_BATCH_SIZE from $TRAIN_BATCH_SIZE to $MINI_BATCH_SIZE so it can contain one PPO minibatch."
    TRAIN_BATCH_SIZE="$MINI_BATCH_SIZE"
fi

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_grpo"}

# =============================================================================
# EXECUTION
# =============================================================================

if [ ! -f "$PROJECT_ROOT/$DATA_PATH/train.parquet" ] || [ ! -f "$PROJECT_ROOT/$DATA_PATH/test.parquet" ]; then
    echo "Parquet files missing for $DATA_PATH; preprocessing JSON data first."
    "$PYTHON_BIN" "$PROJECT_ROOT/data/preprocess.py" --data_source "$PROJECT_ROOT/$DATA_PATH" --num_proc "$PREPROCESS_NUM_PROC"
fi

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/:' '--')
EXP_NAME="LOCAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}-${SUFFIX}"

ARGS=(
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.val_batch_size=$VAL_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "trainer.group_name=GRPO-local"
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.val_before_train=$TRAINER_VAL_BEFORE_TRAIN"
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    "trainer.nnodes=1"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "max_model_len=$MAX_MODEL_LEN"
    "custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.use_remove_padding=False"
    "actor_rollout_ref.actor.use_remove_padding=False"
    "++actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
    "++actor_rollout_ref.model.override_config.max_position_embeddings=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.name=$ROLLOUT_BACKEND"
    "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_BATCH_SIZE"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "algorithm.rollout_correction.rollout_is=token"
)

if [ "$ROLLOUT_BACKEND" = "sglang" ]; then
    ARGS+=("++actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$SGLANG_ATTENTION_BACKEND")
elif [ "$ROLLOUT_BACKEND" = "hf" ]; then
    ARGS+=("actor_rollout_ref.rollout.top_k=0")
    ARGS+=("actor_rollout_ref.rollout.val_kwargs.top_k=0")
fi

echo "----------------------------------------------------------------"
echo "Starting Local GRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Python: $PYTHON_BIN"
echo "GPUs: $N_GPUS_PER_NODE, rollout backend: $ROLLOUT_BACKEND, rollout TP: $ROLLOUT_TP_SIZE"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"
