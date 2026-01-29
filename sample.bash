#!/bin/bash
#SBATCH --job-name=grpo-local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=2:30:00
#SBATCH --output=_logs/%x-%j.log
#SBATCH --partition=a100

set -euo pipefail

echo "Start time: $(date)"
start_time=$(date +%s)
current_date=$(date +%y%m%d)

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"
mkdir -p _logs

export PYTHON_BIN="$PROJECT_ROOT/css/bin/python"
export PATH="$PROJECT_ROOT/css/bin:$PATH"
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

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
if [ -z "${ROLLOUT_TP_SIZE:-}" ]; then
    if [ "${ROLLOUT_BACKEND:-hf}" = "hf" ]; then
        export ROLLOUT_TP_SIZE=1
    elif [ "$N_GPUS_PER_NODE" -ge 2 ]; then
        export ROLLOUT_TP_SIZE=2
    else
        export ROLLOUT_TP_SIZE=1
    fi
fi
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-hf}"

bash "$PROJECT_ROOT/run_local_grpo.sh" "${1:-sample-${current_date}}"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "End time: $(date)"
elapsed_hms=$(printf '%02d:%02d:%02d\n' $((elapsed_time/3600)) $(((elapsed_time%3600)/60)) $((elapsed_time%60)))
echo "Run completed in $elapsed_hms"
