#!/bin/bash
#SBATCH --job-name=halo8-ff
#SBATCH --output=logs/halo8_%j.out
#SBATCH --error=logs/halo8_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu4

# ===========================================================================
# Configurable parameters — override any at submission time:
#   PARAM=value sbatch run_halo8_slurm.sh
#
# To change the partition:    sbatch --partition=<name> run_halo8_slurm.sh
# To resume from checkpoint:  RESUME_FROM=/path/to/ckpt.ckpt sbatch ...
# To use the full dataset:    DATA_LIMIT=0 sbatch ...
# ===========================================================================

# ---- Data / paths ----------------------------------------------------------
DATA_LIMIT=${DATA_LIMIT:-100}      # rows per .db file; 0 = full dataset
REPO_DIR=${REPO_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
CONDA_ENV=${CONDA_ENV:-"reactot"}
HALO8_DATADIR=${HALO8_DATADIR:-"$REPO_DIR/reactot/dataset/Halo8"}

# ---- Resume from a specific checkpoint (empty = start fresh) ---------------
RESUME_FROM=${RESUME_FROM:-""}

# ---- HPC module names (set to empty string to skip loading) ----------------
CUDA_MODULE=${CUDA_MODULE:-"cuda/11.8"}
CUDNN_MODULE=${CUDNN_MODULE:-"cudnn/8.6"}
CONDA_MODULE=${CONDA_MODULE:-"anaconda3"}

# ===========================================================================
# Pre-flight header
# ===========================================================================
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "DATA_LIMIT     : $DATA_LIMIT"
echo "REPO_DIR       : $REPO_DIR"
echo "CONDA_ENV      : $CONDA_ENV"
echo "HALO8_DATADIR  : $HALO8_DATADIR"
echo "RESUME_FROM    : ${RESUME_FROM:-<none>}"
echo "=========================================="

# ===========================================================================
# Create output directories
# ===========================================================================
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/checkpoint" "$REPO_DIR/results" \
    || { echo "ERROR: Cannot create output directories under $REPO_DIR"; exit 1; }

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to REPO_DIR=$REPO_DIR"; exit 1; }

# ===========================================================================
# Load HPC modules (skipped silently if 'module' is unavailable)
# ===========================================================================
if command -v module &>/dev/null; then
    module purge
    [ -n "$CUDA_MODULE" ]  && module load "$CUDA_MODULE"
    [ -n "$CUDNN_MODULE" ] && module load "$CUDNN_MODULE"
    [ -n "$CONDA_MODULE" ] && module load "$CONDA_MODULE"
else
    echo "INFO: 'module' command not found — skipping module load (using PATH as-is)"
fi

# ===========================================================================
# Activate conda environment
# ===========================================================================
# Use the proper shell hook so 'conda activate' works inside batch scripts.
if conda_base=$(conda info --base 2>/dev/null); then
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" \
        || { echo "ERROR: 'conda activate $CONDA_ENV' failed"; exit 1; }
else
    # Legacy fallback (older conda / cluster setups)
    # shellcheck disable=SC1091
    source activate "$CONDA_ENV" \
        || { echo "ERROR: 'source activate $CONDA_ENV' failed"; exit 1; }
fi

echo "Python  : $(which python)"
echo "Env     : $(conda info --name 2>/dev/null || echo $CONDA_ENV)"

# ===========================================================================
# Verify GPU availability
# ===========================================================================
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "GPUs    : $GPU_COUNT available"
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No CUDA GPUs detected. Verify CUDA installation and --gres directive."
    exit 1
fi

# ===========================================================================
# Verify data directory
# ===========================================================================
DB_COUNT=$(find "$HALO8_DATADIR" -maxdepth 1 -name "Halo*.db" 2>/dev/null | wc -l)
if [ "$DB_COUNT" -eq 0 ]; then
    echo "ERROR: No Halo*.db files found in $HALO8_DATADIR"
    echo "       Set HALO8_DATADIR=<path> or place data under reactot/dataset/Halo8/"
    exit 1
fi
echo "Data    : $HALO8_DATADIR ($DB_COUNT Halo*.db files)"

# ===========================================================================
# Derive optimal DataLoader num_workers from CPU / GPU allocation
# Typically: workers_per_gpu = floor(cpus_per_task / num_gpus)
# ===========================================================================
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
NUM_WORKERS=$(python -c "print(max(1, $CPUS_PER_TASK // max(1, $GPU_COUNT)))")
echo "Workers : $NUM_WORKERS per GPU (CPUs=$CPUS_PER_TASK, GPUs=$GPU_COUNT)"

# ===========================================================================
# WandB: fall back to offline mode when no API key is available
# (offline runs are synced later with `wandb sync`)
# ===========================================================================
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "INFO: WANDB_API_KEY not set — WANDB_MODE=$WANDB_MODE"
fi

# ===========================================================================
# Export all configuration to the training script via environment variables
# ===========================================================================
export DATA_LIMIT
export HALO8_DATADIR
export NUM_WORKERS
export RESUME_FROM

# ===========================================================================
# Launch training
# ===========================================================================
TRAIN_SCRIPT="$REPO_DIR/reactot/trainer/train_halo8.py"

echo "=========================================="
echo "Starting training at $(date)"
echo "Script  : $TRAIN_SCRIPT"
echo "=========================================="

python -u "$TRAIN_SCRIPT"

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at $(date) — exit code $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
