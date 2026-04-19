#!/bin/bash
# ===========================================================================
# UBAI cpu-gpu-cluster — gpu6 (A10) partition
# Reference: UBAI 서버사용법.pptx slide 21 (SBATCH template)
#            cpu-gpu-cluster 접속 가이드 v2.0.2.pdf
# Partition→GPU map (pptx slide 27):
#   gpu1/gpu3 → RTX3090   gpu4/gpu5 → A6000   gpu6 → A10   cpu → no GRES
# ===========================================================================
#SBATCH --job-name=halo8-ff
#SBATCH --output=logs/halo8_%j.out
#SBATCH --error=logs/halo8_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ===========================================================================
# Configurable parameters — override any at submission time:
#   PARAM=value sbatch run_halo8_slurm.sh
#
# To change the partition:    sbatch --partition=<name> run_halo8_slurm.sh
# To request more A10 GPUs:   sbatch --gres=gpu:a10:2 run_halo8_slurm.sh
# To resume from checkpoint:  RESUME_FROM=/path/to/ckpt.ckpt sbatch ...
# To use the full dataset:    DATA_LIMIT=0 sbatch ...
# ===========================================================================

# ---- Data / paths ----------------------------------------------------------
DATA_LIMIT=${DATA_LIMIT:-100}      # reaction groups to sample; 0 = full dataset
DATASET_PREFIX=${DATASET_PREFIX:-"Halogen"}   # "Halogen", "T1x", or "Mix"
REPO_DIR=${REPO_DIR:-"${SLURM_SUBMIT_DIR:-$(pwd)}"}
CONDA_ENV=${CONDA_ENV:-"reactot"}

# Canonical data location on UBAI: ~/projects/ts_prediction_project/data
# The trainer's default path is <repo>/reactot/dataset/Halo8 — we symlink the
# canonical directory into that location (auto-created below) so both the
# trainer and this script agree without hard-coded absolute paths.
HALO8_SOURCE=${HALO8_SOURCE:-"$HOME/projects/ts_prediction_project/data"}
HALO8_DATADIR=${HALO8_DATADIR:-"$REPO_DIR/reactot/dataset/Halo8"}

# ---- Resume from a specific checkpoint (empty = start fresh) ---------------
RESUME_FROM=${RESUME_FROM:-""}

# ---- HPC module names — disabled: cluster does not use module load for CUDA/cuDNN/Conda ----
# CUDA_MODULE=${CUDA_MODULE:-"cuda/11.8"}
# CUDNN_MODULE=${CUDNN_MODULE:-"cudnn/8.6"}
# CONDA_MODULE=${CONDA_MODULE:-"anaconda3"}

# ===========================================================================
# Pre-flight header
# ===========================================================================
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "DATA_LIMIT     : $DATA_LIMIT"
echo "DATASET_PREFIX : $DATASET_PREFIX"
echo "REPO_DIR       : $REPO_DIR"
echo "CONDA_ENV      : $CONDA_ENV"
echo "HALO8_SOURCE   : $HALO8_SOURCE"
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
# HPC modules — skipped: cluster does not use module load for Conda/cuDNN
# ===========================================================================
echo "INFO: module load disabled — relying on pre-activated environment and PATH"

# ===========================================================================
# Activate conda environment
# ===========================================================================
# Source the conda shell hook so 'conda activate' works inside batch scripts.
if conda_base=$(conda info --base 2>/dev/null); then
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" \
        || { echo "ERROR: 'conda activate $CONDA_ENV' failed"; exit 1; }
else
    echo "ERROR: 'conda info --base' failed — conda not found in PATH"
    exit 1
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
# ---------------------------------------------------------------------------
# If HALO8_DATADIR is missing / broken and HALO8_SOURCE is a real directory,
# create the symlink automatically so the trainer's default relative path
# (reactot/dataset/Halo8) resolves to the canonical data location.
# Equivalent to the manual setup:
#   rm -rf <repo>/reactot/dataset/Halo8
#   ln -s $HOME/projects/ts_prediction_project/data <repo>/reactot/dataset/Halo8
# ===========================================================================
if [ ! -e "$HALO8_DATADIR" ]; then
    if [ -d "$HALO8_SOURCE" ]; then
        echo "INFO: Linking $HALO8_DATADIR -> $HALO8_SOURCE"
        mkdir -p "$(dirname "$HALO8_DATADIR")"
        ln -sfn "$HALO8_SOURCE" "$HALO8_DATADIR" \
            || { echo "ERROR: Failed to create symlink $HALO8_DATADIR -> $HALO8_SOURCE"; exit 1; }
    else
        echo "ERROR: Neither $HALO8_DATADIR nor $HALO8_SOURCE exists."
        echo "       Place data under ~/projects/ts_prediction_project/data or"
        echo "       override HALO8_SOURCE / HALO8_DATADIR at submission time."
        exit 1
    fi
fi

DB_COUNT=$(find -L "$HALO8_DATADIR" -maxdepth 1 -name "Halo*.db" 2>/dev/null | wc -l)
if [ "$DB_COUNT" -eq 0 ]; then
    echo "ERROR: No Halo*.db files found in $HALO8_DATADIR"
    echo "       Resolved via symlink? $(readlink -f "$HALO8_DATADIR" 2>/dev/null || echo "n/a")"
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
export DATASET_PREFIX
export HALO8_DATADIR
export NUM_WORKERS
export RESUME_FROM

# ===========================================================================
# Launch training
# ===========================================================================
TRAIN_SCRIPT="$REPO_DIR/reactot/trainer/train_rpsb_ts1x.py"

echo "=========================================="
echo "Starting training at $(date)"
echo "Script  : $TRAIN_SCRIPT"
echo "=========================================="

python -u "$TRAIN_SCRIPT" --dataset Halo8

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at $(date) — exit code $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
