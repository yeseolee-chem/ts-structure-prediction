#!/bin/bash
# ===========================================================================
# run_t1x_slurm.sh — T1x-only training on UBAI gpu6 (A10)
# ---------------------------------------------------------------------------
# Self-contained SLURM script (no delegation). Trains React-OT on the Halo8
# LMDB using ONLY the T1x-prefixed reactions (Transition1x subset: C, H, N,
# O). Baseline for the comparison against run_mix_slurm.sh (T1x + Halogen).
#
# Usage:
#   sbatch run_t1x_slurm.sh                            # DATA_LIMIT=300 default
#   DATA_LIMIT=1000 sbatch run_t1x_slurm.sh            # larger subset
#   DATA_LIMIT=0    sbatch run_t1x_slurm.sh            # full dataset
#
# Halo8 data location: $HOME/projects/ts_prediction_project/data
# Auto-linked into <repo>/reactot/dataset/Halo8 at job start.
# ===========================================================================
#SBATCH --job-name=reactot-t1x
#SBATCH --output=logs/t1x_%j.out
#SBATCH --error=logs/t1x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# =============== SCRIPT_VERSION: reactof-halo8 / self-contained ============
# If you DO NOT see the line above in the job log, you are running a stale
# copy — on the server: `cd ~/projects/ts-structure-prediction && git pull`.
# ===========================================================================
echo ">>> run_t1x_slurm.sh VERSION: reactof-halo8 self-contained (no delegation)"

# ---- Fixed dataset selector for this wrapper -------------------------------
DATASET_PREFIX="T1x"

# ---- Configurable parameters -----------------------------------------------
DATA_LIMIT=${DATA_LIMIT:-300}
REPO_DIR=${REPO_DIR:-"${SLURM_SUBMIT_DIR:-$(pwd)}"}
CONDA_ENV=${CONDA_ENV:-"reactot"}
HALO8_SOURCE=${HALO8_SOURCE:-"$HOME/projects/ts_prediction_project/data"}
HALO8_DATADIR=${HALO8_DATADIR:-"$REPO_DIR/reactot/dataset/Halo8"}
RESUME_FROM=${RESUME_FROM:-""}

echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "DATA_LIMIT     : $DATA_LIMIT"
echo "DATASET_PREFIX : $DATASET_PREFIX (T1x-only wrapper)"
echo "REPO_DIR       : $REPO_DIR"
echo "CONDA_ENV      : $CONDA_ENV"
echo "HALO8_SOURCE   : $HALO8_SOURCE"
echo "HALO8_DATADIR  : $HALO8_DATADIR"
echo "RESUME_FROM    : ${RESUME_FROM:-<none>}"
echo "=========================================="

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/checkpoint" "$REPO_DIR/results" \
    || { echo "ERROR: Cannot create output directories under $REPO_DIR"; exit 1; }
cd "$REPO_DIR" || { echo "ERROR: Cannot cd to REPO_DIR=$REPO_DIR"; exit 1; }

# Activate conda env
if conda_base=$(conda info --base 2>/dev/null); then
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" \
        || { echo "ERROR: 'conda activate $CONDA_ENV' failed"; exit 1; }
else
    echo "ERROR: 'conda info --base' failed — conda not found in PATH"
    exit 1
fi
echo "Python  : $(which python)"

# GPU check
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "GPUs    : $GPU_COUNT available"
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No CUDA GPUs detected."
    exit 1
fi

# Auto-link Halo8 data if missing
if [ ! -e "$HALO8_DATADIR" ]; then
    if [ -d "$HALO8_SOURCE" ]; then
        echo "INFO: Linking $HALO8_DATADIR -> $HALO8_SOURCE"
        mkdir -p "$(dirname "$HALO8_DATADIR")"
        ln -sfn "$HALO8_SOURCE" "$HALO8_DATADIR" \
            || { echo "ERROR: Failed to create symlink $HALO8_DATADIR -> $HALO8_SOURCE"; exit 1; }
    else
        echo "ERROR: Neither $HALO8_DATADIR nor $HALO8_SOURCE exists."
        exit 1
    fi
fi

DB_COUNT=$(find -L "$HALO8_DATADIR" -maxdepth 1 -name "Halo*.db" 2>/dev/null | wc -l)
if [ "$DB_COUNT" -eq 0 ]; then
    echo "ERROR: No Halo*.db files found in $HALO8_DATADIR"
    exit 1
fi
echo "Data    : $HALO8_DATADIR ($DB_COUNT Halo*.db files)"

CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
NUM_WORKERS=$(python -c "print(max(1, $CPUS_PER_TASK // max(1, $GPU_COUNT)))")
echo "Workers : $NUM_WORKERS per GPU (CPUs=$CPUS_PER_TASK, GPUs=$GPU_COUNT)"

if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "INFO: WANDB_API_KEY not set — WANDB_MODE=$WANDB_MODE"
fi

export DATA_LIMIT DATASET_PREFIX HALO8_DATADIR NUM_WORKERS RESUME_FROM
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "=========================================="
echo "Starting training at $(date)"
echo "PYTHONPATH : $PYTHONPATH"
echo "=========================================="

WANDB_FLAG="--no-wandb"
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAG=""
fi

python -u -m reactot.trainer.train_rpsb_ts1x --dataset Halo8 $WANDB_FLAG
EXIT_CODE=$?

echo "=========================================="
echo "Training finished at $(date) — exit code $EXIT_CODE"
echo "=========================================="

if [ -n "$WANDB_FLAG" ]; then
    echo "Generating plots from CSV metrics..."
    python -u plot_metrics.py || echo "WARN: plot_metrics.py failed (non-fatal)"
fi

exit $EXIT_CODE
