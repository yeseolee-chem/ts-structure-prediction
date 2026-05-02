#!/bin/bash
# ===========================================================================
# run_mix_slurm.sh — T1x + Halogen mix training on UBAI gpu6 (A10)
# ---------------------------------------------------------------------------
# Self-contained SLURM script (no delegation). Trains React-OT on the Halo8
# LMDB using BOTH T1x-prefixed and Halogen-prefixed reactions (C, H, N, O, F,
# S, Cl, Br). Paired with run_t1x_slurm.sh this gives the dataset-vs-dataset
# comparison: baseline (T1x only) vs augmented (T1x + halogens).
#
# Usage:
#   sbatch run_mix_slurm.sh                            # DATA_LIMIT=300 default
#   DATA_LIMIT=1000 sbatch run_mix_slurm.sh            # larger subset
#   DATA_LIMIT=0    sbatch run_mix_slurm.sh            # full dataset
#
# Halo8 data location: $HOME/projects/ts_prediction_project/data
# Auto-linked into <repo>/reactot/dataset/Halo8 at job start.
# ===========================================================================
#SBATCH --job-name=reactot-mix
#SBATCH --output=logs/mix_%j.out
#SBATCH --error=logs/mix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# =============== SCRIPT_VERSION: reactof-halo8 / self-contained ============
# If you DO NOT see the line above in the job log, you are running a stale
# copy — on the server: `cd ~/projects/ts-structure-prediction && git pull`.
# ===========================================================================
echo ">>> run_mix_slurm.sh VERSION: reactof-halo8 self-contained (no delegation, unique-RUN_NAME)"

# ---- Fixed dataset selector for this wrapper -------------------------------
DATASET_PREFIX="Mix"
WRAPPER_TAG="mix"

# ---- Configurable parameters -----------------------------------------------
DATA_LIMIT=${DATA_LIMIT:-300}
REPO_DIR=${REPO_DIR:-"${SLURM_SUBMIT_DIR:-$(pwd)}"}
CONDA_ENV=${CONDA_ENV:-"reactot"}
HALO8_SOURCE=${HALO8_SOURCE:-"$HOME/projects/ts_prediction_project/data"}
HALO8_DATADIR=${HALO8_DATADIR:-"$REPO_DIR/reactot/dataset/Halo8"}
RESUME_FROM=${RESUME_FROM:-""}
PROJECT_NAME=${PROJECT_NAME:-"RPSB-FT-Schedule"}

# ---- Branch / experiment identifier ---------------------------------------
EXPERIMENT_ID=${EXPERIMENT_ID:-"$(cd "$REPO_DIR" 2>/dev/null && git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-' || echo 'unknown-branch')"}

# ---- Globally unique JOB_TAG ----------------------------------------------
# SLURM_JOB_ID is unique across the cluster, so embedding it in RUN_NAME
# guarantees no two parallel jobs ever share a checkpoint dir even with the
# SAME wrapper, prefix, data limit, and branch. The 8-char random suffix is
# a belt-and-suspenders safety net for non-SLURM smoke tests where
# SLURM_JOB_ID is unset.
JOB_TAG_ID=${SLURM_JOB_ID:-local}
JOB_TAG_RAND=$(python -c 'import uuid; print(uuid.uuid4().hex[:8])' 2>/dev/null || printf '%s%s' "$$" "$(date +%N 2>/dev/null | head -c 6)")
JOB_TAG="job${JOB_TAG_ID}-${JOB_TAG_RAND}"

# ---- Globally unique RUN_NAME ---------------------------------------------
# Format: mix-<prefix>-dl<limit>-<branch>-job<slurm_id>-<rand8>
# Every component is in the name so the directory itself tells you which
# wrapper, dataset, data-limit, branch, and job produced the checkpoint.
# Override RUN_NAME=... at submit time when you intentionally want to
# resume into a specific previous directory.
RUN_NAME=${RUN_NAME:-"${WRAPPER_TAG}-${DATASET_PREFIX}-dl${DATA_LIMIT}-${EXPERIMENT_ID}-${JOB_TAG}"}

echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "DATA_LIMIT     : $DATA_LIMIT"
echo "DATASET_PREFIX : $DATASET_PREFIX (T1x + Halogen mix wrapper)"
echo "REPO_DIR       : $REPO_DIR"
echo "CONDA_ENV      : $CONDA_ENV"
echo "HALO8_SOURCE   : $HALO8_SOURCE"
echo "HALO8_DATADIR  : $HALO8_DATADIR"
echo "EXPERIMENT_ID  : $EXPERIMENT_ID"
echo "JOB_TAG        : $JOB_TAG"
echo "RUN_NAME       : $RUN_NAME"
echo "PROJECT_NAME   : $PROJECT_NAME"
echo "RESUME_FROM    : ${RESUME_FROM:-<none>}"
echo "=========================================="

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/checkpoint" "$REPO_DIR/results" \
    || { echo "ERROR: Cannot create output directories under $REPO_DIR"; exit 1; }
cd "$REPO_DIR" || { echo "ERROR: Cannot cd to REPO_DIR=$REPO_DIR"; exit 1; }

if conda_base=$(conda info --base 2>/dev/null); then
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" \
        || { echo "ERROR: 'conda activate $CONDA_ENV' failed"; exit 1; }
else
    echo "ERROR: 'conda info --base' failed — conda not found in PATH"
    exit 1
fi
echo "Python  : $(which python)"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "GPUs    : $GPU_COUNT available"
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No CUDA GPUs detected."
    exit 1
fi

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
export RUN_NAME PROJECT_NAME EXPERIMENT_ID
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
