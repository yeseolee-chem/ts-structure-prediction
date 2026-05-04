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

# Strip stale __pycache__ before any Python invocation. Switching between
# cb-* / rp-* branches in a shared working dir leaves .pyc files compiled
# from a different branch's source (e.g. cb-D's egnn_dynamics with a
# learn_importance kwarg the current branch doesn't accept). Python's
# mtime-based invalidation does not always notice on this filesystem, so
# the stale .pyc gets imported instead — producing both stale-import
# crashes and silently identical val_rmsd metrics across "different"
# experiments. Always purge first.
find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

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

# ===========================================================================
# Branch-specific activation flags
# ---------------------------------------------------------------------------
# Each cb-* branch implements a different atom-weighting scheme. Some
# branches activate their scheme by default (cb-A, cb-AB, cb-C, cb-D,
# cb-CD, cb-ABD), but others gate activation behind a CLI flag whose
# default leaves the run looking like a different branch's result.
#
# Triaged 2026-05-04 from jobs 611956-611962 where:
#   - cb-B without --element-aware-weights produced bit-identical val_loss
#     to cb-A. cb-B trainer's --element-aware-weights default is False, so
#     element_aware_weights stays off and the run reduces to cb-A's pure
#     graph-distance code path.
#   - cb-BCD without --prior-scheme/--learn-importance produced bit-
#     identical val_loss to reactot-halo8. cb-BCD trainer skips its prior
#     block entirely when args.prior_scheme is None, so no atom-weighting
#     is applied at all.
#   - cb-BC inherits the cb-AB --weighting-scheme default of "AB" — added
#     defensively so the BC kernel is not silently bypassed even though
#     cb-BC was not in the failing batch.
#
# The earlier rounds of fixes (purge stale .pyc, EXPERIMENT_ID guard,
# unique RUN_NAME, per-worktree submission, output centralization via
# symlinks) addressed plumbing — code import path, dir collisions, log
# routing — but none of them activates the branch-specific weighting.
# Without this case statement every "different" cb-* run can silently
# collapse to whichever default its trainer ships with.
#
# When adding a new branch, append a case here. The catch-all ('*')
# warns rather than failing so unrelated branches (claude/*, rp-*, main)
# keep working without manual updates.
# ===========================================================================
EXTRA_FLAGS=""
case "$EXPERIMENT_ID" in
    reactot-halo8)
        ;;
    cb-A|cb-AB|cb-C|cb-D|cb-ABD|cb-CD)
        ;;
    cb-B)
        EXTRA_FLAGS="--element-aware-weights"
        ;;
    cb-BC)
        EXTRA_FLAGS="--weighting-scheme BC"
        ;;
    cb-BCD)
        EXTRA_FLAGS="--prior-scheme BC --learn-importance"
        ;;
    rp-*|claude-*|main)
        ;;
    *)
        echo "WARN: unknown EXPERIMENT_ID='$EXPERIMENT_ID' — no branch-specific flags applied."
        ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Branch flags : $EXTRA_FLAGS"
fi

python -u -m reactot.trainer.train_rpsb_ts1x --dataset Halo8 $WANDB_FLAG $EXTRA_FLAGS
EXIT_CODE=$?

echo "=========================================="
echo "Training finished at $(date) — exit code $EXIT_CODE"
echo "=========================================="

if [ -n "$WANDB_FLAG" ]; then
    echo "Generating plots from CSV metrics..."
    python -u plot_metrics.py || echo "WARN: plot_metrics.py failed (non-fatal)"
fi

exit $EXIT_CODE
