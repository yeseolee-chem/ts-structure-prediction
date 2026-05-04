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

echo ">>> run_halo8_slurm.sh VERSION: reactof-halo8 self-contained (unique-RUN_NAME)"

# ---- Wrapper identity ------------------------------------------------------
WRAPPER_TAG="halo8"

# ---- Data / paths ----------------------------------------------------------
DATA_LIMIT=${DATA_LIMIT:-300}      # reaction groups to sample; 0 = full dataset
                                    # 300 + 300 epochs ≈ 1h on A10 (iteration)
                                    # 1000 + 3000 epochs ≈ 29h (production-ish)
DATASET_PREFIX=${DATASET_PREFIX:-"Halogen"}   # "Halogen", "T1x", or "Mix"
REPO_DIR=${REPO_DIR:-"${SLURM_SUBMIT_DIR:-$(pwd)}"}
CONDA_ENV=${CONDA_ENV:-"reactot"}
PROJECT_NAME=${PROJECT_NAME:-"RPSB-FT-Schedule"}

# Canonical data location on UBAI: ~/projects/ts_prediction_project/data
# The trainer's default path is <repo>/reactot/dataset/Halo8 — we symlink the
# canonical directory into that location (auto-created below) so both the
# trainer and this script agree without hard-coded absolute paths.
HALO8_SOURCE=${HALO8_SOURCE:-"$HOME/projects/ts_prediction_project/data"}
HALO8_DATADIR=${HALO8_DATADIR:-"$REPO_DIR/reactot/dataset/Halo8"}

# ---- Branch / experiment identifier ---------------------------------------
# EXPERIMENT_ID disambiguates checkpoint directories across experimental
# branches that share the same DATASET_PREFIX/DATA_LIMIT. Without it,
# `reactot-halo8 mix dl500` and `cb-D mix dl500` both resolved to the same
# RUN_NAME ("halo8-Mix-dl500") and clobbered each other's last.ckpt — which
# is why their reported val_rmsd numbers were bit-identical.
#
# Default: derive from the current git branch (reactot-halo8, cb-A, cb-D, …).
# Override with EXPERIMENT_ID=<tag> sbatch ... to use a custom label.
EXPERIMENT_ID=${EXPERIMENT_ID:-"$(cd "$REPO_DIR" 2>/dev/null && git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-' || echo 'unknown-branch')"}

# ---- Globally unique JOB_TAG ----------------------------------------------
# SLURM_JOB_ID is unique across the cluster, so embedding it in RUN_NAME
# guarantees no two parallel jobs ever share a checkpoint dir even with the
# SAME wrapper, prefix, data limit, and branch — which used to happen when
# 3 parallel `sbatch run_halo8_slurm.sh` invocations all resolved to
# "halo8-Mix-dl500-cb-D" and stomped on each other's last.ckpt. The 8-char
# random suffix is a belt-and-suspenders safety net for non-SLURM smoke
# tests where SLURM_JOB_ID is unset, and for the unlikely case of two jobs
# on different clusters / partitions sharing a numeric id.
JOB_TAG_ID=${SLURM_JOB_ID:-local}
JOB_TAG_RAND=$(python -c 'import uuid; print(uuid.uuid4().hex[:8])' 2>/dev/null || printf '%s%s' "$$" "$(date +%N 2>/dev/null | head -c 6)")
JOB_TAG="job${JOB_TAG_ID}-${JOB_TAG_RAND}"

# ---- Stable resume key + globally unique RUN_NAME -------------------------
# STABLE_KEY identifies "the same logical experiment" — wrapper + dataset +
# data limit + branch — independent of which SLURM job processed it. Used
# below to find a previous run's last.ckpt for auto-resume on resubmission.
#
# RUN_NAME embeds JOB_TAG so parallel submissions with identical params
# never share a checkpoint dir. Override RUN_NAME=... at submit time only
# when you intentionally want to write into a specific directory (e.g.
# resume into one).
STABLE_KEY="${WRAPPER_TAG}-${DATASET_PREFIX}-dl${DATA_LIMIT}-${EXPERIMENT_ID}"
RUN_NAME=${RUN_NAME:-"${STABLE_KEY}-${JOB_TAG}"}

# ---- EarlyStopping defaults ------------------------------------------------
# Branches with weighted/hierarchical losses (cb-BC, cb-CD, plus the
# combined BCD/ABD variants) have noisier val_ep_scaled_err curves than
# vanilla MSE. Without min_delta, the metric keeps making microscopic
# "improvements" forever and EarlyStopping never fires — combined with
# max_epochs=-1 that means training runs indefinitely (we observed cb-BC /
# cb-CD still going at 14h while cb-A/B/C/D finished in 4-6h). A tiny
# positive min_delta forces a real improvement threshold.
#
# Override at submit time:  EARLY_STOP_MIN_DELTA=2e-4 sbatch ...
case "$EXPERIMENT_ID" in
    cb-BC|cb-CD|cb-BCD|cb-ABD)
        EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-1e-4}
        ;;
    *)
        EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
        ;;
esac
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-150}

# ---- Auto-resume on resubmission ------------------------------------------
# Explicit RESUME_FROM wins. Otherwise scan
# checkpoint/${PROJECT_NAME}/${STABLE_KEY}-job*/last.ckpt and pick the most
# recent. Because every submission gets a fresh JOB_TAG suffix, we can no
# longer rely on an exact RUN_NAME match — but the STABLE_KEY prefix still
# unambiguously identifies "the same logical experiment", so picking the
# newest matching last.ckpt is the right resume target.
RESUME_FROM=${RESUME_FROM:-""}
if [ -z "$RESUME_FROM" ]; then
    LATEST_CKPT=""
    for ckpt in "${REPO_DIR}/checkpoint/${PROJECT_NAME}/${STABLE_KEY}-"*/last.ckpt; do
        [ -f "$ckpt" ] || continue
        if [ -z "$LATEST_CKPT" ] || [ "$ckpt" -nt "$LATEST_CKPT" ]; then
            LATEST_CKPT="$ckpt"
        fi
    done
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_FROM="$LATEST_CKPT"
        echo "INFO: Auto-resuming from $RESUME_FROM"
    else
        echo "INFO: No previous checkpoint matching ${STABLE_KEY}-job*/last.ckpt — starting fresh"
    fi
else
    echo "INFO: Using explicit RESUME_FROM=$RESUME_FROM"
fi

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
echo "EXPERIMENT_ID  : $EXPERIMENT_ID"
echo "JOB_TAG        : $JOB_TAG"
echo "STABLE_KEY     : $STABLE_KEY"
echo "RUN_NAME       : $RUN_NAME"
echo "PROJECT_NAME   : $PROJECT_NAME"
echo "EARLY_STOP     : patience=$EARLY_STOP_PATIENCE min_delta=$EARLY_STOP_MIN_DELTA"
echo "RESUME_FROM    : ${RESUME_FROM:-<none>}"
echo "=========================================="

# ===========================================================================
# Create output directories
# ===========================================================================
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/checkpoint" "$REPO_DIR/results" \
    || { echo "ERROR: Cannot create output directories under $REPO_DIR"; exit 1; }

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to REPO_DIR=$REPO_DIR"; exit 1; }

# ===========================================================================
# Strip stale __pycache__ before any Python invocation
# ===========================================================================
# Switching between cb-* branches leaves .pyc files compiled from a
# different branch's source (e.g. cb-D's egnn_dynamics with learn_importance)
# and Python's mtime-based invalidation does not always notice on this
# filesystem, producing failures like:
#   TypeError: EGNNDynamics.__init__() got an unexpected keyword argument 'learn_importance'
find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

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
export RUN_NAME
export PROJECT_NAME
export EARLY_STOP_MIN_DELTA
export EARLY_STOP_PATIENCE
export EXPERIMENT_ID

# ===========================================================================
# Launch training
# ---------------------------------------------------------------------------
# Run as a module (-m) so that $REPO_DIR (the CWD) is prepended to sys.path
# and `import reactot` resolves without requiring `pip install -e .`.
# PYTHONPATH is also exported as a belt-and-suspenders fallback when users
# launch this script from a different working directory.
# ===========================================================================
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "=========================================="
echo "Starting training at $(date)"
echo "Module  : reactot.trainer.train_rpsb_ts1x"
echo "PYTHONPATH : $PYTHONPATH"
echo "=========================================="

# --no-wandb routes all PL metrics to logs/<run_name>/<run_name>/version_*/metrics.csv
# (CSVLogger) instead of wandb. Set USE_WANDB=1 at submission time to re-enable
# wandb: `USE_WANDB=1 sbatch run_halo8_slurm.sh` (requires WANDB_API_KEY).
WANDB_FLAG="--no-wandb"
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAG=""
fi

python -u -m reactot.trainer.train_rpsb_ts1x --dataset Halo8 $WANDB_FLAG

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at $(date) — exit code $EXIT_CODE"
echo "=========================================="

# ===========================================================================
# Post-training: convert the CSVLogger metrics to PNG plots so the results
# are viewable without wandb. `plot_metrics.py` auto-finds the newest
# metrics.csv under logs/ and drops PNGs next to it under plots/.
# Only runs when --no-wandb is active (i.e. WANDB_FLAG is non-empty and CSV
# metrics exist).
# ===========================================================================
if [ -n "$WANDB_FLAG" ]; then
    echo "=========================================="
    echo "Generating plots from CSV metrics..."
    echo "=========================================="
    # Pass --csv EXPLICITLY. plot_metrics.py's no-arg fallback walks logs/
    # and picks the newest metrics.csv by mtime — when 10 cb-* jobs share
    # one symlinked logs/ directory, that "newest" file is usually a
    # CONCURRENTLY-RUNNING different job's CSV, so the printed RMSD summary
    # and the results/<run>_rmsd_summary.txt file end up labelled with the
    # wrong RUN_NAME (jobs 612171/612172/612173/612174/612177 incident).
    EXPECTED_CSV="$REPO_DIR/logs/$RUN_NAME/$RUN_NAME/version_0/metrics.csv"
    if [ ! -f "$EXPECTED_CSV" ]; then
        echo "WARN: expected metrics.csv not found at: $EXPECTED_CSV"
        echo "      (training may have crashed before any val epoch — skipping plots)"
    else
        python -u plot_metrics.py --csv "$EXPECTED_CSV" \
            || echo "WARN: plot_metrics.py failed (non-fatal)"
        EXPECTED_RESULT="$REPO_DIR/results/${RUN_NAME}_rmsd_summary.txt"
        if [ ! -f "$EXPECTED_RESULT" ]; then
            echo "ERROR: post-training summary missing at expected path:"
            echo "       $EXPECTED_RESULT"
            echo "       plot_metrics.py wrote elsewhere — DO NOT trust other"
            echo "       result files in this directory until rerunning manually:"
            echo "       python -u plot_metrics.py --csv $EXPECTED_CSV"
        else
            echo "OK   : RESULT verified at $EXPECTED_RESULT"
        fi
    fi
fi

exit $EXIT_CODE
