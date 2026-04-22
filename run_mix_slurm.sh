#!/bin/bash
# ===========================================================================
# run_mix_slurm.sh — T1x + Halogen mix training wrapper
# ---------------------------------------------------------------------------
# Trains React-OT on the Halo8 LMDB using BOTH T1x-prefixed and Halogen-
# prefixed reactions (C, H, N, O, F, S, Cl, Br). Paired with run_t1x_slurm.sh
# this gives the dataset-vs-dataset comparison requested in the paper
# follow-up: baseline (T1x only) vs augmented (T1x + halogens).
#
# Usage:
#   sbatch run_mix_slurm.sh                            # DATA_LIMIT=300, default
#   DATA_LIMIT=1000 sbatch run_mix_slurm.sh            # larger subset
#   DATA_LIMIT=0    sbatch run_mix_slurm.sh            # full mixed split
#
# All other overrides from run_halo8_slurm.sh still apply (HALO8_SOURCE,
# HALO8_DATADIR, RESUME_FROM, USE_WANDB, CONDA_ENV, ...).
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

export DATASET_PREFIX=Mix
export DATA_LIMIT=${DATA_LIMIT:-300}

# SLURM copies submitted scripts to /var/spool/slurmd/jobNNN/ and runs them
# from there, so BASH_SOURCE[0] doesn't point at the repo. Prefer
# $SLURM_SUBMIT_DIR (the directory where `sbatch` was invoked) and fall back
# to BASH_SOURCE for local execution outside SLURM.
if [ -n "${REPO_DIR:-}" ]; then
    SCRIPT_DIR="$REPO_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export REPO_DIR="$SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/run_halo8_slurm.sh" ]; then
    echo "ERROR: run_halo8_slurm.sh not found under $SCRIPT_DIR" >&2
    echo "       Submit from the repo root, or set REPO_DIR=<repo> sbatch run_mix_slurm.sh" >&2
    exit 1
fi

exec bash "$SCRIPT_DIR/run_halo8_slurm.sh" "$@"
