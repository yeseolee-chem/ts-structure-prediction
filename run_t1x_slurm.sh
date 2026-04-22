#!/bin/bash
# ===========================================================================
# run_t1x_slurm.sh — T1x-only training wrapper
# ---------------------------------------------------------------------------
# Trains React-OT on the Halo8 LMDB using ONLY the T1x-prefixed reactions
# (Transition1x subset: C, H, N, O). This is the baseline run to compare
# against run_mix_slurm.sh (T1x + Halogen mix).
#
# Usage:
#   sbatch run_t1x_slurm.sh                            # DATA_LIMIT=300, default
#   DATA_LIMIT=1000 sbatch run_t1x_slurm.sh            # larger subset
#   DATA_LIMIT=0    sbatch run_t1x_slurm.sh            # full T1x split
#
# All other overrides from run_halo8_slurm.sh still apply (HALO8_SOURCE,
# HALO8_DATADIR, RESUME_FROM, USE_WANDB, CONDA_ENV, ...).
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
#SBATCH --time=48:00:00

export DATASET_PREFIX=T1x
export DATA_LIMIT=${DATA_LIMIT:-300}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_halo8_slurm.sh" "$@"
