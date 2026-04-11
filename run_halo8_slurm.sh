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
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Number of samples to load from each .db file.
# Set to 0 or leave unset to load the full dataset.
DATA_LIMIT=${DATA_LIMIT:-100}

# Root of the repository (adjust if submitting from a different directory).
REPO_DIR=${REPO_DIR:-"$(pwd)"}

# Conda environment name / path.
CONDA_ENV=${CONDA_ENV:-"reactot"}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "DATA_LIMIT  : $DATA_LIMIT"
echo "REPO_DIR    : $REPO_DIR"
echo "CONDA_ENV   : $CONDA_ENV"
echo "=========================================="

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR" || exit 1

# Load modules (adjust to your cluster's module system).
module purge
module load cuda/11.8 cudnn/8.6 anaconda3

# Activate conda environment.
source activate "$CONDA_ENV"

# Verify GPU availability.
python -c "import torch; print('PyTorch', torch.__version__, '|',
    torch.cuda.device_count(), 'GPU(s) available')"

# ---------------------------------------------------------------------------
# Patch data_limit at runtime via environment variable
# ---------------------------------------------------------------------------
# train_halo8.py reads HALO8_DATA_LIMIT from env if set.
# The Python script has data_limit=100 as default; patch here if needed.
TRAIN_SCRIPT="$REPO_DIR/reactot/trainer/train_halo8.py"

if [ "$DATA_LIMIT" != "0" ] && [ -n "$DATA_LIMIT" ]; then
    echo "INFO: Patching data_limit to $DATA_LIMIT in training script."
    TMP_SCRIPT=$(mktemp /tmp/train_halo8_XXXX.py)
    sed "s/data_limit=[0-9]*/data_limit=$DATA_LIMIT/" "$TRAIN_SCRIPT" > "$TMP_SCRIPT"
    TRAIN_SCRIPT="$TMP_SCRIPT"
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
echo "Starting training at $(date)"

python -u "$TRAIN_SCRIPT"

EXIT_CODE=$?
echo "Training finished at $(date) with exit code $EXIT_CODE"

# Clean up temporary script if created.
if [ -n "$TMP_SCRIPT" ] && [ -f "$TMP_SCRIPT" ]; then
    rm -f "$TMP_SCRIPT"
fi

exit $EXIT_CODE
