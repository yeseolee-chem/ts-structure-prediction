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
# Set to "true" to load only 1/10th of each .db file (for quick experiments).
DATA_LIMIT=${DATA_LIMIT:-"false"}

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
# Patch data_limit flag at runtime if requested
# ---------------------------------------------------------------------------
TRAIN_SCRIPT="$REPO_DIR/reactot/trainer/train_halo8.py"

if [ "$DATA_LIMIT" = "true" ]; then
    echo "INFO: Enabling data_limit=True (1/10 of each file will be loaded)."
    # Create a temporary training script with data_limit patched.
    TMP_SCRIPT=$(mktemp /tmp/train_halo8_XXXX.py)
    sed 's/data_limit=False/data_limit=True/' "$TRAIN_SCRIPT" > "$TMP_SCRIPT"
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
