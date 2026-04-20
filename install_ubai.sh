#!/bin/bash
# ===========================================================================
# install_ubai.sh — one-shot dependency install for UBAI cpu-gpu-cluster
# ---------------------------------------------------------------------------
# Usage (run ONCE on gate1 / gate2 with the reactot conda env activated):
#     conda activate reactot
#     cd ~/projects/ts-structure-prediction
#     bash install_ubai.sh
#
# Re-runnable: pip install is idempotent, so it's safe to run again if a
# later failure points to another missing package.
# ===========================================================================
set -e

# ===========================================================================
# Guard: must be run inside the 'reactot' conda env (Python 3.10)
# ===========================================================================
CURRENT_ENV="${CONDA_DEFAULT_ENV:-base}"
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [ "$CURRENT_ENV" != "reactot" ]; then
    echo "ERROR: Please activate the reactot conda env first:"
    echo "       conda activate reactot"
    echo "       bash install_ubai.sh"
    exit 1
fi

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "ERROR: Expected Python 3.10, got $PYTHON_VERSION"
    echo "       Make sure you're in the reactot env: conda activate reactot"
    exit 1
fi

echo "ENV     : $CURRENT_ENV (Python $PYTHON_VERSION) — OK"

echo "=========================================="
echo "Step 1/7  numpy<2 (torch 2.2.1 ABI)"
echo "=========================================="
pip install --upgrade pip
pip install "numpy<2.0"

echo "=========================================="
echo "Step 2/7  PyTorch 2.2.1 + CUDA 11.8"
echo "=========================================="
pip install torch==2.2.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

echo "=========================================="
echo "Step 3/7  PyTorch Geometric (matching cu118 wheels)"
echo "=========================================="
pip install torch_geometric==2.5.3
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

echo "=========================================="
echo "Step 4/7  PyTorch Lightning stack"
echo "=========================================="
pip install pytorch-lightning==2.4.0 torchmetrics

echo "=========================================="
echo "Step 5/7  Data / chemistry libraries"
echo "=========================================="
pip install pandas "ase>=3.23" pymatgen networkx lmdb tqdm

echo "=========================================="
echo "Step 6/7  Training tools (wandb, ODE, logging)"
echo "=========================================="
pip install wandb torchdiffeq timm rich ipdb colored-traceback

echo "=========================================="
echo "Step 7/7  Install this repo as editable package"
echo "=========================================="
pip install -e .

echo ""
echo "=========================================="
echo "  Installation complete. Running sanity checks..."
echo "=========================================="
python - <<'PY'
import importlib, sys
mods = [
    "torch", "numpy", "pandas", "pytorch_lightning", "torchmetrics",
    "torch_geometric", "torch_scatter", "torch_sparse", "torch_cluster",
    "torchdiffeq", "ase", "pymatgen", "networkx", "lmdb", "tqdm",
    "wandb", "timm", "rich", "ipdb", "reactot",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK  {m}")
    except Exception as e:
        missing.append((m, str(e)))
        print(f"  FAIL {m}: {e}")

import torch
print("")
print(f"  torch.__version__       = {torch.__version__}")
print(f"  torch.version.cuda      = {torch.version.cuda}")
print(f"  torch.cuda.is_available = {torch.cuda.is_available()}  (False on login node is normal)")
if missing:
    print("")
    print(f"  {len(missing)} module(s) failed to import — see above.")
    sys.exit(1)
PY

echo ""
echo "=========================================="
echo "  All dependencies installed and importable."
echo "  Next: sbatch run_halo8_slurm.sh"
echo "=========================================="
