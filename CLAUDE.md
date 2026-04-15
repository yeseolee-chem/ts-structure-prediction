# CLAUDE.md — React-OT (OT-FM) Project Context

## Project Overview

This repository implements **React-OT**, an optimal transport flow matching (OT-FM) model for generating transition state (TS) structures of chemical reactions. It is a fork of OA-ReactDiff (Duan et al., ICML 2023) that replaces stochastic diffusion (SDE) with deterministic optimal transport (ODE).

**Primary reference**: Duan et al., "Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model", *Nat. Mach. Intell.*, 2025, DOI: 10.1038/s42256-025-00988-1

**Branch**: `t1x-and-halogen` — supports both Transition1x (C, H, N, O) and Halo8 (C, H, N, O, F, S, Br) datasets.

**Cleanup status**: Legacy diffusion (DDPMModule, EnVariationalDiffusion), Confidence model, PotentialModule, QM9/Zeolite dataset references have been removed. Only OT-FM (SBModule + EnSB) remains.

## Architecture Summary

```
Input: R(reactant) xyz + P(product) xyz
    ↓
[pre_process.py] Parse xyz → atom types, positions, representation dicts
    ↓
[EGNNDynamics] Wrap model for multi-fragment message passing
    ↓
[LEFTNet] Object-aware equivariant scoring network
    │  - RBF embeddings (96 radial basis, cutoff=10.0 Å)
    │  - 6 message passing layers (hidden_channels=196)
    │  - Predicts velocity field v_θ(x_t, t)
    ↓
[en_sb.py / SBSchedule] OT path: x_t = (1-t)·x_0 + t·x_1
    │  - x_0 = (R + P) / 2 (IDPP-based initial guess)
    │  - x_1 = TS (ground truth)
    │  - Loss: ||v_θ(x_t, t) - (x_1 - x_0)||²
    ↓
[ODE Solver] At inference: integrate dx/dt = v_θ from t=0 to t=1
    │  - Solvers: "ode" (dopri5 via torchdiffeq), "ddpm", "ei"
    │  - Default NFE=10 (converges at this point)
    ↓
Output: Predicted TS geometry (xyz)
```

## Critical Files — DO NOT MODIFY without understanding

| File | Role | Notes |
|------|------|-------|
| `reactot/diffusion/en_sb.py` | OT-FM engine | `EnSchrodingerBridge` class. Contains `forward()` (training loss), `ode_sampling()` (inference), `compute_label()`, `compute_pred_x0()`. This is the CORE of OT-FM. |
| `reactot/model/leftnet.py` | Scoring network | Object-aware LEFTNet. `forward()` takes h, pos, edge_index → returns updated h, pos. The `scalarization()` method builds the local frame. |
| `reactot/dynamics/egnn_dynamics.py` | Model wrapper | Bridges LEFTNet with en_sb. Handles multi-fragment (R, TS, P) message passing via `n_frag_switch`. |
| `reactot/trainer/pl_trainer.py` | Training loop | `SBModule` (PyTorch Lightning). `training_step()`, `eval_sample_batch()`. Only SBModule remains after cleanup. |
| `reactot/dataset/transition1x.py` | T1x data | Loads pickle files from `reactot/data/transition1x/`. |
| `reactot/dataset/ff_lmdb.py` | LMDB data | `LmdbDataset` class. Supports Halo8 with `HALO_ATOM_MAPPING`. |

## Removed (legacy) — do NOT re-add

| Removed item | Was | Why removed |
|---|---|---|
| `en_diffusion.py` | SDE diffusion engine (OA-ReactDiff) | OT-FM uses en_sb.py only |
| `DDPMModule` (in pl_trainer.py) | Diffusion training module | SBModule is the OT-FM module |
| `ConfidenceModule` (in pl_trainer.py) | Confidence scoring | Not part of OT-FM |
| `potential_module.py` | Potential energy training | Different task than TS generation |
| `train_halo8.py` | PotentialModule training script | Not OT-FM |
| `confidence.py` (dynamics) | Confidence model | Not OT-FM |
| `potential.py` (dynamics) | Potential energy model | Not OT-FM |
| `qm9.py` (dataset) | QM9 dataset | Not T1x/Halogen |
| `zeolite.py` (dataset) | Zeolite dataset | Not organic reactions |
| `generate_confidence_sample.py` | Confidence evaluation | Not OT-FM |
| `sample_datasets.py` | QM9 sampling | Not T1x/Halogen |

## Key Training Parameters (from train_rpsb_ts1x.py)

```python
# LEFTNet config
cutoff = 10.0            # Å, interatomic interaction range
num_layers = 6           # message passing layers
hidden_channels = 196    # embedding dimension
num_radial = 96          # radial basis functions
in_hidden_channels = 8   # input encoding dimension
object_aware = True      # multi-fragment awareness

# Training config
batch_size = 14          # per GPU
optimizer = Adam(lr=1e-4, amsgrad=True)
ema_decay = 0.999
max_atoms_per_batch = 2800  # dynamic batching (for 16GB GPU, scale linearly)
solver = "ddpm"          # for training; "ode" for inference
```

## Running Experiments

### Environment Setup
```bash
conda env create -f env.yaml
conda activate reactot
pip install -e .
```

### Data
Download from [Zenodo](https://zenodo.org/records/13131875):
```bash
mkdir -p reactot/data/transition1x
mv *.pkl reactot/data/transition1x/
```

### Training (SLURM)
```bash
#!/bin/bash
#SBATCH --job-name=reactot-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.log

source activate reactot
python -m reactot.trainer.train_rpsb_ts1x
```

### Evaluation
```bash
python evaluation.py --checkpoint PATH --solver ode --nfe 10
```

### Single TS Prediction
```bash
python -m reactot.run_model \
    --rxyz reactant.xyz \
    --pxyz product.xyz \
    --checkpoint PATH \
    --nfe 10 \
    --output_path ./predictions/
```

## OT-FM Mathematical Core

The flow matching loss (Lipman et al., 2023; DOI: 10.48550/arXiv.2210.02747):

```
L_CFM(θ) = E_{t~U[0,1], q(x1), p0(x0)} [ ||v_θ(ψ_t(x0)) - u_t(ψ_t(x0)|x1)||² ]
```

With optimal transport path:
```
ψ_t(x) = t·x1 + (1 - (1-σ_min)·t)·x0     # linear interpolation
u_t(x|x1) = (x1 - (1-σ_min)·x) / (1 - (1-σ_min)·t)  # velocity field
```

In React-OT, `x0 = (R+P)/2` and `x1 = TS`.

## Atom Type Encoding

```python
# Transition1x: 5 types (C, H, N, O + charge)
node_nfs = [9] * 3  # 3 pos + 5 categorical + 1 charge, for R/TS/P

# Halo8 extension: 7 types
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 35: 6}
```

## Common Pitfalls

1. **OOM on GPU**: Reduce `max_num` in `sampler_config` (default 2800 for 16GB). Scale linearly with VRAM.
2. **RMSD too large warning**: If RMSD between R and P > 1 Å after alignment, check atom ordering. Hungarian algorithm in `pre_process.py` handles this.
3. **NFE sensitivity**: NFE < 10 degrades quality. NFE > 100 gives diminishing returns. Use NFE=10 for fast iteration, NFE=100 for final evaluation.
4. **solver choice**: Use `"ode"` for inference (deterministic, fast). `"ddpm"` in training config refers to the SB training schedule, NOT diffusion.
5. **swapping_react_prod**: Must be `False` for evaluation. Can be `True` during training for data augmentation (R↔P swap).

## Dependencies (from env.yaml)

- Python 3.10, PyTorch 2.2.1, PyTorch Lightning 2.4.0
- torch_geometric, torch-scatter, torch-sparse, torch-cluster
- torchdiffeq (ODE solver)
- pymatgen, ase (chemistry tools)
- wandb (experiment tracking)
- lmdb (dataset format)

## Project Structure (after cleanup)

```
ts-structure-prediction/
├── evaluation.py              # Evaluation entry point
├── env.yaml                   # Conda environment
├── pyproject.toml             # Build config
├── setup.cfg
├── README.md
├── LICENSE
├── CLAUDE.md                  # This file
└── reactot/
    ├── __init__.py
    ├── appmain.py             # CLI entry
    ├── pre_process.py         # xyz → representations
    ├── run_model.py           # TS prediction pipeline
    ├── model/
    │   ├── __init__.py
    │   ├── leftnet.py         # ★ Scoring network
    │   ├── block.py           # GCL, EquiMessage, EquiUpdate
    │   ├── core.py            # Model registry
    │   ├── egnn.py            # Base EGNN
    │   └── util_funcs.py      # RBF, MLP helpers
    ├── diffusion/
    │   ├── __init__.py
    │   ├── en_sb.py           # ★ OT-FM engine (Schrodinger Bridge)
    │   ├── _schedule.py       # SBSchedule (+ legacy noise schedules kept for safety)
    │   ├── _normalizer.py     # Feature normalization
    │   ├── _utils.py          # remove_mean_batch, etc.
    │   └── _node_dist.py      # Node distribution
    ├── dynamics/
    │   ├── __init__.py
    │   ├── _base.py           # Base dynamics
    │   └── egnn_dynamics.py   # ★ Model wrapper (LEFTNet ↔ en_sb bridge)
    ├── dataset/
    │   ├── __init__.py
    │   ├── base_dataset.py    # Base dataset class
    │   ├── datasets_config.py # Dataset configs
    │   ├── ff_lmdb.py         # LMDB loader (Halo8 support)
    │   ├── sampler.py         # Dynamic batch sampler
    │   └── transition1x.py    # T1x dataset
    ├── trainer/
    │   ├── pl_trainer.py      # ★ SBModule only (DDPMModule, ConfidenceModule removed)
    │   ├── train_rpsb_ts1x.py # Training script
    │   └── ema.py             # EMA callback
    ├── evaluate/
    │   └── ...                # Evaluation scripts (RMSD, energy diff analysis)
    ├── analyze/
    │   └── rmsd.py            # batch_rmsd_sb, batch_rmsd (used by pl_trainer)
    └── utils/
        ├── __init__.py
        ├── _graph_tools.py
        ├── bond_analyze.py
        ├── sampling_tools.py
        ├── training_tools.py
        └── xyz2mol.py
```

## Pipeline Context (pipeline_v4)

This OT-FM model is **Stage 2** of a larger explainable activation energy prediction pipeline:
- Stage 1: Energy prediction (AIMNet2)
- **Stage 2: TS structure generation (React-OT / OT-FM)** ← this repo
- Stage 5: Proxy-based EDA decomposition (Strain, Pauli, Elstat, Orbital, Dispersion)
- Stage 6: GPR with ARD kernel for E_a prediction

The weighted RMSD in Stage 5 shares the same continuous weight function as the FM loss weighting in Stage 2, ensuring pipeline consistency.
