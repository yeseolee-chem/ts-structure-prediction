"""Training script for force-field potential on the Halo8 SQLite dataset.

Atom types in Halo8: H(1), C(6), N(7), O(8), F(9), S(16), Br(35)  → 7 classes

node_nf per atom  = pos(3) + one_hot(7) + charge(1) = 11
                  → node_nfs = [11]
"""

from typing import List, Optional, Tuple
from uuid import uuid4
import json
from pathlib import Path

import torch
import numpy as np

from reactot.trainer.potential_module import PotentialModule
from reactot.dataset.ff_lmdb import N_HALO_ATOM_TYPES
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet


# ---------------------------------------------------------------------------
# Metrics history callback – collects per-epoch scalars for post-run plots
# ---------------------------------------------------------------------------
class MetricsHistoryCallback(Callback):
    """Accumulates train/val metrics each epoch for offline plotting."""

    TRACKED = [
        "train-totloss",
        "val-totloss",
        "val-MAE_E",
        "val-MAE_F",
        "val-MAPE_E",
        "val-MAPE_F",
        "val-MAE_Fcos",
        "val-Loss_E",
        "val-Loss_F",
    ]

    def __init__(self):
        super().__init__()
        self.history: dict = {k: [] for k in self.TRACKED}
        self.epochs: list = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        self.epochs.append(pl_module.current_epoch)
        for key in self.TRACKED:
            val = metrics.get(key)
            self.history[key].append(float(val) if val is not None else float("nan"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_type = "leftnet"
version = "halo8-ff"
project = "Halo8-FF"

leftnet_config = dict(
    pos_require_grad=True,   # needed for force prediction
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=True,
    single_layer_output=True,
    object_aware=False,
)
model = LEFTNet

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
optimizer_config = dict(
    lr=1e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_config = dict(
    # ---- dataset ----
    datadir="reactot/dataset/Halo8",   # folder that contains Halo_*.db files
    use_sqlite=True,                    # use HaloSQLiteDataset instead of LmdbDataset
    data_limit=100,                      # load first 100 samples from each .db file
    # ---- loader ----
    bz=32,
    num_workers=4,
    # ---- training ----
    clip_grad=True,
    gradient_clip_val=None,
    ema=True,
    ema_decay=0.999,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=10,
    ),
)

# ---------------------------------------------------------------------------
# Architecture dimensions
# ---------------------------------------------------------------------------
# pos(3) + one_hot(N_HALO_ATOM_TYPES) + charge(1) = 3 + 7 + 1 = 11
POS_DIM = 3
node_nfs: List[int] = [POS_DIM + N_HALO_ATOM_TYPES + 1]
edge_nf: int = 0
condition_nf: int = 1
fragment_names: List[str] = ["struct"]
pos_dim: int = POS_DIM
condition_time: bool = True
timesteps: int = 5000
use_autograd: bool = False

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
eval_epochs = 1
save_epochs = 1

# Ensure output directories exist before any I/O.
for _dir in ("logs", "checkpoint", "results"):
    Path(_dir).mkdir(parents=True, exist_ok=True)

seed_everything(42, workers=True)

run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

potential = PotentialModule(
    model_config=leftnet_config,
    optimizer_config=optimizer_config,
    training_config=training_config,
    node_nfs=node_nfs,
    edge_nf=edge_nf,
    condition_nf=condition_nf,
    fragment_names=fragment_names,
    pos_dim=pos_dim,
    edge_cutoff=None,
    model=model,
    use_autograd=use_autograd,
    timesteps=timesteps,
    condition_time=condition_time,
)

config = leftnet_config.copy()
config.update(optimizer_config)
config.update(training_config)

wandb_logger = WandbLogger(
    project=project,
    log_model=False,
    name=run_name,
)
try:
    wandb_logger.experiment.config.update(config)
except Exception:
    pass

ckpt_path = f"checkpoint/{project}/{run_name}"
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ff-{epoch:03d}-{val-totloss:.4f}",
    every_n_epochs=save_epochs,
    save_top_k=5,
)
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=500,
    verbose=True,
    log_rank_zero_only=True,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
metrics_history = MetricsHistoryCallback()
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor, metrics_history]

if training_config["ema"]:
    callbacks.append(
        EMACallback(pl_module=potential, decay=training_config["ema_decay"])
    )

strategy = None
devices = list(range(torch.cuda.device_count())) or [0]
if len(devices) > 1:
    strategy = DDPStrategy(find_unused_parameters=True)

print("Config:", config)
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=20,
    callbacks=callbacks,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    replace_sampler_ddp=False,
)

trainer.fit(potential)

# ---------------------------------------------------------------------------
# Post-training: save metrics JSON + generate visualisation plots
# ---------------------------------------------------------------------------
results_dir = Path("results") / run_name
results_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Save raw metrics history as JSON ---
metrics_out = {"run_name": run_name, "epochs": metrics_history.epochs}
metrics_out.update(metrics_history.history)
json_path = results_dir / "metrics_history.json"
with open(json_path, "w") as _f:
    json.dump(metrics_out, _f, indent=2)
print(f"Metrics JSON saved to {json_path}")

# --- 2. Generate plots (gracefully skip if matplotlib not available) ---
try:
    import matplotlib
    matplotlib.use("Agg")          # headless backend (safe on cluster)
    import matplotlib.pyplot as plt

    epochs = metrics_history.epochs
    history = metrics_history.history

    def _plot(ax, key, label, color):
        vals = history.get(key, [])
        valid = [(e, v) for e, v in zip(epochs, vals) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, label=label, color=color)

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot(ax, "train-totloss", "Train loss", "steelblue")
    _plot(ax, "val-totloss",   "Val loss",   "tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title(f"Training & validation loss\n{run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = results_dir / "loss_curves.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Loss-curve plot saved to {loss_path}")

    # --- MAE: energy & force ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot(axes[0], "val-MAE_E", "Val MAE(E)", "steelblue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE (energy)")
    axes[0].set_title("Validation MAE – Energy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    _plot(axes[1], "val-MAE_F", "Val MAE(F)", "darkorange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE (force)")
    axes[1].set_title("Validation MAE – Force")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(run_name)
    fig.tight_layout()
    mae_path = results_dir / "mae_curves.png"
    fig.savefig(mae_path, dpi=150)
    plt.close(fig)
    print(f"MAE-curve plot saved to {mae_path}")

    # --- Force cosine similarity (direction error) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot(ax, "val-MAE_Fcos", "Force cosine error", "purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 – cos(F_pred, F_true)")
    ax.set_title(f"Force direction error\n{run_name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cos_path = results_dir / "force_cosine_error.png"
    fig.savefig(cos_path, dpi=150)
    plt.close(fig)
    print(f"Force-cosine plot saved to {cos_path}")

    # --- MAPE: energy & force ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot(axes[0], "val-MAPE_E", "Val MAPE(E)", "steelblue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAPE (energy)")
    axes[0].set_title("Validation MAPE – Energy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    _plot(axes[1], "val-MAPE_F", "Val MAPE(F)", "darkorange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAPE (force)")
    axes[1].set_title("Validation MAPE – Force")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(run_name)
    fig.tight_layout()
    mape_path = results_dir / "mape_curves.png"
    fig.savefig(mape_path, dpi=150)
    plt.close(fig)
    print(f"MAPE-curve plot saved to {mape_path}")

    print(f"\nAll result plots saved under: {results_dir}/")

except ImportError:
    print("matplotlib not found – skipping visualisation. Install with: pip install matplotlib")
