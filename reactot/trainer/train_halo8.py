"""Training script for force-field potential on the Halo8 SQLite dataset.

Atom types in Halo8: H(1), C(6), N(7), O(8), F(9), S(16), Br(35)  → 7 classes

node_nf per atom  = pos(3) + one_hot(7) + charge(1) = 11
                  → node_nfs = [11]
"""

from typing import List, Optional, Tuple
from uuid import uuid4
import json
import logging
import os
import sys
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
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(results_dir: Path) -> logging.Logger:
    """Attach file handlers to the root logger for this run.

    Writes:
      * ``results_dir/train.log``  — INFO and above (full execution log)
      * ``results_dir/train.err``  — WARNING and above (errors / warnings)

    A StreamHandler to stdout is also added so the same messages appear on
    the console (and in the SLURM ``.out`` file).
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO+) — mirrors to SLURM stdout capture.
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # train.log — full INFO+ execution log.
    fh = logging.FileHandler(results_dir / "train.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # train.err — WARNING+ errors and exceptions only.
    eh = logging.FileHandler(results_dir / "train.err", mode="w", encoding="utf-8")
    eh.setLevel(logging.WARNING)
    eh.setFormatter(formatter)
    root.addHandler(eh)

    return logging.getLogger(__name__)


def _write_summary(results_dir: Path, run_name: str, config: dict,
                   metrics_cb: "MetricsHistoryCallback") -> None:
    """Write a human-readable ``summary.out`` with config and final metrics."""
    import datetime
    SEP = "=" * 70
    lines = [
        SEP,
        f"RUN SUMMARY  —  {run_name}",
        f"Completed    :  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
        "",
        "[Configuration]",
    ]
    for k, v in config.items():
        lines.append(f"  {k:<30s} = {v}")

    epochs = metrics_cb.epochs
    history = metrics_cb.history

    if epochs:
        lines += ["", f"[Final metrics  (epoch {epochs[-1]})]"]
        for k, vals in history.items():
            v = vals[-1] if vals else float("nan")
            lines.append(f"  {k:<30s} = {v:.6f}")

        lines += ["", "[Best metrics]"]
        for k, vals in history.items():
            finite = [v for v in vals if v == v]   # drop NaN
            if finite:
                best = min(finite)
                best_ep = epochs[vals.index(best)]
                lines.append(f"  {k:<30s} = {best:.6f}  (epoch {best_ep})")

    lines += ["", "[Output files]"]
    for f in sorted(results_dir.iterdir()):
        lines.append(f"  {f.name}")

    out_path = results_dir / "summary.out"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Summary written to %s", out_path)


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
# ---------------------------------------------------------------------------
# Runtime configuration — values are read from environment variables so the
# sbatch script can override them without patching this file.
#
#   DATA_LIMIT    int | "0"   rows per .db file after dand_id filtering
#                             (0 or unset → load all matching rows)
#   HALO8_DATADIR str         absolute path to the Halo8/ database folder
#   NUM_WORKERS   int         DataLoader workers per GPU process
#   RESUME_FROM   str         path to a .ckpt file to resume training from
# ---------------------------------------------------------------------------

def _resolve_datadir() -> str:
    """Return absolute path to the Halo8 dataset directory.

    Priority: HALO8_DATADIR env var → sibling of this file's package root.
    """
    env = os.environ.get("HALO8_DATADIR", "").strip()
    if env:
        return env
    # Derive from file location: reactot/trainer/ → reactot/dataset/Halo8
    return str(Path(__file__).resolve().parent.parent / "dataset" / "Halo8")


_data_limit_env = os.environ.get("DATA_LIMIT", "").strip()
_data_limit: int | None = (
    None if not _data_limit_env or _data_limit_env == "0"
    else int(_data_limit_env)
)

training_config = dict(
    # ---- dataset ----
    datadir=_resolve_datadir(),
    use_sqlite=True,                    # use HaloSQLiteDataset (not LmdbDataset)
    prefix=os.environ.get("DATASET_PREFIX", "Halogen"),  # "Halogen", "T1x", or "Mix"
    data_limit=_data_limit,             # None = load all dand_id-filtered rows
    # ---- loader ----
    bz=32,
    num_workers=int(os.environ.get("NUM_WORKERS", "4")),
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

# Create the per-run results directory early so logging can write there.
results_dir = Path("results") / run_name
results_dir.mkdir(parents=True, exist_ok=True)
log = _setup_logging(results_dir)
log.info("Run name    : %s", run_name)
log.info("Results dir : %s", results_dir.resolve())

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

# Checkpoint to resume from (RESUME_FROM env var, or None for a fresh run).
resume_from: str | None = os.environ.get("RESUME_FROM") or None
if resume_from:
    log.info("Resuming from checkpoint: %s", resume_from)

# WandB: respect WANDB_MODE env var (sbatch script sets it to 'offline' when
# no WANDB_API_KEY is available; runs can be synced later with `wandb sync`).
_wandb_offline = os.environ.get("WANDB_MODE", "online").lower() == "offline"
wandb_logger = WandbLogger(
    project=project,
    log_model=False,
    name=run_name,
    offline=_wandb_offline,
)
if _wandb_offline:
    log.info("WandB running in offline mode (sync later with `wandb sync`)")
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

log.info("Config:\n%s", json.dumps(config, indent=2, default=str))
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
)

log.info("Training started%s", f" (resuming from {resume_from})" if resume_from else "")
try:
    trainer.fit(potential, ckpt_path=resume_from)
except Exception:
    log.exception("Training failed with unhandled exception")
    raise
log.info("Training complete")

# ---------------------------------------------------------------------------
# Post-training: save metrics JSON + generate visualisation plots
# ---------------------------------------------------------------------------

# --- 1. Save raw metrics history as JSON ---
metrics_out = {"run_name": run_name, "epochs": metrics_history.epochs}
metrics_out.update(metrics_history.history)
json_path = results_dir / "metrics_history.json"
with open(json_path, "w") as _f:
    json.dump(metrics_out, _f, indent=2)
log.info("Metrics JSON saved to %s", json_path)

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
    log.info("Loss-curve plot saved to %s", loss_path)

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
    log.info("MAE-curve plot saved to %s", mae_path)

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
    log.info("Force-cosine plot saved to %s", cos_path)

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
    log.info("MAPE-curve plot saved to %s", mape_path)

    log.info("All result plots saved under: %s/", results_dir)

except ImportError:
    log.warning("matplotlib not found – skipping visualisation. Install with: pip install matplotlib")

_write_summary(results_dir, run_name, config, metrics_history)
