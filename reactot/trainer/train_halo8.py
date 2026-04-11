"""Training script for force-field potential on the Halo8 SQLite dataset.

Atom types in Halo8: H(1), C(6), N(7), O(8), F(9), S(16), Br(35)  → 7 classes

node_nf per atom  = pos(3) + one_hot(7) + charge(1) = 11
                  → node_nfs = [11]
"""

from typing import List, Optional, Tuple
from uuid import uuid4
import torch

from reactot.trainer.potential_module import PotentialModule
from reactot.dataset.ff_lmdb import N_HALO_ATOM_TYPES
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet


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
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]

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
