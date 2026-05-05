from typing import List, Optional, Tuple
from uuid import uuid4
import argparse
import os
import shutil
import subprocess
import sys

import torch


def _git_branch_tag() -> str:
    """Branch identifier for run_name disambiguation.

    Prefers the EXPERIMENT_ID env var (set by the SLURM script before
    Python ever starts) over a live `git rev-parse`. The live read is
    UNRELIABLE when multiple parallel jobs share one git working
    directory and the user runs `git checkout` between submissions:
    every running job's `git rev-parse` reflects whatever branch is
    checked out RIGHT NOW, not the branch that was checked out when
    the job was queued. Two symptoms used to follow:

      1. Checkpoint dirs ended up double-tagged like
         `mix-Mix-dl500-cb-A-job610942-…-cb-B` because the slurm
         script captured "cb-A" but the trainer's later `git
         rev-parse` returned "cb-B".
      2. Even when the run-name looked right, the .py code being
         imported was whichever branch was checked out at trainer
         start, so "different" experiments produced bit-identical
         val_rmsd curves.

    When BOTH the env var and the live git branch are present and
    they DISAGREE we raise loudly. The mismatch means the working
    directory was switched between submission and execution; the
    Python module import path resolves to the LIVE branch's source,
    not the env's, so any results this run produces are invalid.
    Fix: give each parallel experiment its own checkout via
    `git worktree add ../<dir> <branch>` and submit each job from
    its own worktree.
    """
    env_tag = os.environ.get("EXPERIMENT_ID", "").strip()
    git_tag = ""
    try:
        git_tag = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .replace("/", "-")
        )
    except Exception:
        pass

    if env_tag and git_tag and env_tag != git_tag:
        raise RuntimeError(
            "Cross-branch contamination detected — refusing to train.\n"
            f"  EXPERIMENT_ID = {env_tag!r}  (set by SLURM at job submit)\n"
            f"  live git HEAD = {git_tag!r}  (working dir at trainer start)\n"
            f"The Python source being imported is from {git_tag!r}, NOT "
            f"{env_tag!r}, so this run would produce results for the wrong "
            "experiment.\n"
            "Fix: give each parallel job its own checkout, e.g.\n"
            f"    git worktree add ../ts-prediction-{env_tag} {env_tag}\n"
            f"    cd ../ts-prediction-{env_tag} && sbatch run_mix_slurm.sh"
        )
    return env_tag or git_tag or "unknown-branch"

from reactot.trainer.pl_trainer import SBModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet


class OPT:
    def __init__(self, solver, method):
        self.solver = solver
        self.method = method
        self.atol = 1e-2
        self.rtol = 1e-2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="React-OT / OT-FM training entry point")
    p.add_argument(
        "--dataset",
        choices=["TS1x", "Halo8"],
        default=os.environ.get("DATASET_PREFIX_KIND", "TS1x"),
        help="Dataset kind to train on.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Directory with training data. For TS1x: contains train_rpsb_all.pkl, "
        "valid_rpsb_all.pkl. For Halo8: contains Halo_*.db files.",
    )
    p.add_argument(
        "--data-limit",
        type=int,
        default=None,
        help="Halo8 only: cap the number of reaction groups sampled. 0 = all.",
    )
    p.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers."
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny CPU end-to-end run: 10 reactions, 1 train / 1 val batch, "
        "2 epochs, wandb disabled, no DDP.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force accelerator='cpu' regardless of CUDA availability.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip WandbLogger; log to CSV in logs/ instead.",
    )
    p.add_argument(
        "--halo-prefix",
        choices=["Halogen", "T1x", "Mix"],
        default=None,
        help="Halo8 only: dand_id prefix filter. 'Mix' accepts both Halogen "
        "and T1x. Falls back to the DATASET_PREFIX env var, then 'Halogen'.",
    )
    args = p.parse_args(argv)

    if args.data_dir is None:
        args.data_dir = os.environ.get("HALO8_DATADIR") if args.dataset == "Halo8" else None
    if args.data_limit is None and os.environ.get("DATA_LIMIT") is not None:
        args.data_limit = int(os.environ["DATA_LIMIT"])
    if args.num_workers is None and os.environ.get("NUM_WORKERS") is not None:
        args.num_workers = int(os.environ["NUM_WORKERS"])
    if args.halo_prefix is None:
        args.halo_prefix = os.environ.get("DATASET_PREFIX", "Halogen")

    return args


def build_configs(args):
    """Build the four configuration dicts plus misc module-level settings.

    Returns a dict of all the values main() needs so nothing leaks into module
    scope.
    """
    model_type = "leftnet"
    version = "ts_guess_NEBCI-xtb-ema"
    project = "RPSB-FT-Schedule"

    leftnet_config = dict(
        pos_require_grad=False,
        cutoff=10.0,
        num_layers=6,
        hidden_channels=196,
        num_radial=96,
        in_hidden_channels=8,
        reflect_equiv=True,
        legacy=True,
        update=True,
        pos_grad=False,
        single_layer_output=True,
        object_aware=True,
    )
    if model_type != "leftnet":
        raise KeyError("model type not implemented.")

    optimizer_config = dict(
        lr=1e-4,
        betas=[0.9, 0.999],
        weight_decay=0,
        amsgrad=True,
    )

    # Dataset-dependent defaults
    if args.dataset == "TS1x":
        default_datadir = "reactot/data/transition1x/"
        node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
        process_type = "TS1x"
    else:  # Halo8
        default_datadir = args.data_dir or "reactot/dataset/Halo8"
        # Halo8: 3 (pos) + 8 (one_hot: H/C/N/O/F/S/Cl/Br) + 1 (charge) = 12
        node_nfs: List[int] = [12] * 3
        process_type = "Halo8"

    datadir = args.data_dir or default_datadir
    if args.dataset == "Halo8" and datadir is None:
        raise ValueError(
            "Halo8 requires --data-dir or HALO8_DATADIR to be set."
        )

    training_config = dict(
        datadir=datadir,
        remove_h=False,
        bz=16,
        num_workers=args.num_workers if args.num_workers is not None else 0,
        clip_grad=True,
        gradient_clip_val=None,
        ema=True,
        ema_decay=0.999,
        swapping_react_prod=False,
        append_frag=False,
        use_by_ind=True,
        reflection=False,
        single_frag_only=False,
        only_ts=False,
        lr_schedule_type=None,
        lr_schedule_config=dict(gamma=0.8, step_size=10),
        use_sampler=True,
        sampler_config=dict(
            max_num=2800,
            mode="node^2",
            shuffle=True,
            ddp=False,
        ),
    )
    if args.dataset == "Halo8":
        training_config["data_limit"] = args.data_limit
        training_config["single_frag_only"] = False
        training_config["prefix"] = args.halo_prefix

    # Smoke-test overrides
    if args.smoke_test:
        args.cpu = True
        args.no_wandb = True
        training_config["bz"] = 2
        training_config["num_workers"] = 0
        training_config["use_sampler"] = False
        training_config["ema"] = False
        if args.dataset == "Halo8":
            training_config["data_limit"] = (
                args.data_limit if args.data_limit is not None else 10
            )

    return {
        "model_type": model_type,
        "version": version,
        "project": project,
        "leftnet_config": leftnet_config,
        "optimizer_config": optimizer_config,
        "training_config": training_config,
        "node_nfs": node_nfs,
        "process_type": process_type,
    }


def main(argv=None):
    args = parse_args(argv)

    if args.smoke_test or args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    cfgs = build_configs(args)
    training_config = cfgs["training_config"]

    edge_nf: int = 0
    condition_nf: int = 1
    fragment_names: List[str] = ["R", "TS", "P"]
    pos_dim: int = 3
    update_pocket_coords: bool = True
    condition_time: bool = True
    edge_cutoff: Optional[float] = None
    loss_type = "l2"
    pos_only = True
    enforce_same_encoding = None
    scales = [1.0, 2.0, 1.0]
    fixed_idx = [0, 2]
    eval_epochs = 1
    save_epochs = 1

    # Normalizer
    norm_values: Tuple = (1.0, 1.0, 1.0)
    norm_biases: Tuple = (0.0, 0.0, 0.0)

    # Schedule
    timesteps: int = 3000
    beta_max: float = 0.3
    power: float = 0.5
    inv_power: float = 1
    precision: float = 1e-5
    noise_schedule: str = "cosine"

    # SB
    mapping: str = "R+P->TS"
    mapping_initial: str = "RP"
    nfe: int = 25
    ot_ode: bool = True
    sigma: float = 0.0
    ts_guess = None

    # Use a stable RUN_NAME from env when provided so the checkpoint dir is
    # predictable across resubmissions. When a SLURM job hits its walltime
    # and is resubmitted, the new run lands in the same directory and can
    # resume from last.ckpt. Falls back to a uuid-suffixed name otherwise.
    #
    # Branch tag is appended to BOTH the env path and the uuid path so that
    # two branches running the same DATASET_PREFIX/DATA_LIMIT cannot collide
    # on a checkpoint directory (was the root cause of identical val_rmsd
    # across reactot-halo8/cb-D Mix and reactot-halo8 T1x/cb-BCD).
    branch_tag = _git_branch_tag()
    env_run_name = os.environ.get("RUN_NAME")
    if env_run_name:
        # Only append the branch tag if it isn't already present, so
        # callers (e.g. run_halo8_slurm.sh, which now includes it) don't
        # get a doubled suffix like "halo8-Mix-dl500-cb-D-cb-D".
        if branch_tag and branch_tag not in env_run_name:
            run_name = f"{env_run_name}-{branch_tag}"
        else:
            run_name = env_run_name
    else:
        run_name = (
            f"{cfgs['model_type']}-{cfgs['version']}-{branch_tag}-"
            + str(uuid4()).split("-")[-1]
        )

    opt = OPT(solver="ddpm", method="midpoint")

    seed_everything(42, workers=True)
    ddpm = SBModule(
        cfgs["leftnet_config"],
        cfgs["optimizer_config"],
        training_config,
        cfgs["node_nfs"],
        edge_nf,
        condition_nf,
        fragment_names,
        pos_dim,
        update_pocket_coords,
        condition_time,
        edge_cutoff,
        norm_values,
        norm_biases,
        noise_schedule,
        timesteps,
        precision,
        loss_type,
        pos_only,
        cfgs["process_type"],
        LEFTNet,
        enforce_same_encoding,
        scales,
        source=None,
        fixed_idx=fixed_idx,
        eval_epochs=eval_epochs,
        mapping=mapping,
        mapping_initial=mapping_initial,
        nfe=nfe,
        beta_max=beta_max,
        ot_ode=ot_ode,
        power=power,
        inv_power=inv_power,
        sigma=sigma,
        ts_guess=ts_guess,
        x0_method=os.environ.get("X0_METHOD", "ic"),
    )
    ddpm.ddpm.opt = opt

    config = cfgs["leftnet_config"].copy()
    config.update(cfgs["optimizer_config"])
    config.update(training_config)

    # Checkpoint directory is always keyed by the local run_name (uuid-based)
    # so it is predictable regardless of wandb mode (online / offline / disabled).
    ckpt_path = f"checkpoint/{cfgs['project']}/{run_name}"

    if args.no_wandb:
        csv_dir = os.path.join("logs", "smoke" if args.smoke_test else run_name)
        os.makedirs(csv_dir, exist_ok=True)
        logger = CSVLogger(save_dir=csv_dir, name=run_name)
    else:
        wandb_logger = WandbLogger(
            project=cfgs["project"], log_model=False, name=run_name
        )
        try:
            wandb_logger.experiment.config.update(config)
            wandb_logger.watch(
                ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False
            )
        except Exception:
            pass
        logger = wandb_logger

    # Pre-create checkpoint dir — PL 2.x ModelCheckpoint usually does this
    # lazily but has been observed to silently skip saves when the parent
    # directory doesn't yet exist at trainer init.
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint dir : {ckpt_path}")

    # Embed the branch tag in the checkpoint FILENAME too (not just the
    # directory). Belt-and-suspenders against the historical bug where
    # multiple branches resolved to the same RUN_NAME — even if dirs
    # somehow collide, filenames would still be unique.
    ckpt_filename = f"sb-{branch_tag}-{{epoch:03d}}-{{val_ep_scaled_err:.4f}}"

    # EarlyStopping configuration. min_delta defaults to 0 (strict
    # improvement), but cb-BC / cb-CD have noisier weighted-MSE loss
    # surfaces that can keep wiggling forever — they use a small positive
    # min_delta so EarlyStopping actually fires before max_epochs=-1 turns
    # into "train indefinitely". Tunable via EARLY_STOP_MIN_DELTA env var.
    early_stop_min_delta = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.0"))
    early_stop_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "150"))

    callbacks = [
        ModelCheckpoint(
            monitor="val_ep_scaled_err",
            dirpath=ckpt_path,
            filename=ckpt_filename,
            every_n_epochs=save_epochs,
            save_top_k=3,
            # Always keep a 'last.ckpt' as a safety net — even if the monitor
            # filename is skipped due to a missing metric on a given epoch,
            # last.ckpt lets us resume / evaluate the final weights.
            save_last=True,
            # Manual validation runs inside SBModule.on_train_epoch_end, so
            # val_ep_scaled_err is available in the train-epoch-end metrics
            # dict. Save there instead of on val-epoch-end (auto val is
            # disabled via check_val_every_n_epoch=1e9).
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor="val_ep_scaled_err",
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode="min",
            verbose=True,
            # val_ep_scaled_err is logged in on_train_epoch_end (manual val);
            # auto val is disabled via check_val_every_n_epoch=1e9.
            check_on_train_epoch_end=True,
        ),
        TQDMProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]

    if training_config["ema"]:
        callbacks.append(
            EMACallback(pl_module=ddpm, decay=training_config["ema_decay"])
        )

    # Device / strategy
    cuda_available = torch.cuda.is_available()
    if args.cpu or not cuda_available:
        accelerator = "cpu"
        devices = 1
        strategy = None
    else:
        accelerator = "gpu"
        if torch.cuda.device_count() > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
            devices = list(range(torch.cuda.device_count()))
        else:
            strategy = None
            devices = [0]

    trainer_kwargs = dict(
        # Unlimited epochs — training is bounded by EarlyStopping
        # (monitor=val_ep_scaled_err, patience=150).
        max_epochs=-1,
        accelerator=accelerator,
        deterministic=False,
        devices=devices,
        log_every_n_steps=20,
        callbacks=callbacks,
        profiler=None,
        logger=logger,
        accumulate_grad_batches=1,
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_train_batches=200,
        limit_val_batches=20,
        use_distributed_sampler=False,
        # Disable PL's auto end-of-epoch validation. The
        # (batch_idx+1) % val_check_batch == 0 schedule gets stuck at 42
        # when DynamicBatchSampler's __len__ overestimates the per-epoch
        # yield (Halo8 T1x runs ~10 batches/epoch), so auto-validation is
        # silently skipped. SBModule.on_train_epoch_end runs validation
        # manually every epoch instead. Sanity check still runs.
        check_val_every_n_epoch=10 ** 9,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    if args.smoke_test:
        trainer_kwargs.update(
            dict(
                max_epochs=2,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
        )

    print("config: ", config)
    trainer = Trainer(**trainer_kwargs)

    # Resume from a previous checkpoint when RESUME_FROM is set. PL restores
    # model weights, optimizer state, scheduler state, and current_epoch — so
    # training continues from where the previous (walltime-killed) job left off.
    resume_from = os.environ.get("RESUME_FROM", "").strip() or None
    if resume_from and not os.path.exists(resume_from):
        print(f"WARNING: RESUME_FROM={resume_from} not found — starting fresh")
        resume_from = None
    if resume_from:
        print(f"[reactot] Resuming from checkpoint: {resume_from}")
    trainer.fit(ddpm, ckpt_path=resume_from)
    return trainer


if __name__ == "__main__":
    main()
