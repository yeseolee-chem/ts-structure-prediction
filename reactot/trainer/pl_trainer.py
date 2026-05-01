from typing import Dict, List, Optional, Tuple

from pathlib import Path
import os
import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from pytorch_lightning import LightningModule
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, MeanAbsoluteError

from reactot.dataset import ProcessedTS1x, ProcessedHalo8, DynamicBatchSampler
from reactot.dynamics import EGNNDynamics
from reactot.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule, SBSchedule
from reactot.diffusion._normalizer import Normalizer, FEATURE_MAPPING
from reactot.diffusion.en_sb import EnSB
from reactot.trainer._metrics import average_over_batch_metrics, pretty_print
import reactot.utils.training_tools as utils
from reactot.analyze.rmsd import batch_rmsd_sb, batch_rmsd
from reactot.utils.sampling_tools import write_tmp_xyz

from tqdm import tqdm

PROCESS_FUNC = {
    "TS1x": ProcessedTS1x,
    "Halo8": ProcessedHalo8,
}
FILE_TYPE = {
    "TS1x": ".pkl",
    "Halo8": "",
}
LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}

class SBModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
        node_nfs: List[int] = [9] * 3,
        edge_nf: int = 4,
        condition_nf: int = 3,
        fragment_names: List[str] = ["inorg_node", "org_edge", "org_node"],
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        norm_values: Tuple = (1.0, 1.0, 1.0),
        norm_biases: Tuple = (0.0, 0.0, 0.0),
        noise_schedule: str = "polynomial_2",
        timesteps: int = 1000,
        precision: float = 1e-5,
        loss_type: str = "l2",
        pos_only: bool = False,
        process_type: Optional[str] = None,
        model: nn.Module = None,
        enforce_same_encoding: Optional[List] = None,
        scales: List[float] = [1., 1., 1.],
        eval_epochs: int = 20,
        source: Optional[Dict] = None,
        fixed_idx: Optional[List] = None,
        mapping: str = "R+P->TS",
        mapping_initial: str = "RP",
        beta_max: float = 0.3,
        nfe: int = 100,
        ot_ode: bool = True,
        power: float = 1,
        inv_power: float = 1,
        sigma: float = 0.0,
        ts_guess: bool = False,
        idx: int = 1,
        pbc: bool = False,
        learn_importance: bool = False,
        kl_weight: float = 0.1,
        learned_w_min: float = 0.1,
    ) -> None:
        super().__init__()
        egnn_dynamics = EGNNDynamics(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
            model=model,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
            learn_importance=learn_importance,
        )

        normalizer = Normalizer(
            norm_values=norm_values,
            norm_biases=norm_biases,
            pos_dim=pos_dim,
        )

        schedule = SBSchedule(
            timesteps=timesteps,
            beta_max=beta_max,
            power=power,
            inv_power=inv_power,
        )

        self.ddpm = EnSB(
            dynamics=egnn_dynamics,
            schdule=schedule,
            normalizer=normalizer,
            size_histogram=None,
            loss_type=loss_type,
            pos_only=pos_only,
            fixed_idx=fixed_idx,
            mapping=mapping,
            mapping_initial=mapping_initial,
            sigma=sigma,
            ts_guess=ts_guess,
            idx=idx,
            kl_weight=kl_weight,
            learned_w_min=learned_w_min,
        )
        self.learn_importance = learn_importance
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.loss_type = loss_type
        self.n_fragments = len(fragment_names)
        self.remove_h = training_config["remove_h"]
        self.pos_only = pos_only
        self.process_type = process_type or "QM9"
        self.scales = scales
        self.eval_epochs = eval_epochs
        self.nfe = nfe
        self.ot_ode = ot_ode
        self.ts_guess = ts_guess
        self.eval_keys = ["rmsd_mean", "rmsd_median", "rmsd_std", "ep_loss", "ep_scaled_err"]

        if not "use_sampler" in self.training_config:
            self.training_config["use_sampler"] = False

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            self.gradnorm_queue.add(3000)
        self._train_step_outputs: List = []
        self._val_step_outputs: List = []
        self._test_step_outputs: List = []
        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.ddpm.parameters(),
            **self.optimizer_config
        )
        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer,
                **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def setup(
        self,
        stage: Optional[str] = None,
        device: str = None,
        swapping_react_prod: Optional[bool] = None,

    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        func = PROCESS_FUNC[self.process_type]
        ft = FILE_TYPE[self.process_type]
        if swapping_react_prod is not None:
            self.training_config.update(
                {"swapping_react_prod": swapping_react_prod}
            )
        self.training_config.update({"ts_guess": self.ts_guess})

        def _data_path(split: str) -> Path:
            """Return the path handed to the dataset constructor.

            TS1x expects a concrete .pkl file (e.g. ``train_rpsb_all.pkl``).
            Halo8 passes the raw directory and sub-samples via ``data_limit``.
            """
            if self.process_type == "Halo8":
                return Path(self.training_config["datadir"])
            return Path(
                self.training_config["datadir"],
                f"{split}_rpsb_all{ft}",
            )

        halo_base_seed = int(self.training_config.get("halo_seed", 42))
        # Halo8 deterministic-split kwargs. The split assignment is hashed
        # on dand_id (see ff_lmdb._assign_split), so train/val/test never
        # overlap regardless of seed/data_limit. The OLD setup just passed
        # different sampling seeds against the same pool, which let
        # randomly-overlapping subsets land in train and val.
        halo_split_kwargs = (
            dict(
                val_fraction=float(self.training_config.get("halo_val_fraction", 0.1)),
                test_fraction=float(self.training_config.get("halo_test_fraction", 0.1)),
                split_seed=int(self.training_config.get("halo_split_seed", 42)),
            )
            if self.process_type == "Halo8" else {}
        )

        if stage == "fit":
            extra_train = (
                {"seed": halo_base_seed, "split": "train", **halo_split_kwargs}
                if self.process_type == "Halo8" else {}
            )
            self.train_dataset = func(
                _data_path("train"),
                # device=device,
                **self.training_config,
                **extra_train,
            )
            self.training_config["reflection"] = False  # Turn off reflection in val.
            extra_val = (
                {"seed": halo_base_seed + 1, "split": "val", **halo_split_kwargs}
                if self.process_type == "Halo8" else {}
            )
            self.val_dataset = func(
                _data_path("valid"),
                # device=device,
                **self.training_config,
                **extra_val,
            )

            # self.training_config["swapping_react_prod"] = False  # uncomment if one does not want swapping in full validation
            val_dataset_no_swap = func(
                _data_path("valid"),
                device=device,
                **self.training_config,
                **extra_val,
            )
            if self.training_config["use_sampler"]:
                _config = self.training_config["sampler_config"].copy()
                _config["max_num"] = int(_config["max_num"] * 3)
                # Pass an explicit val seed so the val loader is reproducible
                # WITHOUT colliding with the train sampler (used to be a
                # hard-coded 42 for both — see sampler.py). This keeps val
                # batch order stable across epochs/runs while letting train
                # use a different, freshly-derived generator.
                _config.setdefault("seed", halo_base_seed + 11)
                sampler = DynamicBatchSampler(
                    dataset=val_dataset_no_swap,
                    max_batch=100,  # This is hard coded.
                    **_config,
                )
                self.val_loader_no_swap = DataLoader(
                    val_dataset_no_swap,
                    batch_sampler=sampler,
                    num_workers=self.training_config["num_workers"],
                    collate_fn=val_dataset_no_swap.collate_fn,
                )
            else:
                self.val_loader_no_swap = DataLoader(
                    val_dataset_no_swap,
                    4 * self.training_config["bz"],
                    shuffle=False,
                    num_workers=self.training_config["num_workers"],
                    collate_fn=val_dataset_no_swap.collate_fn,
                )
        elif stage == "test":
            if self.process_type == "Halo8":
                test_path = Path(self.training_config["datadir"])
            else:
                test_path = Path(self.training_config["datadir"], f"test{ft}")
            extra_test = (
                {"seed": halo_base_seed + 2, "split": "test", **halo_split_kwargs}
                if self.process_type == "Halo8" else {}
            )
            self.test_dataset = func(
                test_path,
                # device=device,
                **self.training_config,
                **extra_test,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, bz: Optional[int] = None) -> DataLoader:
        halo_base_seed = int(self.training_config.get("halo_seed", 42))
        if self.training_config["use_sampler"]:
            _config = self.training_config["sampler_config"].copy()
            # Train sampler seed = halo_base_seed. Decoupled from val seed
            # below so train/val see independent permutations.
            _config.setdefault("seed", halo_base_seed)
            sampler = DynamicBatchSampler(
                dataset=self.train_dataset,
                **_config,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.training_config["num_workers"],
                collate_fn=self.train_dataset.collate_fn,
            )
        bz = bz or self.training_config["bz"]
        return DataLoader(
            self.train_dataset,
            bz,
            shuffle=True,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self, bz: Optional[int] = None, shuffle: bool = True,) -> DataLoader:
        halo_base_seed = int(self.training_config.get("halo_seed", 42))
        if self.training_config["use_sampler"]:
            _config = self.training_config["sampler_config"].copy()
            _config["max_num"] = int(_config["max_num"] * 3)
            # Val sampler seed = halo_base_seed + 11. Different offset from
            # train so the val batch order is independent and reproducible.
            _config.setdefault("seed", halo_base_seed + 11)
            sampler = DynamicBatchSampler(
                dataset=self.val_dataset,
                **_config,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=sampler,
                num_workers=self.training_config["num_workers"],
                collate_fn=self.val_dataset.collate_fn,
            )
        bz = bz or 4 * self.training_config["bz"]
        return DataLoader(
            self.val_dataset,
            bz,
            shuffle=shuffle,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self, bz: Optional[int] = None) -> DataLoader:
        bz = bz or self.training_config["bz"]
        return DataLoader(
            self.test_dataset,
            bz,
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.test_dataset.collate_fn,
        )

    def compute_loss(self, batch):
        representations, conditions = batch
        # Idea 1-CD: pull the precomputed atom_weights (hierarchical C-prior
        # in CD mode, flat exp-decay in A mode) off the target fragment. When
        # the dataset was built without weighting (zero_charge=True or
        # graph_weights_enabled=False) this is None and en_sb falls back to
        # uniform F.mse_loss.
        target_rep = representations[self.ddpm.idx]
        atom_weights_prior = target_rep.get("atom_weights", None)
        loss_terms = self.ddpm.forward(
            representations,
            conditions,
            ot_ode=self.ot_ode,
            atom_weights_prior=atom_weights_prior,
        )
        info = {
            "loss": loss_terms["loss"],
            "scaled_err": loss_terms["scaled_err"],
        }
        if "loss_fm" in loss_terms:
            info["loss_fm"] = loss_terms["loss_fm"]
        if "loss_kl" in loss_terms:
            info["loss_kl"] = loss_terms["loss_kl"]
        return info

    @torch.no_grad()
    def eval_sample_batch(
        self,
        batch: List,
        return_rmsd: bool = False,
        write_xyz: bool = False,
        batch_idx: int = 0,
        bz: int = 32,
        localpath: str = "sb/ot_ode-10/",
        refpath: str = "ref_ts/",
        return_all: bool = False,
    ):

        self.ddpm.eval()

        representations, conditions = batch
        x0, x1, cond, x0_size, x0_other = self.ddpm.sample_batch(
            representations, conditions, return_timesteps=False, training=False)

        with torch.no_grad():
            xs, pred_x0 = self.ddpm.sample(
                x1, representations, conditions, nfe=self.nfe, ot_ode=self.ot_ode)
            info = self.compute_loss(batch)
        x0_pred = xs[:, 0, ...]

        target_xh = torch.cat([x0.cpu(),      x0_other.cpu()], dim=1)
        pred_xh   = torch.cat([x0_pred.cpu(), x0_other.cpu()], dim=1)
        rmsds = batch_rmsd_sb(x0_size.cpu(), pred_xh, target_xh, same_order=True)

        if return_all:
            return cond["r_pos"], x0_pred.cpu(), cond["p_pos"], x0_size, x0_other, rmsds

        if write_xyz:
            if not os.path.isdir(localpath):
                os.makedirs(localpath)

            out_samples = [
                [], pred_xh, []
            ]
            fragments_nodes = [x0_size, [], []]
            
            os.makedirs(localpath, exist_ok=True,)
            write_tmp_xyz(
                fragments_nodes=fragments_nodes,
                out_samples=out_samples,
                idx=[1],
                prefix="gen",
                localpath=localpath,
                ex_ind=batch_idx * bz,
            )
            out_samples = [
                [], target_xh, []
            ]
            os.makedirs(refpath, exist_ok=True,)
            write_tmp_xyz(
                fragments_nodes=fragments_nodes,
                out_samples=out_samples,
                idx=[1],
                prefix="gen",
                localpath=refpath,
                ex_ind=batch_idx * bz,
            )

        self.ddpm.train()

        res = {
            "rmsd_mean": np.mean(rmsds),
            "rmsd_median": np.median(rmsds),
            "rmsd_std": np.std(rmsds),
            "ep_loss": info["loss"].item(),
            "ep_scaled_err": info["scaled_err"].item(),
        }
        if return_rmsd:
            return res, rmsds
        return res

    def training_step(self, batch, batch_idx):
        info = self.compute_loss(batch)
        for k, v in info.items():
            self.log(f"tr_{k}", v.item(), rank_zero_only=True)

        if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
            if self.trainer.is_global_zero:
                print("evaluation on samping for training batch...", batch[0][0]["size"].shape, batch_idx)
            res = self.eval_sample_batch(batch)
            info.update(res)
        else:
            for k in self.eval_keys:
                info[k] = np.nan
        self._train_step_outputs.append(info)
        return info

    @torch.no_grad()
    def _shared_eval(self, batch, batch_idx, prefix, *args):
        info = self.compute_loss(batch)

        ip = {}
        for k, v in info.items():
            ip[f"{prefix}_{k}"] = v.item()

        # Compute RMSD on EVERY val/test batch (not just batch_idx == 0).
        #
        # Was: `if (epoch+1) % eval_epochs == 0 and batch_idx == 0` — which
        # meant val_rmsd_* aggregated by `average_over_batch_metrics` only
        # ever saw batch 0 (because batches 1..N had NaN values that the
        # aggregator skips and the value at ii==0 is taken verbatim — see
        # reactot/trainer/_metrics.py). With limit_val_batches=20 and DDP-
        # style dynamic batching that's ~3-7 reactions reported as the
        # entire val set's RMSD — statistically meaningless and identical
        # across runs that share a val split (because the sampler used to be
        # hard-seeded). Now we spend the extra forward passes to get a real
        # number.
        if (self.current_epoch + 1) % self.eval_epochs == 0:
            if self.trainer.is_global_zero and batch_idx == 0:
                print(
                    "evaluation on samping for validation batch...",
                    batch[0][0]["size"].shape, batch_idx,
                )
            res = self.eval_sample_batch(batch)
            for k, v in res.items():
                ip[f"{prefix}_{k}"] = v
        else:
            for k in self.eval_keys:
                ip[f"{prefix}_{k}"] = np.nan
        return ip

    def validation_step(self, batch, batch_idx, *args):
        info = self._shared_eval(batch, batch_idx, "val", *args)
        self._val_step_outputs.append(info)
        return info

    def test_step(self, batch, batch_idx, *args):
        info = self._shared_eval(batch, batch_idx, "test", *args)
        self._test_step_outputs.append(info)
        return info

    def on_validation_epoch_end(self) -> None:
        val_step_outputs = self._val_step_outputs
        val_epoch_metrics = average_over_batch_metrics(val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

        if self.current_epoch % 10 == 0 and self.current_epoch > 1:  # this is hard coded.
            _, rmsds = self.eval_rmsd(
                self.val_loader_no_swap,
                write_xyz=False,
            )
            rmsds_mean, rmsds_median, rmsds_std = np.mean(rmsds), np.median(rmsds), np.std(rmsds)
            rmsds_len = len(rmsds)
            self.log("val_ep_rmsd_length", rmsds_len, sync_dist=True)
            self.log("val_ep_rmsd_mean", float(rmsds_mean), sync_dist=True)
            self.log("val_ep_rmsd_median", float(rmsds_median), sync_dist=True)
            self.log("val_ep_rmsd_std", float(rmsds_std), sync_dist=True)

        self._val_step_outputs.clear()

    def on_train_epoch_end(self) -> None:
        outputs = self._train_step_outputs
        epoch_metrics = average_over_batch_metrics(outputs, allowed=self.eval_keys)
        for k, v in epoch_metrics.items():
            self.log(f"tr_{k}", v, sync_dist=True)
        self._train_step_outputs.clear()

        # PL's auto-validation schedules via (batch_idx+1) % val_check_batch == 0,
        # which silently skips validation whenever DynamicBatchSampler's __len__
        # overestimates the per-epoch yield (Halo8 T1x: __len__ claims 42, iter
        # yields ~10 because molecules are small). Run validation ourselves so
        # val_* metrics are always logged — EarlyStopping/ModelCheckpoint then
        # read a populated monitor every epoch.
        if not getattr(self.trainer, "sanity_checking", False):
            self._run_manual_validation()

    @torch.no_grad()
    def _run_manual_validation(self) -> None:
        """Run one validation pass and log aggregated val_* metrics.

        Lives in the train-epoch-end hook rather than relying on PL's
        batch-index-based val scheduler, which breaks when a custom
        batch-sampler's __len__ disagrees with its actual iteration count.
        Swaps EMA weights in/out via the existing EMACallback so this mirrors
        what PL would have done through on_validation_epoch_start/end.
        """
        from reactot.trainer.ema import EMACallback

        trainer = self.trainer
        ema_cb = None
        if trainer is not None:
            for cb in getattr(trainer, "callbacks", []) or []:
                if isinstance(cb, EMACallback):
                    ema_cb = cb
                    break

        if ema_cb is not None:
            ema_cb.store(self.parameters())
            ema_cb.copy_to(ema_cb.ema.module.parameters(), self.parameters())

        try:
            loader = self.val_dataloader()
            limit = getattr(trainer, "limit_val_batches", None) if trainer is not None else None
            if not isinstance(limit, int):
                limit = None  # float fraction or None → iterate all batches

            self.ddpm.eval()
            try:
                for batch_idx, batch in enumerate(loader):
                    if limit is not None and batch_idx >= limit:
                        break
                    batch = self.transfer_batch_to_device(
                        batch, self.device, dataloader_idx=0
                    )
                    info = self._shared_eval(batch, batch_idx, "val")
                    self._val_step_outputs.append(info)
            finally:
                self.ddpm.train()

            val_epoch_metrics = average_over_batch_metrics(self._val_step_outputs)
            if trainer is not None and trainer.is_global_zero:
                pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
            # _shared_eval prefixes RMSD keys as "val_rmsd_*" but plot_metrics.py
            # and the EarlyStopping/ModelCheckpoint monitors expect "val_ep_rmsd_*".
            # Rename here so the CSV columns match what every downstream consumer
            # (plot_metrics.py _SUMMARY_COLS, rmsd_summary.txt) looks for.
            _RENAME = {
                "val_rmsd_mean":   "val_ep_rmsd_mean",
                "val_rmsd_median": "val_ep_rmsd_median",
                "val_rmsd_std":    "val_ep_rmsd_std",
            }
            for k, v in val_epoch_metrics.items():
                self.log(_RENAME.get(k, k), v, sync_dist=True, on_epoch=True)
            self._val_step_outputs.clear()
        finally:
            if ema_cb is not None:
                ema_cb.restore(self.parameters())

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
            3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm and self.local_rank == 0:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')

    @torch.no_grad()
    def eval_rmsd(
        self,
        loader,
        verbose: bool = True,
        write_xyz: bool = False,
        bz: int = 48,
        localpath: str = "sb/ot_ode-10/",
        refpath: str = "ref_ts/",
        max_num_batch: Optional[int] = None,
    ):
        outputs, rmsds = [], []
        for ii, batch in tqdm(enumerate(loader), total=len(loader)):
            if verbose:
                print(f"batch #{ii} / {len(loader)}")
            res, _rmsds = self.eval_sample_batch(
                batch,
                return_rmsd=True,
                write_xyz=write_xyz,
                batch_idx=ii,
                bz=bz,
                localpath=localpath,
                refpath=refpath,
            )
            outputs.append(res)
            rmsds += _rmsds

            if max_num_batch is not None and ii > max_num_batch:
                break
        res = average_over_batch_metrics(outputs, allowed=self.eval_keys)
        return res, rmsds
