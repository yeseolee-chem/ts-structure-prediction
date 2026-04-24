import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from reactot.dataset.datasets_config import ATOM_MAPPING, SAM_CHARGED_ATOM_MAPPING
from reactot.utils.weighting import (
    compute_weights_for_batch,
    compute_element_weights_for_batch,
    compute_hybrid_weights,
    compute_bc_weights,
)


class BaseDataset(Dataset):
    def __init__(
        self,
        npz_path,
        center=True,
        zero_charge=False,
        device="cpu",
        remove_h=False,
        n_fragment=3,
        atom_mapping=ATOM_MAPPING,
    ) -> None:
        super().__init__()

        if ".npz" in str(npz_path):
            with np.load(npz_path, allow_pickle=True) as f:
                data = {key: val for key, val in f.items()}
        elif ".pkl" in str(npz_path):
            data = pickle.load(open(npz_path, "rb"))
        else:
            raise ValueError("data file should be either .npz or .pkl")

        self.raw_dataset = data
        self.n_samples = -1
        self.data = {}
        self.n_fragment = n_fragment

        self.remove_h = remove_h
        self.zero_charge = zero_charge
        self.center = center
        self.device = device
        
        self.atom_mapping = atom_mapping
        self.n_element = len(list(atom_mapping.keys()))

    def __len__(self):
        return len(self.data["size_0"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        sizes = []
        for k in batch[0].keys():
            if "size" in k:
                sizes.append(int(k.split("_")[-1]))
        n_fragment = len(sizes)
        out = [{} for _ in range(n_fragment)]
        res = {}
        for prop in batch[0].keys():
            # print(prop)
            if prop not in ["condition", "target", "rmsd", "ediff", "ts_guess"]:
                idx = int(prop.split("_")[-1])
                _prop = prop.replace(f"_{idx}", "")
            if "size" in prop:
                out[idx][_prop] = torch.tensor(
                    [x[prop] for x in batch],
                    device=batch[0][prop].device,
                )
            elif "mask" in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[idx][_prop] = torch.cat(
                    [
                        i * torch.ones(len(x[prop]), device=x[prop].device).long()
                        for i, x in enumerate(batch)
                    ],
                    dim=0,
                )
            elif prop in ["condition", "target", "rmsd", "ediff", "ts_guess"]:
                res[prop] = torch.cat([x[prop] for x in batch], dim=0)
            else:
                out[idx][_prop] = torch.cat([x[prop] for x in batch], dim=0)

        if len(list(res.keys())) == 1:
            return out, res["condition"]
        return out, res

    def attach_atom_weights(
        self,
        r_idx: int = 0,
        p_idx: int = 2,
        n_fragments: int = 3,
        w_min: float = 0.1,
        lambda_decay: float = 2.0,
        element_aware: bool = False,
        alpha_dict: dict = None,
        weighting_scheme: str = "AB",
        interface_max_hop: int = 2,
        beta_angle: float = 1.0,
    ):
        """Precompute graph-distance-based continuous atom weights per sample.

        Args:
            r_idx: fragment index for reactant positions.
            p_idx: fragment index for product positions.
            n_fragments: number of fragments to replicate weights across.
            w_min: minimum weight at graph-infinity (Idea 1-A default 0.1).
            lambda_decay: decay length in hop units (Idea 1-A default 2.0).
            element_aware: Idea 1-B — multiply by α_Z and normalize per-molecule.
            alpha_dict: custom per-element α_Z. When None, uses defaults.
            weighting_scheme: "AB" (default) for graph-distance × α_Z hybrid,
                or "BC" for 3-tier × α_Z (B+C). When "BC", `element_aware` is
                ignored — α_Z is always applied.
            interface_max_hop: BC only — tier-2 max hop (default 2).
            beta_angle: BC only — bond-angle correction strength (default 1.0).
        """
        pos_R_list = self.data[f"pos_{r_idx}"]
        pos_P_list = self.data[f"pos_{p_idx}"]
        if self.zero_charge:
            raise ValueError(
                "attach_atom_weights requires non-zero charge_ tensors for "
                "atomic numbers; run before zero_charge or provide charges."
            )
        charge_list = self.data[f"charge_{r_idx}"]

        if weighting_scheme not in ("AB", "BC"):
            raise ValueError(
                f"Unknown weighting_scheme: {weighting_scheme!r}. "
                "Expected 'AB' or 'BC'."
            )

        weights_per_sample = []
        for pos_R_t, pos_P_t, charge_t in zip(pos_R_list, pos_P_list, charge_list):
            pos_R = pos_R_t.detach().cpu().numpy().astype(np.float64)
            pos_P = pos_P_t.detach().cpu().numpy().astype(np.float64)
            atomic_numbers = charge_t.detach().cpu().numpy().reshape(-1).astype(np.int64)
            if weighting_scheme == "BC":
                # Idea 1-BC: 3-tier × α_Z (B+C). A is used inside C — not re-applied.
                w = compute_bc_weights(
                    pos_R, pos_P, atomic_numbers,
                    w_min=w_min, lambda_decay=lambda_decay,
                    interface_max_hop=interface_max_hop,
                    beta_angle=beta_angle,
                    element_alpha=alpha_dict, normalize=True,
                )
            elif element_aware:
                # Idea 1-AB hybrid: graph-distance × α_Z, per-molecule mean=1
                w = compute_hybrid_weights(
                    pos_R, pos_P, atomic_numbers,
                    w_min=w_min, lambda_decay=lambda_decay,
                    element_alpha=alpha_dict, normalize=True,
                )
            else:
                w = compute_weights_for_batch(
                    pos_R, pos_P, atomic_numbers,
                    w_min=w_min, lambda_decay=lambda_decay,
                )
            weights_per_sample.append(
                torch.tensor(w, dtype=torch.float32, device=self.device)
            )

        for idx in range(n_fragments):
            self.data[f"atom_weights_{idx}"] = weights_per_sample

    def patch_dummy_molecules(self, idx):
        self.data[f"size_{idx}"] = torch.ones_like(
            self.data[f"size_0"], device=self.device,
        )
        self.data[f"pos_{idx}"] = [
            torch.tensor([[0, 0, 0]], device=self.device,)
            for _ in range(self.n_samples)
        ]

        self.data[f"one_hot_{idx}"] = [
            torch.tensor([0], device=self.device,)
            for _ in range(self.n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            F.one_hot(_z, num_classes=self.n_element) for _z in self.data[f"one_hot_{idx}"]
        ]

        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                torch.zeros(size=(1, 1), dtype=torch.int64, device=self.device,)
                for _ in range(self.n_samples)
            ]
        else:
            self.data[f"charge_{idx}"] = [
                torch.ones(size=(1, 1), dtype=torch.int64, device=self.device,)
                for _ in range(self.n_samples)
            ]

        self.data[f"mask_{idx}"] = [
            torch.zeros(size=(1,), dtype=torch.int64, device=self.device,)
            for _ in range(self.n_samples)
        ]

    def process_molecules(self, dataset_name, n_samples, idx, append_charge=None,
                          position_key="positions"):
        data = getattr(self, dataset_name)
        self.data[f"size_{idx}"] = torch.tensor(data["num_atoms"], device=self.device)
        self.data[f"pos_{idx}"] = [
            torch.tensor(
                data[position_key][ii][: data["num_atoms"][ii]],
                device=self.device,
                dtype=torch.float32,
            )
            for ii in range(n_samples)
        ]

        self.data[f"one_hot_{idx}"] = [
            torch.tensor(
                [
                    self.atom_mapping[_at]
                    for _at in data["charges"][ii][: data["num_atoms"][ii]]
                ],
                device=self.device,
            )
            for ii in range(n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            F.one_hot(_z, num_classes=self.n_element)
            for _z in self.data[f"one_hot_{idx}"]
        ]

        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                torch.zeros(size=(_size, 1), dtype=torch.int64, device=self.device,)
                for _size in data["num_atoms"]
            ]
        else:
            if append_charge is None:
                self.data[f"charge_{idx}"] = [
                    torch.tensor(
                        data["charges"][ii][: data["num_atoms"][ii]],
                        device=self.device,
                    ).view(-1, 1)
                    for ii in range(n_samples)
                ]
            else:
                self.data[f"charge_{idx}"] = [
                    torch.cat(
                        [
                            torch.tensor(
                                data["charges"][ii][: data["num_atoms"][ii]],
                                device=self.device,
                            ).view(-1, 1),
                            torch.tensor(
                                [append_charge for _ in range(data["num_atoms"][ii])],
                                device=self.device,
                            ).view(-1, 1),
                        ],
                        dim=1,
                    )
                    for ii in range(n_samples)
                ]

        self.data[f"mask_{idx}"] = [
            torch.zeros(size=(_size,), dtype=torch.int64, device=self.device,)
            for _size in data["num_atoms"]
        ]

        if self.center:
            self.data[f"pos_{idx}"] = [
                pos - torch.mean(pos, dim=0) for pos in self.data[f"pos_{idx}"]
            ]
