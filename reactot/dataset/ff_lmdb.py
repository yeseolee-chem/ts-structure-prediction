"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import pickle
import random
import re
import sqlite3
from pathlib import Path

import ase.db

logger = logging.getLogger(__name__)

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# Atom mapping for Halo8 dataset: H, C, N, O, F, S, Br
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 35: 6}
N_HALO_ATOM_TYPES = len(HALO_ATOM_MAPPING)

# Compiled once at import time.  Strips the trailing conformer/frame index
# from a dand_id string, e.g. "T1x_C2H2N2O_rxn00001_99" → "T1x_C2H2N2O_rxn00001".
_RXN_SUFFIX_RE = re.compile(r"_\d+$")


class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, src, transform=None, **kwargs):
        super(LmdbDataset, self).__init__()

        self.path = Path(src)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=1099511627776 * 2,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


class HaloSQLiteDataset(Dataset):
    r"""Dataset for loading ASE SQLite databases (Halo8 format).

    Each .db file is an ASE-format SQLite database containing molecular
    structures with atomic numbers, positions, energies, and forces.

    Supports atoms: H(1), C(6), N(7), O(8), F(9), S(16), Br(35).

    Returns :class:`torch_geometric.data.Data` objects with:
        - ``pos``      : atom positions  [N, 3]
        - ``one_hot``  : one-hot atom type [N, n_atom_types]
        - ``charges``  : atomic numbers  [N, 1]
        - ``ae``       : total energy    scalar
        - ``forces``   : atomic forces   [N, 3]
        - ``natoms``   : atom count      scalar

    Args:
        src (str): Path to directory containing ``*.db`` files.
        data_limit (int, optional): If set, randomly sample up to this many
            rows per **reaction group**.  Groups are formed by reading
            ``dand_id`` from ``row.data`` and stripping the trailing
            ``_<digits>`` conformer index (e.g. ``T1x_C2H2N2O_rxn00001_99``
            → ``T1x_C2H2N2O_rxn00001``).  Defaults to ``None`` (load all
            rows in every group).
        transform (callable, optional): Data transform applied after
            loading each sample. Defaults to ``None``.
        center (bool): Subtract the centroid from positions.
            Defaults to ``True``.
        atom_mapping (dict): Mapping from atomic number to class index.
            Defaults to :data:`HALO_ATOM_MAPPING`.
    """

    def __init__(
        self,
        src: str,
        data_limit: int = None,
        transform=None,
        center: bool = True,
        atom_mapping: dict = None,
    ):
        super().__init__()

        self.src = Path(src)
        self.transform = transform
        self.center = center
        self.atom_mapping = atom_mapping if atom_mapping is not None else HALO_ATOM_MAPPING
        self.n_atom_types = len(self.atom_mapping)

        # Only load files whose names start with 'Halo'.
        all_db_paths = sorted(self.src.glob("*.db"))
        db_paths = [p for p in all_db_paths if p.name.startswith("Halo")]
        assert len(db_paths) > 0, (
            f"No .db files starting with 'Halo' found in '{self.src}' "
            f"(found {len(all_db_paths)} total .db files)"
        )
        self._db_paths = db_paths

        # For each file, group rows by base reaction ID (read from row.data),
        # then randomly sample up to data_limit rows from each reaction group.
        self._row_ids: list = []   # list[list[int]] – one sub-list per file
        self._counts: list = []
        for db_path in db_paths:
            adb = ase.db.connect(str(db_path))
            total = adb.count()

            # Step 1 — group row IDs by base reaction ID.
            # dand_id example : "T1x_C2H2N2O_rxn00001_99"
            # base reaction ID: "T1x_C2H2N2O_rxn00001"  (strip trailing _<digits>)
            reaction_groups: dict = {}
            for row in adb:
                dand_id = str((row.data or {}).get("dand_id", ""))
                base_rxn = _RXN_SUFFIX_RE.sub("", dand_id) if dand_id else "__no_dand_id__"
                reaction_groups.setdefault(base_rxn, []).append(row.id)

            n_groups = len(reaction_groups)

            # Step 2 — sample up to data_limit rows from each reaction group.
            sampled_ids: list = []
            for ids in reaction_groups.values():
                if data_limit is not None and data_limit < len(ids):
                    sampled_ids.extend(random.sample(ids, data_limit))
                else:
                    sampled_ids.extend(ids)

            # Step 3 — sort for stable bisect-based indexing in __getitem__.
            sampled_ids.sort()

            self._row_ids.append(sampled_ids)
            self._counts.append(len(sampled_ids))

            msg = f"{db_path.name}: {total} rows, {n_groups} reaction group(s)"
            if data_limit is not None:
                msg += f"  →  sampled {len(sampled_ids)} (≤{data_limit}/group)"
            logger.info(msg)

        self._cumulative = np.cumsum(self._counts).tolist()
        self.num_samples = sum(self._counts)
        summary = (
            f"HaloSQLiteDataset: {self.num_samples} total samples "
            f"from {len(db_paths)} Halo* file(s)"
        )
        if data_limit is not None:
            summary += f" [data_limit={data_limit} rows per reaction group]"
        logger.info(summary)

        # Lazy per-process SQLite connection cache (populated in __getitem__).
        self._conns: dict = {}

    def __len__(self) -> int:
        return self.num_samples

    def _get_conn(self, db_idx: int) -> sqlite3.Connection:
        """Return a cached read-only SQLite connection for ``db_idx``."""
        if db_idx not in self._conns:
            self._conns[db_idx] = sqlite3.connect(
                str(self._db_paths[db_idx]),
                check_same_thread=False,
            )
        return self._conns[db_idx]

    def __getitem__(self, idx: int) -> Data:
        # Locate which db file and which position within that file's sample list.
        db_idx = bisect.bisect(self._cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._cumulative[db_idx - 1]
        # Use the pre-sampled, 1-based row ID for this position.
        row_id = self._row_ids[db_idx][el_idx]

        conn = self._get_conn(db_idx)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT numbers, positions, energy, forces, natoms "
            "FROM systems WHERE id = ?",
            (row_id,),
        )
        row = cursor.fetchone()
        assert row is not None, (
            f"Row id={row_id} not found in {self._db_paths[db_idx]}"
        )

        numbers_blob, pos_blob, energy, forces_blob, natoms = row

        numbers = np.frombuffer(numbers_blob, dtype=np.int32).copy()
        positions = np.frombuffer(pos_blob, dtype=np.float64).reshape(natoms, 3).copy()
        forces = np.frombuffer(forces_blob, dtype=np.float64).reshape(natoms, 3).copy()

        # Convert to tensors.
        pos = torch.tensor(positions, dtype=torch.float32)
        if self.center:
            pos = pos - pos.mean(dim=0)

        atom_indices = torch.tensor(
            [self.atom_mapping[int(z)] for z in numbers], dtype=torch.long
        )
        one_hot = F.one_hot(atom_indices, num_classes=self.n_atom_types).float()
        charges = torch.tensor(numbers, dtype=torch.float32).unsqueeze(1)
        ae = torch.tensor([energy], dtype=torch.float32)
        forces_t = torch.tensor(forces, dtype=torch.float32)

        data = Data(
            pos=pos,
            one_hot=one_hot,
            charges=charges,
            ae=ae,
            forces=forces_t,
            natoms=torch.tensor([natoms], dtype=torch.long),
        )

        if self.transform is not None:
            data = self.transform(data)

        return data

    def close_db(self):
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
