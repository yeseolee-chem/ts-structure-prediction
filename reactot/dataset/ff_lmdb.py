"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import pickle
import sqlite3
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# Atom mapping for Halo8 dataset: H, C, N, O, F, S, Br
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 35: 6}
N_HALO_ATOM_TYPES = len(HALO_ATOM_MAPPING)


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
        data_limit (bool): If ``True``, load only the first 1/10th of
            rows from each individual file. Defaults to ``False``.
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

        db_paths = sorted(self.src.glob("*.db"))
        assert len(db_paths) > 0, f"No .db files found in '{self.src}'"
        self._db_paths = db_paths

        # For each file record how many rows are actually used.
        self._counts = []
        for db_path in db_paths:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM systems")
            total = cursor.fetchone()[0]
            conn.close()

            effective = min(total, data_limit) if data_limit is not None else total
            self._counts.append(effective)
            print(
                f"  {db_path.name}: {total} rows"
                + (f"  →  using first {effective}" if data_limit is not None else "")
            )

        self._cumulative = np.cumsum(self._counts).tolist()
        self.num_samples = sum(self._counts)
        print(
            f"HaloSQLiteDataset: {self.num_samples} total samples "
            f"from {len(db_paths)} file(s)"
            + (f" [data_limit={data_limit} per file]" if data_limit is not None else "")
        )

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
        # Locate which db file and which row within that file.
        db_idx = bisect.bisect(self._cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._cumulative[db_idx - 1]
        # SQLite row ids are 1-based and sequential from 1.
        row_id = el_idx + 1

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
