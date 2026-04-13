"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import json as _json
import logging
import pickle
import random
import re
import sqlite3
from pathlib import Path

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

# Compiled once at import time.
# _RXN_SUFFIX_RE  – strips  the trailing conformer index, yielding the base reaction ID.
# _CONF_IDX_RE    – captures the trailing conformer index as an integer.
# e.g. "Halogen_C6FH7O_rxn18781_42" → base "Halogen_C6FH7O_rxn18781", conf_idx 42
_RXN_SUFFIX_RE = re.compile(r"_\d+$")
_CONF_IDX_RE   = re.compile(r"_(\d+)$")


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


def _index_rxn_groups(
    db_path: Path,
    prefix: str = "Halogen",
) -> tuple:
    """Scan one file and return all reaction groups whose dand_id starts with
    ``prefix``.

    Strategy
    --------
    1. **Index** – For ``prefix="Halogen"``, JOIN ``systems`` with
       ``species WHERE Z IN (9,17,35)`` to restrict to rows that contain
       F, Cl, or Br atoms (fast path).  For any other prefix the full
       ``systems`` table is scanned directly.
    2. **Filter** – locate the JSON payload in the data BLOB (skip the
       8-byte binary header via ``blob.find(b'{')``), parse with
       ``json.loads``, keep only rows where ``dand_id`` starts with
       ``prefix`` (case-sensitive).
    3. **Group** – strip the trailing ``_<digits>`` conformer index from
       ``dand_id`` (using :data:`_RXN_SUFFIX_RE`) to obtain the base
       reaction ID, then group row IDs by that key.

    No sampling is performed here; sampling happens globally in
    :class:`HaloSQLiteDataset.__init__` after all files are scanned.

    Parameters
    ----------
    db_path : Path
        Path to an ASE SQLite ``.db`` file.
    prefix : str
        ``dand_id`` prefix to keep.  Use ``"Halogen"`` (default) for the
        Halo8 halogen dataset or ``"T1x"`` for the Transition1x dataset.

    Returns
    -------
    tuple: ``(reaction_groups, total_rows, matched_rows)``
        * ``reaction_groups`` – ``dict[str, list[tuple]]`` mapping each base
          reaction ID to a list of ``(row_id, conformer_idx, energy)`` triples
          for every conformer of that reaction in this file
        * ``total_rows``      – total rows in the file (for logging)
        * ``matched_rows``    – rows that passed the prefix filter
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    cur = conn.cursor()

    # Total row count (used only for the log message).
    cur.execute("SELECT COUNT(*) FROM systems")
    total_rows = cur.fetchone()[0]

    # For Halogen data, the species JOIN restricts to rows containing
    # F (Z=9), Cl (Z=17), or Br (Z=35), which is much faster than a full
    # systems scan (~1.3 M rows).  For other prefixes (e.g. T1x) all atom
    # types are present, so we scan the full table.
    if prefix == "Halogen":
        cur.execute("""
            SELECT s.id, s.data, s.energy
            FROM   systems s
            INNER JOIN (
                SELECT DISTINCT id FROM species WHERE Z IN (9, 17, 35)
            ) h ON s.id = h.id
        """)
    else:
        cur.execute("SELECT id, data, energy FROM systems")

    reaction_groups: dict = {}
    matched_rows = 0

    for row_id, blob, energy in cur:
        if not blob:
            continue
        # Locate the JSON object — skip the 8-byte binary header prepended
        # by ASE's encoder (find the first '{' byte).
        start = blob.find(b"{")
        if start == -1:
            continue
        try:
            dand_id = _json.loads(blob[start:]).get("dand_id", "")
        except Exception:
            continue
        if not str(dand_id).startswith(prefix):
            continue
        matched_rows += 1
        base_rxn = _RXN_SUFFIX_RE.sub("", dand_id)
        m = _CONF_IDX_RE.search(dand_id)
        conf_idx = int(m.group(1)) if m else 0
        reaction_groups.setdefault(base_rxn, []).append(
            (row_id, conf_idx, energy)
        )

    conn.close()
    return reaction_groups, total_rows, matched_rows


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
        data_limit (int, optional): If set, randomly sample exactly this
            many **reaction groups** from the global pool of all
            Halogen-prefixed groups found across all Halo_*.db files.
            For each chosen group exactly **3 conformers** are kept:

            * **R  (label 0)** – conformer with the lowest ``dand_id`` index
            * **TS (label 1)** – conformer with the highest energy (saddle point)
            * **P  (label 2)** – conformer with the highest ``dand_id`` index

            All other intermediate conformers are discarded.
            Defaults to ``None`` (keep all groups, 3 conformers each).
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
        prefix: str = "Halogen",
        data_limit: int = None,
        transform=None,
        center: bool = True,
        atom_mapping: dict = None,
    ):
        super().__init__()

        self.src = Path(src)
        self.prefix = prefix
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

        # Phase A: scan all files, build global reaction-group pool
        # global_groups: base_rxn -> list of (db_idx, row_id, conf_idx, energy)
        global_groups: dict = {}

        for db_idx, db_path in enumerate(db_paths):
            rxn_groups, total, matched_rows = _index_rxn_groups(db_path, prefix)

            if matched_rows == 0:
                logger.warning(
                    "%s: no %s-prefixed dand_id found - 0 samples from this file",
                    db_path.name, prefix,
                )

            for base_rxn, conformers in rxn_groups.items():
                entries = global_groups.setdefault(base_rxn, [])
                for row_id, conf_idx, energy in conformers:
                    entries.append((db_idx, row_id, conf_idx, energy))

            logger.info(
                "%s: %d rows, %d %s entries, %d reaction group(s)",
                db_path.name, total, matched_rows, prefix, len(rxn_groups),
            )

        # Phase B: sample reaction groups globally
        all_rxn_keys = list(global_groups.keys())
        n_total_groups = len(all_rxn_keys)

        if data_limit is not None and data_limit < n_total_groups:
            chosen_keys = random.sample(all_rxn_keys, data_limit)
            logger.info(
                "Sampled %d reaction groups from %d total",
                data_limit, n_total_groups,
            )
        else:
            chosen_keys = all_rxn_keys
            logger.info("Keeping all %d reaction groups", n_total_groups)

        # Phase C: pick R / TS / P for each group, build per-file lists
        # Each chosen group contributes exactly 3 (row_id, label) pairs:
        #   R  (label 0) = conformer with minimum conf_idx  (reactant end)
        #   TS (label 1) = conformer with maximum energy    (saddle point)
        #   P  (label 2) = conformer with maximum conf_idx  (product end)
        per_file_rows: list = [[] for _ in db_paths]

        for base_rxn in chosen_keys:
            entries = global_groups[base_rxn]
            # entries: [(db_idx, row_id, conf_idx, energy), ...]
            r_entry  = min(entries, key=lambda e: e[2])   # min conf_idx -> R
            p_entry  = max(entries, key=lambda e: e[2])   # max conf_idx -> P
            ts_entry = max(entries, key=lambda e: e[3])   # max energy   -> TS
            for entry, label in ((r_entry, 0), (ts_entry, 1), (p_entry, 2)):
                db_idx, row_id = entry[0], entry[1]
                per_file_rows[db_idx].append((row_id, label))

        # Sort by row_id; keep labels aligned.
        sorted_pairs = [sorted(rows, key=lambda x: x[0]) for rows in per_file_rows]
        self._row_ids = [[r for r, _ in pf] for pf in sorted_pairs]
        self._labels  = [[l for _, l in pf] for pf in sorted_pairs]
        self._counts  = [len(ids) for ids in self._row_ids]

        # Phase D: summary
        self._cumulative = np.cumsum(self._counts).tolist()
        self.num_samples = sum(self._counts)
        summary = (
            f"HaloSQLiteDataset: {self.num_samples} total conformers "
            f"(3 per group: R/TS/P) from {len(chosen_keys)} reaction group(s) "
            f"across {len(db_paths)} Halo* file(s)"
        )
        if data_limit is not None:
            summary += f" [data_limit={data_limit} groups]"
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
        # Use the pre-sampled, 1-based row ID and its R/TS/P label.
        row_id = self._row_ids[db_idx][el_idx]
        label  = self._labels[db_idx][el_idx]

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
            label=torch.tensor([label], dtype=torch.long),
        )

        if self.transform is not None:
            data = self.transform(data)

        return data

    def close_db(self):
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
