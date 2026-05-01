"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import hashlib
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


# Atom mapping for Halo8 dataset: H, C, N, O, F, S, Cl, Br (8 element types —
# hence the "Halo8" name). Ordered by atomic number so one-hot indices are
# easy to inspect.
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6, 35: 7}
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


def _normalize_prefix(prefix):
    """Normalize a prefix argument to a tuple of prefix strings.

    Accepts a single string, a tuple/list of strings, or the sentinel
    ``"Mix"`` which expands to ``("Halogen", "T1x")``.  Case-insensitive
    match for ``Mix``.
    """
    if isinstance(prefix, str):
        if prefix.lower() == "mix":
            return ("Halogen", "T1x")
        return (prefix,)
    return tuple(prefix)


# ---------------------------------------------------------------------------
# Deterministic train / val / test split for Halo8.
#
# Why this exists: the previous Halo8 setup pointed train and val datasets at
# the SAME directory and "split" them only by passing different sampling
# seeds (seed=42 for train, seed=43 for val). With a finite pool, two
# independent random samples DO overlap — for data_limit=500 and a pool of
# 2,000 reactions you expect ~25% of the val set to also appear in train.
# That inflates val metrics and makes generalization claims unreliable.
#
# Fix: assign each reaction group to a split *deterministically* by hashing
# the base reaction ID (dand_id with conformer suffix stripped). The hash
# bucket lives in [0, 1); thresholds carve out test/val/train slices that
# never overlap regardless of how data_limit or sampling seed change.
# ---------------------------------------------------------------------------
SPLIT_NAMES = ("all", "train", "val", "test")


def _assign_split(
    base_rxn: str,
    val_fraction: float,
    test_fraction: float,
    split_seed: int = 42,
) -> str:
    """Hash-based deterministic train/val/test assignment.

    Returns one of ``"train" | "val" | "test"``. Same input always returns
    the same split; a given reaction is therefore in exactly one set, full
    stop. Adding/removing data files does not move existing reactions
    across splits.
    """
    h = hashlib.sha256(f"{int(split_seed)}:{base_rxn}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(1 << 32)  # uniform in [0, 1)
    if bucket < float(test_fraction):
        return "test"
    if bucket < float(test_fraction) + float(val_fraction):
        return "val"
    return "train"


def _index_rxn_groups(
    db_path: Path,
    prefix="Halogen",
    max_groups: int = None,
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
    prefixes = _normalize_prefix(prefix)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    cur = conn.cursor()

    # Total row count (used only for the log message).
    cur.execute("SELECT COUNT(*) FROM systems")
    total_rows = cur.fetchone()[0]

    # Fast path only when the single requested prefix is 'Halogen':
    # the species JOIN restricts to rows containing F (Z=9), Cl (Z=17),
    # or Br (Z=35) – much faster than a full systems scan.  For any
    # other prefix set (T1x, Mix, custom) we must scan the full table.
    if prefixes == ("Halogen",):
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
        dand_id_s = str(dand_id)
        if not any(dand_id_s.startswith(p) for p in prefixes):
            continue
        matched_rows += 1
        base_rxn = _RXN_SUFFIX_RE.sub("", dand_id)
        m = _CONF_IDX_RE.search(dand_id)
        conf_idx = int(m.group(1)) if m else 0
        reaction_groups.setdefault(base_rxn, []).append(
            (row_id, conf_idx, energy)
        )
        if max_groups is not None and len(reaction_groups) > max_groups:
            break

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
        prefix="Halogen",
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


class ProcessedHalo8(Dataset):
    """Halo8 adapter with BaseDataset-compatible interface.

    Loads R/TS/P conformer triples from ASE SQLite .db files and exposes them
    in the same dict layout (``pos_0/1/2``, ``one_hot_0/1/2``, ``charge_0/1/2``,
    ``size_0/1/2``, ``mask_0/1/2``) that ``reactot.dataset.base_dataset.BaseDataset``
    uses, so the existing ``collate_fn`` and the OT-FM training pipeline work
    without modification.
    """

    def __init__(
        self,
        npz_path,
        center: bool = True,
        device: str = "cpu",
        zero_charge: bool = False,
        remove_h: bool = False,
        atom_mapping: dict = None,
        data_limit=None,
        prefix="Halogen",
        seed=None,
        max_db_files=None,
        split: str = "all",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        if atom_mapping is None:
            atom_mapping = HALO_ATOM_MAPPING
        # Normalize string like "Mix" → tuple form once, so both the
        # per-file scan path and any downstream logic see the same value.
        prefix = _normalize_prefix(prefix)

        # Validate split kwargs early — silent typos here would silently
        # disable the split (everything would land in "train").
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"Unknown split={split!r}. Expected one of {SPLIT_NAMES}."
            )
        val_fraction = float(val_fraction)
        test_fraction = float(test_fraction)
        if not (0.0 <= val_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
            raise ValueError(
                "val_fraction and test_fraction must each be in [0, 1)."
            )
        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction must be < 1 to leave a train slice."
            )
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = int(split_seed)

        self.center = center
        self.device = device
        self.zero_charge = zero_charge
        self.remove_h = remove_h
        self.atom_mapping = atom_mapping
        self.n_element = len(atom_mapping)
        self.n_fragment = 3
        self.data = {}

        if data_limit is not None:
            data_limit = int(data_limit)
            if data_limit <= 0:
                data_limit = None

        src = Path(npz_path)
        all_db_paths = sorted(src.glob("*.db"))
        db_paths = [p for p in all_db_paths if p.name.startswith("Halo")]
        assert len(db_paths) > 0, (
            f"No .db files starting with 'Halo' found in '{src}' "
            f"(found {len(all_db_paths)} total .db files)"
        )
        if max_db_files is not None and max_db_files > 0:
            db_paths = db_paths[: int(max_db_files)]

        # `prefix` is already a normalized tuple, e.g. ('T1x',) or
        # ('Halogen', 'T1x').  When there are multiple prefixes (Mix mode)
        # we sample independently from each prefix so that the final
        # dataset contains exactly data_limit // n_prefixes reactions from
        # each prefix.  This prevents the old early-termination bug where
        # scanning stopped after finding data_limit groups from the first
        # file — which happened to be all T1x — making Mix identical to T1x.
        n_prefixes = len(prefix)
        per_prefix_limit = (data_limit // n_prefixes) if data_limit is not None else None

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        global_groups: dict = {}

        # The early-termination per-file cap was based on per_prefix_limit
        # alone. With deterministic splits we now also need enough candidates
        # to survive the split filter — only ~val_fraction of scanned
        # reactions land in the val set, so we have to scan more rows up
        # front. Rather than guess a multiplier, drop the cap entirely when
        # split != "all" (val/test), so we scan every file. When split=="all"
        # or "train" the original 3× cap is plenty (train fraction is
        # large).
        scan_full = self.split in ("val", "test")

        for single_prefix in prefix:
            prefix_groups: dict = {}
            # Cap per-file scan to 3× the quota for this prefix so we don't
            # read the whole (multi-million row) file when data_limit is small.
            per_file_cap = (
                None if scan_full
                else (per_prefix_limit * 3 if per_prefix_limit is not None else None)
            )

            for db_idx, db_path in enumerate(db_paths):
                rxn_groups, _total, _matched = _index_rxn_groups(
                    db_path, single_prefix, max_groups=per_file_cap
                )
                for base_rxn, conformers in rxn_groups.items():
                    entries = prefix_groups.setdefault(base_rxn, [])
                    for row_id, conf_idx, energy in conformers:
                        entries.append((db_idx, row_id, conf_idx, energy))
                # Early termination: once we have enough candidates for this
                # prefix, stop scanning further files.
                if (
                    not scan_full
                    and per_prefix_limit is not None
                    and len(prefix_groups) >= per_prefix_limit * 3
                ):
                    break

            # Filter by deterministic split assignment BEFORE the random
            # data_limit sampling. This guarantees train/val/test are
            # disjoint regardless of (seed, data_limit, prefix) settings.
            if self.split == "all":
                split_keys = sorted(prefix_groups.keys())
            else:
                split_keys = sorted(
                    k for k in prefix_groups
                    if _assign_split(
                        k, self.val_fraction, self.test_fraction, self.split_seed
                    ) == self.split
                )
            logger.info(
                "prefix=%s split=%s: %d/%d groups after deterministic split",
                single_prefix, self.split, len(split_keys), len(prefix_groups),
            )

            # Randomly sample per_prefix_limit groups from the split.
            if per_prefix_limit is not None and per_prefix_limit < len(split_keys):
                sampled = rng.sample(split_keys, per_prefix_limit)
            else:
                sampled = split_keys
                if per_prefix_limit is not None and per_prefix_limit > len(split_keys):
                    logger.warning(
                        "prefix=%s split=%s: data_limit/prefix=%d > split size=%d; "
                        "using all available split members",
                        single_prefix, self.split, per_prefix_limit, len(split_keys),
                    )
            logger.info(
                "prefix=%s split=%s: %d candidate groups, selected %d",
                single_prefix, self.split, len(split_keys), len(sampled),
            )
            for k in sampled:
                global_groups[k] = prefix_groups[k]

        chosen_keys = list(global_groups.keys())

        groups_per_frag: dict = {0: [], 1: [], 2: []}
        conn_cache: dict = {}

        def _get_conn(db_idx: int) -> sqlite3.Connection:
            if db_idx not in conn_cache:
                conn_cache[db_idx] = sqlite3.connect(
                    str(db_paths[db_idx]), check_same_thread=False
                )
            return conn_cache[db_idx]

        for base_rxn in chosen_keys:
            entries = global_groups[base_rxn]
            r_entry = min(entries, key=lambda e: e[2])
            p_entry = max(entries, key=lambda e: e[2])
            ts_entry = max(entries, key=lambda e: e[3])
            for entry, frag_idx in ((r_entry, 0), (ts_entry, 1), (p_entry, 2)):
                db_idx, row_id = entry[0], entry[1]
                conn = _get_conn(db_idx)
                cur = conn.cursor()
                cur.execute(
                    "SELECT numbers, positions, natoms FROM systems WHERE id = ?",
                    (row_id,),
                )
                row = cur.fetchone()
                assert row is not None, (
                    f"Row id={row_id} not found in {db_paths[db_idx]}"
                )
                numbers_blob, pos_blob, natoms = row
                numbers = np.frombuffer(numbers_blob, dtype=np.int32).copy()
                positions = (
                    np.frombuffer(pos_blob, dtype=np.float64)
                    .reshape(natoms, 3)
                    .copy()
                )
                groups_per_frag[frag_idx].append(
                    {"numbers": numbers, "positions": positions, "natoms": int(natoms)}
                )

        for conn in conn_cache.values():
            conn.close()

        self.n_samples = len(chosen_keys)

        for frag_idx in (0, 1, 2):
            entries = groups_per_frag[frag_idx]
            self.data[f"size_{frag_idx}"] = torch.tensor(
                [d["natoms"] for d in entries], device=self.device
            )
            pos_list = [
                torch.tensor(d["positions"], device=self.device, dtype=torch.float32)
                for d in entries
            ]
            if self.center:
                pos_list = [p - torch.mean(p, dim=0) for p in pos_list]
            self.data[f"pos_{frag_idx}"] = pos_list

            atom_idx_list = [
                torch.tensor(
                    [self.atom_mapping[int(z)] for z in d["numbers"]],
                    device=self.device,
                    dtype=torch.long,
                )
                for d in entries
            ]
            self.data[f"one_hot_{frag_idx}"] = [
                F.one_hot(_z, num_classes=self.n_element) for _z in atom_idx_list
            ]

            if self.zero_charge:
                self.data[f"charge_{frag_idx}"] = [
                    torch.zeros(
                        size=(d["natoms"], 1), dtype=torch.int64, device=self.device
                    )
                    for d in entries
                ]
            else:
                self.data[f"charge_{frag_idx}"] = [
                    torch.tensor(
                        d["numbers"], device=self.device, dtype=torch.int64
                    ).view(-1, 1)
                    for d in entries
                ]

            self.data[f"mask_{frag_idx}"] = [
                torch.zeros(
                    size=(d["natoms"],), dtype=torch.int64, device=self.device
                )
                for d in entries
            ]

        self.data["condition"] = [
            torch.zeros(size=(1, 1), dtype=torch.int64, device=self.device)
            for _ in range(self.n_samples)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        from reactot.dataset.base_dataset import BaseDataset

        return BaseDataset.collate_fn(batch)
