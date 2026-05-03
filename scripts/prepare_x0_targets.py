"""
Halo8/Transition1x 데이터셋에서 각 반응의 approximate TS를 추출한다.

각 반응 경로(~1000 snapshots)에서:
1. 에너지가 최대인 구조를 approximate TS로 선택
2. (R, P, TS_approx) 삼중항을 저장

사용법:
    python scripts/prepare_x0_targets.py \
        --halo8_path PATH_TO_HALO8 \
        --output_path data/x0_training_data/x0_training_data.npz
"""
import argparse
import json as _json
import re
import sqlite3
from pathlib import Path

import numpy as np


# 프로젝트의 dand_id 컨벤션과 동일: 마지막 conformer index를 분리
_RXN_SUFFIX_RE = re.compile(r"_\d+$")
_CONF_IDX_RE = re.compile(r"_(\d+)$")


def extract_approximate_ts(pathway_positions, pathway_energies):
    """
    반응 경로에서 에너지가 최대인 구조를 approximate TS로 선택.

    Args:
        pathway_positions: (T, N, 3) — 경로상 T개의 구조
        pathway_energies: (T,) — 에너지

    Returns:
        ts_approx: (N, 3)
        ts_idx: int
        energy_max: float
    """
    ts_idx = int(np.argmax(pathway_energies))
    return pathway_positions[ts_idx], ts_idx, float(pathway_energies[ts_idx])


def iter_halo8_pathways(halo8_path: str, prefix: str = "Halogen",
                        max_groups: int = None):
    """
    Halo8 데이터셋의 각 반응 경로를 순회하는 generator.

    프로젝트의 데이터 형식(ASE SQLite db; reactot/dataset/ff_lmdb.py 참고)에 맞춘
    구현. ``Halo_*.db`` 들을 모두 스캔하여, ``dand_id`` prefix가 일치하는 row들을
    base reaction id 기준으로 묶고 conformer index 순으로 정렬해 반환한다.

    Yields: (pathway_id, pathway_positions, pathway_energies, atomic_numbers)
        - pathway_positions : (T, N, 3) float64
        - pathway_energies  : (T,)      float64
        - atomic_numbers    : (N,)      int32
    """
    src = Path(halo8_path)
    db_paths = sorted(src.glob("Halo_*.db")) if src.is_dir() else [src]
    if len(db_paths) == 0:
        raise FileNotFoundError(
            f"No Halo_*.db files found under {src}. "
            "Provide --halo8_path that points at a directory containing Halo_*.db, "
            "or a single .db file."
        )

    # global_groups: base_rxn -> list[(db_idx, row_id, conf_idx, energy)]
    # max_groups: stop after the indexer has seen this many distinct base_rxns
    # (smoke-test convenience — full scan of a 3.6GB Halo_*.db is slow).
    global_groups: dict = {}
    stop_indexing = False
    for db_idx, db_path in enumerate(db_paths):
        if stop_indexing:
            break
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT id, data, energy FROM systems")
        for row_id, blob, energy in cur:
            if not blob:
                continue
            start = blob.find(b"{")
            if start == -1:
                continue
            try:
                dand_id = _json.loads(blob[start:]).get("dand_id", "")
            except Exception:
                continue
            dand_id_s = str(dand_id)
            if not dand_id_s.startswith(prefix):
                continue
            base_rxn = _RXN_SUFFIX_RE.sub("", dand_id_s)
            m = _CONF_IDX_RE.search(dand_id_s)
            conf_idx = int(m.group(1)) if m else 0
            global_groups.setdefault(base_rxn, []).append(
                (db_idx, row_id, conf_idx, energy)
            )
            if max_groups is not None and len(global_groups) > max_groups:
                stop_indexing = True
                break
        conn.close()

    # Lazy connection cache for the actual position/numbers fetch
    _conns: dict = {}

    def _get_conn(db_idx: int):
        if db_idx not in _conns:
            _conns[db_idx] = sqlite3.connect(
                str(db_paths[db_idx]), check_same_thread=False
            )
        return _conns[db_idx]

    try:
        for base_rxn, entries in global_groups.items():
            # conformer index 순으로 정렬 → R end 부터 P end 까지의 경로
            entries_sorted = sorted(entries, key=lambda e: e[2])
            positions_list = []
            energies_list = []
            atomic_numbers_ref = None
            for db_idx, row_id, _conf_idx, energy in entries_sorted:
                cur = _get_conn(db_idx).cursor()
                cur.execute(
                    "SELECT numbers, positions, energy, natoms "
                    "FROM systems WHERE id = ?",
                    (row_id,),
                )
                row = cur.fetchone()
                if row is None:
                    continue
                numbers_blob, pos_blob, e, natoms = row
                numbers = np.frombuffer(numbers_blob, dtype=np.int32).copy()
                positions = (
                    np.frombuffer(pos_blob, dtype=np.float64)
                    .reshape(natoms, 3)
                    .copy()
                )
                if atomic_numbers_ref is None:
                    atomic_numbers_ref = numbers
                else:
                    # 원자 순서/조성이 경로 내에서 동일해야 함
                    if numbers.shape != atomic_numbers_ref.shape or not np.array_equal(
                        numbers, atomic_numbers_ref
                    ):
                        # 비일관 → 이 conformer 스킵
                        continue
                positions_list.append(positions)
                energies_list.append(float(e if e is not None else energy or 0.0))

            if len(positions_list) < 3 or atomic_numbers_ref is None:
                continue

            yield (
                base_rxn,
                np.stack(positions_list, axis=0),
                np.asarray(energies_list, dtype=np.float64),
                atomic_numbers_ref,
            )
    finally:
        for c in _conns.values():
            try:
                c.close()
            except Exception:
                pass


def process_halo8_dataset(halo8_path: str, output_npz: str, prefix: str = "Halogen",
                          max_groups: int = None):
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos_R_list, pos_P_list, ts_list, z_list, pid_list = [], [], [], [], []

    for i, (pid, positions, energies, atomic_numbers) in enumerate(
        iter_halo8_pathways(halo8_path, prefix=prefix, max_groups=max_groups)
    ):
        if len(positions) < 3:
            continue
        ts_approx, _, _ = extract_approximate_ts(positions, energies)
        pos_R_list.append(positions[0])
        pos_P_list.append(positions[-1])
        ts_list.append(ts_approx)
        z_list.append(atomic_numbers)
        pid_list.append(pid)
        if max_groups is not None and (i + 1) >= max_groups:
            break

    # 분자 크기가 다양하므로 object array로 저장
    np.savez(
        output_path,
        pos_R=np.array(pos_R_list, dtype=object),
        pos_P=np.array(pos_P_list, dtype=object),
        ts_approx=np.array(ts_list, dtype=object),
        atomic_numbers=np.array(z_list, dtype=object),
        pathway_ids=np.array(pid_list),
    )
    print(f"Extracted {len(pos_R_list)} training samples → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--halo8_path', required=True)
    parser.add_argument('--output_path', default='data/x0_training_data/x0_training_data.npz')
    parser.add_argument('--prefix', default='Halogen',
                        help="dand_id prefix filter (Halogen, T1x, ...)")
    parser.add_argument('--max_groups', type=int, default=None,
                        help="cap the number of reaction groups (smoke test).")
    args = parser.parse_args()

    process_halo8_dataset(args.halo8_path, args.output_path,
                          prefix=args.prefix, max_groups=args.max_groups)
