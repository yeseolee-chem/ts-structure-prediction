"""
scripts/prepare_x0_targets.py — Idea 2-F training target extraction.

For each Halo8 / Transition1x reaction pathway, take the
energy-maximum structure as the "approximate TS" and write
(R, P, TS_approx, atomic_numbers) tuples to npz for x_0 predictor training.

사용법:
    python scripts/prepare_x0_targets.py \\
        --halo8_path PATH \\
        --output_path data/x0_training_data/x0_training_data.npz
"""
import argparse
from pathlib import Path

import numpy as np


def extract_approximate_ts(pathway_positions, pathway_energies):
    ts_idx = int(np.argmax(pathway_energies))
    return (pathway_positions[ts_idx], ts_idx,
            float(pathway_energies[ts_idx]))


def iter_halo8_pathways(halo8_path: str):
    """Project-specific iterator. Replace with the dataset's pathway loader."""
    raise NotImplementedError(
        "iter_halo8_pathways를 프로젝트의 데이터 형식에 맞게 구현하세요. "
        "Halo8 ASE db (E:/Halo_*.db)을 쓰려면 ase.db.connect 위에 wrap 하세요."
    )


def process_halo8_dataset(halo8_path: str, output_npz: str):
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos_R_list, pos_P_list, ts_list, z_list, pid_list = [], [], [], [], []

    for pid, positions, energies, atomic_numbers in iter_halo8_pathways(halo8_path):
        if len(positions) < 3:
            continue
        ts_approx, _, _ = extract_approximate_ts(positions, energies)
        pos_R_list.append(positions[0])
        pos_P_list.append(positions[-1])
        ts_list.append(ts_approx)
        z_list.append(atomic_numbers)
        pid_list.append(pid)

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
    parser.add_argument('--output_path',
                        default='data/x0_training_data/x0_training_data.npz')
    args = parser.parse_args()
    process_halo8_dataset(args.halo8_path, args.output_path)
