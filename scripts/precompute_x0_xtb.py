"""
scripts/precompute_x0_xtb.py — Combo [AD] x_0 cache generator.

For each reaction in the dataset, runs Stage 1 (IDPP) + Stage 2 (GFN2-xTB
short refinement) on (R, P) and saves the resulting x_0 to disk. The
generated cache directory can then be passed to ProcessedTS1x via
``x0_cache_dir`` so the training loop bypasses on-the-fly xTB.

xtb-python is process-safe (Pool OK) but not thread-safe. Use Pool with
n_workers > 1 for parallelism.

사용법:
    python scripts/precompute_x0_xtb.py \\
        --data_path PATH_TO_DATASET \\
        --output_path data/x0_xtb_cache_AD/ \\
        --n_workers 4 \\
        --interpolate a_d
"""
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x


def process_single_reaction(args):
    """Single reaction worker. Returns (idx, x0, success_flag)."""
    idx, pos_R, pos_P, atomic_numbers, charge, interpolate = args
    try:
        if interpolate == "a_d":
            from reactot.utils.initial_guess import compute_x0_AD
            x0 = compute_x0_AD(
                pos_R, pos_P, atomic_numbers,
                charge=charge,
            )
        else:
            from reactot.utils.initial_guess import compute_xtb_refined_x0_with_idpp
            x0 = compute_xtb_refined_x0_with_idpp(
                pos_R, pos_P, atomic_numbers,
                xtb_kwargs={'charge': charge},
            )
        return idx, x0, True
    except Exception as e:
        print(f"Reaction {idx} failed: {e}")
        return idx, 0.5 * (pos_R + pos_P), False


def load_task_list(data_path: str, interpolate: str):
    """
    Load (idx, pos_R, pos_P, atomic_numbers, charge, interpolate) tuples
    from the dataset. Replace this with the project's actual loader (npz,
    ASE db, lmdb, ...). Returns an iterable of tuples.
    """
    raise NotImplementedError(
        "load_task_list를 프로젝트의 dataset loader로 구현하세요. "
        "Halo8 (E:/Halo_*.db)을 사용하려면 ASE db 로더 위에 wrap 하세요."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--interpolate', default='a_d',
                        choices=['a_d', 'xtb_refine'])
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = load_task_list(args.data_path, args.interpolate)

    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_reaction, task_list),
                total=len(task_list),
            ))
    else:
        results = [process_single_reaction(t) for t in tqdm(task_list)]

    n_success = 0
    for idx, x0, success in results:
        np.save(output_dir / f"x0_{idx:06d}.npy", x0)
        if success:
            n_success += 1

    print(f"성공: {n_success} / {len(results)}")


if __name__ == '__main__':
    main()
