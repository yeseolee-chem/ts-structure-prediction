"""
scripts/precompute_x0_xtb.py
전체 데이터셋에 대해 xTB-refined x0를 사전 계산하여 저장한다.

사용법:
    python scripts/precompute_x0_xtb.py \\
        --data_path PATH_TO_DATASET \\
        --output_path data/x0_xtb_cache/ \\
        --n_workers 4

주의: xTB calculator는 process-level 전역 상태를 사용할 수 있으므로
multiprocessing은 안전하지만 thread는 비추천.
"""
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


def process_single_reaction(args):
    """단일 반응에 대해 xTB-refined x0 계산. (멀티프로세스 워커)"""
    idx, pos_R, pos_P, atomic_numbers, charge = args
    try:
        from reactot.utils.initial_guess import compute_xtb_refined_x0_with_idpp
        x0 = compute_xtb_refined_x0_with_idpp(
            pos_R, pos_P, atomic_numbers,
            xtb_kwargs={'charge': charge},
        )
        return idx, x0, True
    except Exception as e:
        print(f"Reaction {idx} failed: {e}")
        # fallback: midpoint
        return idx, 0.5 * (pos_R + pos_P), False


def load_task_list(data_path: str):
    """
    데이터셋에서 (idx, pos_R, pos_P, atomic_numbers, charge) 튜플 목록을 반환.

    프로젝트의 데이터 형식에 맞게 구현해야 함. 아래는 예시 시그니처:
    """
    # === TODO: 프로젝트의 데이터 로딩 코드로 교체 ===
    # 예: ase.db.connect(data_path)에서 각 row 읽기
    # 또는 npz/lmdb 등의 형식
    raise NotImplementedError(
        "load_task_list를 프로젝트의 dataset loader로 구현하세요"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로딩
    task_list = load_task_list(args.data_path)

    # 병렬 처리
    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_reaction, task_list),
                total=len(task_list),
            ))
    else:
        results = [process_single_reaction(t) for t in tqdm(task_list)]

    # 결과 저장
    n_success = 0
    for idx, x0, success in results:
        np.save(output_dir / f"x0_{idx:06d}.npy", x0)
        if success:
            n_success += 1

    print(f"성공: {n_success} / {len(results)}")


if __name__ == '__main__':
    main()
