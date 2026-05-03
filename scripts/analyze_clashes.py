"""
Halo8 validation set에서 (R+P)/2의 clash 통계를 분석한다.

사용법:
    python scripts/analyze_clashes.py \\
        --data_path PATH_TO_HALO8 \\
        --output results/clash_analysis.json

출력:
    - 전체 반응 중 clash가 있는 비율
    - 할로겐 포함 반응 vs 비포함 반응의 clash 비율 비교
    - 원소쌍별 clash 빈도
    - vdW correction 전후 비교
"""
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

from reactot.utils.initial_guess import (
    compute_vdw_corrected_midpoint, count_clashes
)


def analyze_one_reaction(pos_R, pos_P, atomic_numbers):
    """단일 반응에 대해 midpoint vs corrected의 clash 통계를 반환."""
    midpoint = 0.5 * (pos_R + pos_P)
    corrected = compute_vdw_corrected_midpoint(pos_R, pos_P, atomic_numbers)

    return {
        'midpoint': count_clashes(midpoint, atomic_numbers),
        'corrected': count_clashes(corrected, atomic_numbers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output', default='results/clash_analysis.json')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # === 데이터 로딩 (프로젝트의 dataset loader로 교체) ===
    # reactions = load_halo8_reactions(args.data_path)
    # 각 reaction은 (pos_R, pos_P, atomic_numbers) 튜플이라고 가정
    reactions = []  # TODO: 프로젝트 데이터 로딩 코드로 교체

    pair_counter_midpoint = Counter()
    n_with_clash_midpoint = 0
    n_with_clash_corrected = 0
    n_total = 0

    for pos_R, pos_P, z in reactions:
        result = analyze_one_reaction(pos_R, pos_P, z)
        n_total += 1
        if result['midpoint']['n_clashes'] > 0:
            n_with_clash_midpoint += 1
        if result['corrected']['n_clashes'] > 0:
            n_with_clash_corrected += 1

        for i, j, _, _ in result['midpoint']['clash_pairs']:
            pair = tuple(sorted([int(z[i]), int(z[j])]))
            pair_counter_midpoint[pair] += 1

    stats = {
        'midpoint': {
            'total_reactions': n_total,
            'reactions_with_clash': n_with_clash_midpoint,
            'pair_frequency': {f"{a}-{b}": c
                               for (a, b), c in pair_counter_midpoint.items()},
        },
        'corrected': {
            'total_reactions': n_total,
            'reactions_with_clash': n_with_clash_corrected,
        },
    }

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"분석 완료: {n_with_clash_midpoint}/{n_total} 반응에서 midpoint clash 발생, "
          f"보정 후 {n_with_clash_corrected}/{n_total}")


if __name__ == '__main__':
    main()
