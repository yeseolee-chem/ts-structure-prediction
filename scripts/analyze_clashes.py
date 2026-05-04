"""
scripts/analyze_clashes.py — Idea 2-C clash diagnostics.

Compute (R+P)/2 vs vdW-corrected midpoint clash statistics across the
dataset. Loader is project-specific (TODO).
"""
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

from reactot.utils.initial_guess import (
    compute_vdw_corrected_midpoint, count_clashes,
)


def analyze_one_reaction(pos_R, pos_P, atomic_numbers):
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
    reactions = []  # TODO: project-specific loader
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

    print(f"midpoint clash: {n_with_clash_midpoint}/{n_total}, "
          f"corrected: {n_with_clash_corrected}/{n_total}")


if __name__ == '__main__':
    main()
