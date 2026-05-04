"""
scripts/evaluate_ensemble.py — Combo [CE] K vs sigma ablation.

사용법:
    python scripts/evaluate_ensemble.py \\
        --checkpoint PATH \\
        --data_path PATH \\
        --K_values 1 3 5 10 \\
        --sigma_values 0.05 0.1 0.2 0.5 \\
        --output results/ensemble_ablation.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def evaluate_one_setting(model, dataset, K, sigma):
    from reactot.run_model import predict_ts_CE
    from reactot.utils.initial_guess import kabsch_rmsd

    rmsds = []
    times = []
    for sample in dataset:
        pos_R = sample['pos_R']
        pos_P = sample['pos_P']
        z = sample['atomic_numbers']
        ts_true = sample['ts_true']

        t0 = time.time()
        result = predict_ts_CE(
            model, pos_R, pos_P, z, K=K, sigma_base=sigma,
        )
        times.append(time.time() - t0)
        rmsds.append(kabsch_rmsd(result['best_ts'], ts_true))

    return {
        'K': K, 'sigma': sigma,
        'rmsd_mean': float(np.mean(rmsds)),
        'rmsd_median': float(np.median(rmsds)),
        'rmsd_std': float(np.std(rmsds)),
        'time_mean_s': float(np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--K_values', nargs='+', type=int,
                        default=[1, 3, 5, 10])
    parser.add_argument('--sigma_values', nargs='+', type=float,
                        default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument('--output', default='results/ensemble_ablation.json')
    args = parser.parse_args()

    model = None
    dataset = None
    if model is None or dataset is None:
        raise NotImplementedError(
            "프로젝트의 model/dataset 로딩 코드로 교체하세요"
        )

    results = []
    for K in args.K_values:
        for sigma in args.sigma_values:
            print(f"K={K}, sigma={sigma} 평가 중...")
            results.append(evaluate_one_setting(model, dataset, K, sigma))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
