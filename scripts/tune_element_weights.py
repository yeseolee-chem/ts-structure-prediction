"""
Idea 1-B α_Z 하이퍼파라미터를 validation set에서 grid search로 튜닝한다.

사용법:
    python scripts/tune_element_weights.py \
        --checkpoint PATH \
        --data_path PATH_TO_HALO8 \
        --output results/alpha_tuning.json

노트:
- 각 그리드 점마다 validation forward만 돌리고 TS RMSD 평균을 기록한다.
- 재학습이 아니라 "현재 체크포인트가 해당 가중치 분포에서 어떤 분자가
  가장 심하게 실패하는지"를 분석하는 목적. 즉, α_Z의 영향을 빠르게 스크리닝.
- 최종 α_Z로 재학습은 별도 잡에서 돌린다 (train_rpsb_ts1x.py의
  --element-aware-weights / --alpha-* 인자 참조).
- 전체 조합 수가 너무 많으면 sklearn.model_selection.ParameterSampler 또는
  Bayesian optimization (skopt.gp_minimize 등) 으로 전환 권장.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


# Grid search 범위 (Idea 1-B md 스펙)
ALPHA_GRID = {
    "alpha_H":      [0.3, 0.5, 0.7],
    "alpha_C":      [1.0],          # 기준, 고정
    "alpha_N":      [1.0, 1.1, 1.2],
    "alpha_O":      [1.0, 1.1, 1.2],
    "alpha_F":      [1.0, 1.2, 1.4],
    "alpha_S":      [1.1, 1.3, 1.5],
    "alpha_Br":     [1.2, 1.4, 1.6],
    "lambda_decay": [1.5, 2.0, 2.5, 3.0],
    "w_min":        [0.05, 0.1, 0.2],
}


def alpha_dict_from_params(params: dict) -> dict:
    """Grid search 파라미터를 α_Z 딕셔너리로 변환.

    Cl (Z=17) 은 Br과 같은 계수를 쓰되 md 기본값에 맞춰 1.3을 유지.
    필요 시 grid 에 'alpha_Cl' 을 추가해 확장할 것.
    """
    return {
        1:  params["alpha_H"],
        6:  params["alpha_C"],
        7:  params["alpha_N"],
        8:  params["alpha_O"],
        9:  params["alpha_F"],
        16: params["alpha_S"],
        17: 1.3,
        35: params["alpha_Br"],
    }


def iter_grid(grid: dict):
    """Dict-of-lists → iterator of single-point dicts (Cartesian product)."""
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def evaluate_with_weights(
    checkpoint: str,
    data_path: str,
    alpha_dict: dict,
    lambda_decay: float,
    w_min: float,
) -> dict:
    """
    주어진 가중치로 validation TS RMSD를 계산한다.

    (evaluation.py의 entry point를 재사용하는 스텁 — 실제 평가 구현 시
    아래 TODO 를 채운다.)
    """
    # TODO: checkpoint에서 SBModule 로딩
    # TODO: ProcessedTS1x/ProcessedHalo8 를 element_aware_weights=True,
    #       element_alpha=alpha_dict, graph_weights_w_min=w_min,
    #       graph_weights_lambda=lambda_decay 로 인스턴스화
    # TODO: DataLoader → ddpm.ode_sampling → batch_rmsd_sb 로 TS RMSD 계산
    # TODO: 필요 시 Sella 최적화 단계 수 (n_opt_steps) 도 함께 기록
    return {
        "val_rmsd_mean": float("nan"),
        "val_rmsd_median": float("nan"),
        "val_rmsd_std": float("nan"),
        "n_opt_steps_mean": float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search α_Z (Idea 1-B element-aware weighting)."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Trained SBModule checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Halo8 .db directory (or Transition1x .pkl directory).",
    )
    parser.add_argument(
        "--output", default="results/alpha_tuning.json",
        help="Output JSON path for per-grid-point metrics.",
    )
    parser.add_argument(
        "--max_points", type=int, default=None,
        help="Optional cap on grid points (random subsample). Full grid by default.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for random subsample.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points = list(iter_grid(ALPHA_GRID))
    if args.max_points is not None and args.max_points < len(points):
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(points), size=args.max_points, replace=False)
        points = [points[i] for i in idxs]

    print(f"Grid size: {len(points)} points. Writing to {out_path}.")

    results = []
    for i, params in enumerate(points):
        alpha_dict = alpha_dict_from_params(params)
        metrics = evaluate_with_weights(
            checkpoint=args.checkpoint,
            data_path=args.data_path,
            alpha_dict=alpha_dict,
            lambda_decay=params["lambda_decay"],
            w_min=params["w_min"],
        )
        row = {**params, **metrics}
        results.append(row)
        print(f"[{i+1}/{len(points)}] {row}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
