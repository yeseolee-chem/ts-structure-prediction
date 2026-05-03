"""Ablation study for Idea 2-E ensemble size K and perturbation scale sigma_base.

Sweeps the (K, sigma_base) grid on a validation dataset and writes a JSON
summary capturing TS RMSD (mean / median / std) plus wall-clock time per
reaction. The output JSON is consumed by downstream plotting helpers.

Usage:
    python scripts/evaluate_ensemble.py \\
        --checkpoint PATH \\
        --data_path PATH \\
        --K_values 1 3 5 10 20 \\
        --sigma_values 0.05 0.1 0.2 0.5 \\
        --output results/ensemble_ablation.json

Output schema (per row in the resulting JSON list):
    {
      "K": int,
      "sigma": float,
      "rmsd_mean":   float,   # Kabsch-aligned RMSD vs. ground-truth TS
      "rmsd_median": float,
      "rmsd_std":    float,
      "time_mean_s": float,   # mean wall-clock seconds per reaction
      "n_samples":   int,     # how many reactions actually ran
      "n_failures":  int,     # how many were skipped due to errors
    }

Loading the model + dataset is project-specific; the two helpers
``load_model`` and ``load_dataset`` are stubs you point at the right
checkpoint / .pkl / .lmdb. The defaults match the rp-E layout
(SBModule + ProcessedTS1x / ProcessedHalo8).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def evaluate_one_setting(
    model,
    dataset: Iterable[Dict[str, Any]],
    K: int,
    sigma: float,
    nfe: int = 10,
    seed: Optional[int] = None,
    selection_method: str = "rmsd_consensus",
) -> Dict[str, Any]:
    """Evaluate one (K, sigma) point on the supplied dataset iterable.

    Each ``sample`` in the iterable must yield a dict with at minimum:
        pos_R, pos_P                : (N, 3) numpy arrays
        atomic_numbers              : (N,) numpy int array
        ts_true                     : (N, 3) numpy ground-truth TS
        representations, conditions : the pre-built model batch (for
                                      `predict_ts_ensemble`)

    Returns aggregated stats; failed reactions are skipped but counted in
    ``n_failures`` so a bad sample never silently masks a regression.
    """
    from reactot.run_model import predict_ts_ensemble
    from reactot.utils.initial_guess import kabsch_rmsd

    rmsds: List[float] = []
    times: List[float] = []
    failures = 0

    for sample in dataset:
        try:
            t0 = time.time()
            result = predict_ts_ensemble(
                model,
                representations=sample["representations"],
                conditions=sample["conditions"],
                pos_R=sample["pos_R"],
                pos_P=sample["pos_P"],
                atomic_numbers=sample["atomic_numbers"],
                K=K,
                sigma_base=sigma,
                nfe=nfe,
                seed=seed,
                selection_method=selection_method,
            )
            times.append(time.time() - t0)
            rmsds.append(kabsch_rmsd(result["best_ts"], sample["ts_true"]))
        except Exception:
            failures += 1
            traceback.print_exc()
            continue

    if not rmsds:
        return {
            "K": K,
            "sigma": sigma,
            "rmsd_mean": float("nan"),
            "rmsd_median": float("nan"),
            "rmsd_std": float("nan"),
            "time_mean_s": float("nan"),
            "n_samples": 0,
            "n_failures": failures,
        }
    return {
        "K": K,
        "sigma": sigma,
        "rmsd_mean": float(np.mean(rmsds)),
        "rmsd_median": float(np.median(rmsds)),
        "rmsd_std": float(np.std(rmsds)),
        "time_mean_s": float(np.mean(times)),
        "n_samples": len(rmsds),
        "n_failures": failures,
    }


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a SBModule from a Lightning checkpoint and put it in eval mode.

    Project-specific defaults match the layout that train_rpsb_ts1x.py uses
    on rp-E: SBModule + LEFTNet + EnSB. Kept minimal so it can be replaced
    by a project-specific loader if the checkpoint format changes.
    """
    import torch as _torch

    from reactot.trainer.pl_trainer import SBModule

    model = SBModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )
    model = model.eval().to(device)
    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    return model


def load_dataset(data_path: str, *, dataset_kind: str = "TS1x"):
    """Iterate over the validation split, yielding samples for the ablation.

    Stub: project-specific. Replace with the loader that matches your
    `data_path`. The shape of the yielded dicts is documented in
    `evaluate_one_setting`. Raises NotImplementedError by default so a
    misconfigured run fails loudly rather than silently logging zero RMSDs.
    """
    raise NotImplementedError(
        "Replace load_dataset() with the project-specific loader. The yielded "
        "dicts must contain pos_R, pos_P, atomic_numbers, ts_true, "
        "representations, conditions. See evaluate_one_setting() for the "
        "exact schema."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Idea 2-E ensemble ablation: sweep (K, sigma_base) on a val set.",
    )
    parser.add_argument("--checkpoint", required=True, help="SBModule .ckpt path")
    parser.add_argument("--data_path", required=True, help="Validation dataset path")
    parser.add_argument(
        "--dataset_kind",
        choices=["TS1x", "Halo8"],
        default="TS1x",
    )
    parser.add_argument(
        "--K_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Ensemble sizes to sweep.",
    )
    parser.add_argument(
        "--sigma_values",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.5],
        help="Base perturbation scales (Angstrom) to sweep.",
    )
    parser.add_argument("--nfe", type=int, default=10, help="ODE function evals.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_method",
        choices=["rmsd_consensus", "midpoint_distance"],
        default="rmsd_consensus",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (cuda / cpu).",
    )
    parser.add_argument(
        "--output",
        default="results/ensemble_ablation.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    dataset = list(load_dataset(args.data_path, dataset_kind=args.dataset_kind))

    results: List[Dict[str, Any]] = []
    for K in args.K_values:
        for sigma in args.sigma_values:
            print(f"[ensemble-ablation] K={K} sigma={sigma:.3f} ...", flush=True)
            results.append(
                evaluate_one_setting(
                    model,
                    dataset,
                    K=K,
                    sigma=sigma,
                    nfe=args.nfe,
                    seed=args.seed,
                    selection_method=args.selection_method,
                )
            )
            print(f"  -> {results[-1]}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ensemble-ablation] wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
