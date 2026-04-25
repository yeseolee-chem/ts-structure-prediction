"""Decompose BCD-trained importance weights into tier / element / residual.

Usage:
    python scripts/analyze_bcd.py \\
        --checkpoint checkpoint/.../sb-epoch=XXX.ckpt \\
        --dataset TS1x --data-dir reactot/data/transition1x \\
        --split valid --n-samples 64 --out bcd_decomp.png

Emits a 4-panel PNG:
    (1) scatter of learned vs BC prior weights (identity line = no D residual)
    (2) per-tier boxplot of learned weights (Core / Interface / Peripheral)
    (3) per-element boxplot of learned weights
    (4) residual (learned - prior) vs graph distance to reactive core

And a CSV ``<out>_per_atom.csv`` with one row per atom:
    sample_id, atom_idx, z, tier, graph_dist, alpha, prior_weight,
    learned_weight, residual
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from reactot.utils import compute_BC_prior_with_tier
from reactot.trainer.pl_trainer import SBModule


_TIER_LABEL = {1: "Core", 2: "Interface", 3: "Peripheral"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=["TS1x", "Halo8"], default="TS1x")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--device", default=None, help="cuda / cpu. Defaults to auto.")
    p.add_argument("--out", default="bcd_decomp.png")
    p.add_argument(
        "--prior-kwargs",
        default=None,
        help="Optional JSON dict overriding BC prior hyperparameters.",
    )
    return p.parse_args()


@torch.no_grad()
def _extract_importance(module: SBModule, batch) -> torch.Tensor:
    """Run one forward pass on the dynamics and return per-atom importance
    logits for the target fragment (idx=1). Bypasses the full training path."""
    from reactot.utils import get_edges_index, get_n_frag_switch
    from reactot.diffusion._normalizer import FEATURE_MAPPING

    ensb = module.ddpm
    representations, conditions = batch
    masks = [r["mask"] for r in representations]
    combined_mask = torch.cat(masks)
    edge_index = get_edges_index(combined_mask, remove_self_edge=True)
    fragments_nodes = [r["size"] for r in representations]
    n_frag_switch = get_n_frag_switch(fragments_nodes)

    representations_n = ensb.normalizer.normalize(representations)
    xh_t = [
        torch.cat(
            [r[f] for f in FEATURE_MAPPING],
            dim=1,
        )
        for r in representations_n
    ]
    num_sample = representations[0]["size"].size(0)
    device = representations[0]["pos"].device
    t = torch.zeros((num_sample, 1), device=device)  # any valid t works

    cond = conditions["condition"] if ensb.ts_guess else conditions
    _, _, imp_per_frag = ensb.dynamics(
        xh=xh_t,
        edge_index=edge_index,
        t=t,
        conditions=cond,
        n_frag_switch=n_frag_switch,
        combined_mask=combined_mask,
        edge_attr=None,
    )
    if imp_per_frag is None:
        raise RuntimeError(
            "Checkpoint's dynamics was not built with learn_importance=True; "
            "nothing to decompose."
        )
    return imp_per_frag[ensb.idx]  # (n_target_atoms,)


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    module = SBModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    module.eval()
    module.to(args.device)

    # setup for inference
    if args.data_dir is not None:
        module.training_config["datadir"] = args.data_dir

    # Use the test_dataset entry point since it's the simplest for inference.
    module.setup(stage="test")
    loader = module.test_dataloader(bz=4)

    from reactot.dataset import ATOM_MAPPING  # noqa: F401  (re-export check)

    learned_weights: List[np.ndarray] = []
    prior_weights: List[np.ndarray] = []
    tiers: List[np.ndarray] = []
    alphas: List[np.ndarray] = []
    graph_dists: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []

    w_min = float(module.ddpm.learned_w_min)
    sample_counter = 0

    for batch_idx, batch in enumerate(loader):
        batch = module.transfer_batch_to_device(batch, module.device, dataloader_idx=0)
        representations, _conditions = batch

        r_rep = representations[0]
        t_rep = representations[module.ddpm.idx]
        p_rep = representations[2]

        # Shapes: pos (n_total, 3), charge (n_total, 1), mask (n_total,)
        pos_r = r_rep["pos"].cpu().numpy()
        pos_p = p_rep["pos"].cpu().numpy()
        z_all = t_rep["charge"].view(-1).cpu().numpy()
        mask = t_rep["mask"].cpu().numpy()

        imp_logits = _extract_importance(module, batch).cpu().numpy()
        learned = 1.0 / (1.0 + np.exp(-imp_logits)) * (1.0 - w_min) + w_min

        sizes = t_rep["size"].cpu().numpy()
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        for s in range(len(sizes)):
            lo, hi = offsets[s], offsets[s + 1]
            p_r = pos_r[lo:hi]
            p_p = pos_p[lo:hi]
            z = z_all[lo:hi].astype(np.int64)
            w_prior, meta = compute_BC_prior_with_tier(p_r, p_p, z)

            learned_weights.append(learned[lo:hi])
            prior_weights.append(w_prior)
            tiers.append(meta["tier"])
            alphas.append(meta["alpha"])
            graph_dists.append(meta["graph_dist"])
            zs.append(z)
            sample_ids.append(np.full(hi - lo, sample_counter, dtype=np.int64))
            sample_counter += 1

            if sample_counter >= args.n_samples:
                break
        if sample_counter >= args.n_samples:
            break

    lw = np.concatenate(learned_weights)
    pw = np.concatenate(prior_weights)
    tier = np.concatenate(tiers)
    alpha = np.concatenate(alphas)
    gd = np.concatenate(graph_dists)
    z_arr = np.concatenate(zs)
    sid = np.concatenate(sample_ids)
    residual = lw - pw

    # --- CSV dump ----------------------------------------------------------
    csv_path = Path(args.out).with_suffix("")
    csv_path = csv_path.parent / (csv_path.name + "_per_atom.csv")
    with open(csv_path, "w") as f:
        f.write("sample_id,atom_idx,z,tier,graph_dist,alpha,prior_w,learned_w,residual\n")
        for i in range(len(lw)):
            f.write(
                f"{sid[i]},{i},{z_arr[i]},{tier[i]},{gd[i]},"
                f"{alpha[i]:.6f},{pw[i]:.6f},{lw[i]:.6f},{residual[i]:.6f}\n"
            )
    print(f"Wrote per-atom CSV → {csv_path}")

    # --- plots -------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(pw, lw, s=8, alpha=0.4)
    lo, hi = min(pw.min(), lw.min()), max(pw.max(), lw.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel("BC prior weight")
    ax.set_ylabel("learned weight")
    ax.set_title("Learned vs BC prior")
    ax.legend()

    ax = axes[0, 1]
    tier_groups = [lw[tier == t] for t in (1, 2, 3)]
    ax.boxplot(tier_groups, labels=[_TIER_LABEL[t] for t in (1, 2, 3)])
    ax.set_ylabel("learned weight")
    ax.set_title("Learned weight by tier")

    ax = axes[1, 0]
    unique_z = sorted(np.unique(z_arr).tolist())
    z_groups = [lw[z_arr == z] for z in unique_z]
    ax.boxplot(z_groups, labels=[str(z) for z in unique_z])
    ax.set_xlabel("atomic number Z")
    ax.set_ylabel("learned weight")
    ax.set_title("Learned weight by element")

    ax = axes[1, 1]
    ax.scatter(gd, residual, s=8, alpha=0.4)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("graph distance to reactive core")
    ax.set_ylabel("learned - prior")
    ax.set_title("D residual vs graph distance")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote decomposition plot → {args.out}")


if __name__ == "__main__":
    main()
