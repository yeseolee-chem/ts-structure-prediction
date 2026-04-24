"""Visualize the per-atom importance weights a cb-D checkpoint has learned.

Panel 1: hop-from-core vs mean weight — learned vs. prior (Idea 1-A).
Panel 2: per-element mean weight — learned vs. prior.
Panel 3: scatter of learned vs. prior, colored by hop distance.

Usage:
    python scripts/visualize_learned_weights.py \
        --checkpoint checkpoint/.../sb-xxx.ckpt \
        --dataset TS1x \
        --datadir reactot/data/transition1x/ \
        --output figures/learned_weights.png
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reactot.trainer.pl_trainer import SBModule, PROCESS_FUNC, FILE_TYPE
from reactot.utils.weighting import (
    build_adjacency_matrix,
    find_reactive_core_from_positions,
    graph_distances_to_core,
)


ELEMENT_LABELS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}


def load_checkpoint(ckpt_path: str, device: str) -> SBModule:
    module = SBModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    if not getattr(module.ddpm.dynamics, "learn_importance", False):
        raise RuntimeError(
            "Checkpoint was trained without learn_importance=True; nothing to visualize."
        )
    return module


def _make_loader(module: SBModule, split: str, datadir: str, dataset: str, bz: int):
    func = PROCESS_FUNC[dataset]
    ft = FILE_TYPE[dataset]
    training_config = dict(module.training_config)
    training_config["datadir"] = datadir

    if dataset == "Halo8":
        path = Path(datadir)
    else:
        path = Path(datadir, f"{split}_rpsb_all{ft}")

    ds = func(path, **training_config)
    return DataLoader(ds, batch_size=bz, shuffle=False, collate_fn=ds.collate_fn)


@torch.no_grad()
def extract_learned(module: SBModule, loader, max_batches: int, device: str):
    learned, priors, hops, elements = [], [], [], []
    w_min = float(module.ddpm.learned_w_min)

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        representations, _ = batch
        representations = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in rep.items()}
            for rep in representations
        ]

        # Run through the dynamics to get raw importance logits.
        masks = [r["mask"] for r in representations]
        combined_mask = torch.cat(masks)
        from reactot.utils import get_edges_index, get_n_frag_switch
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [r["size"] for r in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        normalizer = module.ddpm.normalizer
        reprs_norm = normalizer.normalize(representations)
        from reactot.diffusion._normalizer import FEATURE_MAPPING
        xh_t = [
            torch.cat([rep[k] for k in FEATURE_MAPPING], dim=1)
            for rep in reprs_norm
        ]
        t = torch.zeros(fragments_nodes[0].shape[0], 1, device=device)

        _, _, imp = module.ddpm.dynamics(
            xh=xh_t, edge_index=edge_index, t=t,
            conditions=torch.zeros_like(combined_mask).unsqueeze(-1),
            n_frag_switch=n_frag_switch, combined_mask=combined_mask, edge_attr=None,
        )
        if imp is None:
            raise RuntimeError("Dynamics returned no importance — check learn_importance.")

        target_imp = imp[module.ddpm.idx]
        target_w = torch.sigmoid(target_imp) * (1.0 - w_min) + w_min

        target_rep = representations[module.ddpm.idx]
        prior = target_rep.get("atom_weights", None)

        pos_R_concat = representations[0]["pos"].cpu().numpy()
        pos_P_concat = representations[2]["pos"].cpu().numpy()
        charge_concat = representations[0]["charge"].cpu().numpy().reshape(-1)
        mask_concat = representations[0]["mask"].cpu().numpy()

        cursor = 0
        for sample_idx in np.unique(mask_concat):
            sel = np.where(mask_concat == sample_idx)[0]
            pos_R = pos_R_concat[sel]
            pos_P = pos_P_concat[sel]
            zs = charge_concat[sel].astype(np.int64)

            core = find_reactive_core_from_positions(pos_R, pos_P, zs)
            adj = build_adjacency_matrix(pos_R, zs)
            hop = graph_distances_to_core(adj, core)

            n = len(sel)
            w_slice = target_w[cursor:cursor + n].detach().cpu().numpy()
            p_slice = (
                prior[cursor:cursor + n].detach().cpu().numpy()
                if prior is not None
                else np.full(n, np.nan)
            )
            cursor += n

            learned.extend(w_slice.tolist())
            priors.extend(p_slice.tolist())
            hops.extend(hop.astype(int).tolist())
            elements.extend(zs.tolist())

    return (
        np.array(learned), np.array(priors),
        np.array(hops), np.array(elements),
    )


def plot(learned, priors, hops, elements, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    unique_hops = sorted(set(int(h) for h in hops))
    learned_by_hop = [learned[hops == h].mean() for h in unique_hops]
    prior_by_hop = [priors[hops == h].mean() for h in unique_hops]
    axes[0].plot(unique_hops, learned_by_hop, "o-", label="learned")
    axes[0].plot(unique_hops, prior_by_hop, "s--", label="prior (Idea 1-A)")
    axes[0].set_xlabel("Graph distance to reactive core")
    axes[0].set_ylabel("Mean weight")
    axes[0].set_title("Weight vs. hop")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    unique_el = sorted(set(int(e) for e in elements))
    labels = [ELEMENT_LABELS.get(e, str(e)) for e in unique_el]
    learned_by_el = [learned[elements == e].mean() for e in unique_el]
    prior_by_el = [priors[elements == e].mean() for e in unique_el]
    x = np.arange(len(unique_el))
    axes[1].bar(x - 0.2, learned_by_el, width=0.4, label="learned")
    axes[1].bar(x + 0.2, prior_by_el, width=0.4, label="prior")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean weight")
    axes[1].set_title("Weight by element")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    sc = axes[2].scatter(priors, learned, c=hops, s=8, alpha=0.5, cmap="viridis")
    lim = [0, 1.05]
    axes[2].plot(lim, lim, "k--", alpha=0.5)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel("Prior weight")
    axes[2].set_ylabel("Learned weight")
    axes[2].set_title("Learned vs. prior")
    plt.colorbar(sc, ax=axes[2], label="hop")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="TS1x", choices=["TS1x", "Halo8"])
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--output", default="figures/learned_weights.png")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--bz", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    module = load_checkpoint(args.checkpoint, args.device).to(args.device)
    loader = _make_loader(module, args.split, args.datadir, args.dataset, args.bz)
    learned, priors, hops, elements = extract_learned(
        module, loader, args.max_batches, args.device
    )
    plot(learned, priors, hops, elements, args.output)


if __name__ == "__main__":
    main()
