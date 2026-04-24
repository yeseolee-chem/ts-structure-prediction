"""Idea 1-CD diagnostic: tier-vs-learned-weight check.

After a CD training run (``prior_scheme='C'``, ``store_tier=True``,
``learn_importance=True``), collect the per-atom learned weights on a
validation loader and bin them by the C-prior tier label.

Hypothesis: for the KL regularization to be doing its job, the average
learned weight should satisfy  Core >= Interface >= Peripheral. A Core/
Peripheral ratio near the prior ratio (~ 1 / w_min ≈ 10x by default)
means D is respecting the hierarchy; a flat distribution means the KL
term is too weak (or the importance head collapsed).

Usage:
    python scripts/analyze_cd_tiers.py \\
        --checkpoint PATH/to/sb.ckpt \\
        --output figures/cd_tier_analysis.png \\
        [--dataset Halo8 --data-dir PATH] [--max-batches 50]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from reactot.trainer.pl_trainer import SBModule


def collect_learned_by_tier(
    model: SBModule,
    dataloader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[int, np.ndarray]:
    """Run the dynamics forward to get per-atom learned weights, bin by tier.

    Returns a dict ``{1: core_weights, 2: interface_weights, 3: peripheral_weights}``
    where each value is a flat numpy array across the batches seen.
    """
    learned_by_tier: Dict[int, List[np.ndarray]] = {1: [], 2: [], 3: []}

    model.eval()
    w_min = float(getattr(model.ddpm, "learned_w_min", 0.1))
    target_idx = int(model.ddpm.idx)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            representations, conditions = batch
            # Move batches to device — the vanilla collate_fn keeps tensors on
            # the dataset's device, which is "cpu" during eval dataloading.
            for rep in representations:
                for k, v in rep.items():
                    if torch.is_tensor(v):
                        rep[k] = v.to(device)
            if torch.is_tensor(conditions):
                conditions = conditions.to(device)

            target_rep = representations[target_idx]
            if "atom_tier" not in target_rep:
                raise KeyError(
                    "atom_tier missing from the target fragment. Re-run the "
                    "dataset with graph_weights_store_tier=True."
                )

            # The SB forward produces importance per fragment via the dynamics.
            # We short-circuit to the dynamics directly here because we only
            # need the raw logits — not a loss.
            from reactot.utils import get_edges_index, get_n_frag_switch
            from reactot.diffusion.en_sb import FEATURE_MAPPING  # type: ignore

            masks = [r["mask"] for r in representations]
            combined_mask = torch.cat(masks)
            edge_index = get_edges_index(combined_mask, remove_self_edge=True)
            fragments_nodes = [r["size"] for r in representations]
            n_frag_switch = get_n_frag_switch(fragments_nodes)

            normalized = model.ddpm.normalizer.normalize(representations)
            xh_t = [
                torch.cat(
                    [rep[feat] for feat in FEATURE_MAPPING],
                    dim=1,
                )
                for rep in normalized
            ]
            t = torch.zeros(
                (normalized[0]["size"].size(0), 1),
                device=device,
                dtype=torch.float32,
            )
            cond = conditions["condition"] if model.ddpm.ts_guess else conditions
            _, _, importance_per_frag = model.ddpm.dynamics(
                xh=xh_t,
                edge_index=edge_index,
                t=t,
                conditions=cond,
                n_frag_switch=n_frag_switch,
                combined_mask=combined_mask,
                edge_attr=None,
            )
            if importance_per_frag is None:
                raise RuntimeError(
                    "Dynamics has no importance head — this checkpoint was "
                    "not trained with learn_importance=True."
                )

            raw = importance_per_frag[target_idx]
            learned = (torch.sigmoid(raw) * (1.0 - w_min) + w_min).cpu().numpy()
            tier = target_rep["atom_tier"].cpu().numpy()

            for t_val in (1, 2, 3):
                mask = tier == t_val
                if mask.any():
                    learned_by_tier[t_val].append(learned[mask])

    return {t: (np.concatenate(v) if v else np.empty(0)) for t, v in learned_by_tier.items()}


def plot_tier_distribution(
    learned_by_tier: Dict[int, np.ndarray],
    save_path: str | os.PathLike,
) -> None:
    """Tier violin plot + sanity assertions."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Core", "Interface", "Peripheral"]
    data = [learned_by_tier[t] for t in (1, 2, 3)]
    non_empty = [d for d in data if d.size > 0]
    if len(non_empty) < 3:
        print("Warning: one or more tiers have no samples in this dataloader.")

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(d.size > 0 for d in data):
        ax.violinplot(
            [d if d.size > 0 else np.array([0.0]) for d in data],
            showmeans=True,
            showmedians=True,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Learned weight")
    ax.set_title("CD: learned-weight distribution by C-prior tier")

    for i, d in enumerate(data, 1):
        if d.size > 0:
            ax.text(i, d.mean() + 0.05, f"{d.mean():.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Sanity checks — these are *hypothesis checks*, not hard failures.
    if data[0].size > 0 and data[2].size > 0:
        core_mean = data[0].mean()
        peripheral_mean = data[2].mean()
        if core_mean > peripheral_mean:
            print(
                f"  [OK] Core mean {core_mean:.3f} > Peripheral mean "
                f"{peripheral_mean:.3f} — D is honoring the CD hierarchy."
            )
        else:
            print(
                f"  [WARN] Core mean {core_mean:.3f} <= Peripheral mean "
                f"{peripheral_mean:.3f}. Either KL weight is too low or the "
                f"importance head collapsed."
            )
    if data[1].size > 0 and data[2].size > 0:
        if data[1].mean() < data[2].mean():
            print(
                f"  [WARN] Interface mean {data[1].mean():.3f} < Peripheral "
                f"mean {data[2].mean():.3f} — consider raising KL_WEIGHT."
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to SBModule *.ckpt.")
    p.add_argument(
        "--output",
        default="figures/cd_tier_analysis.png",
        help="Where to write the violin plot.",
    )
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    model = SBModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model.setup(stage="fit")

    learned_by_tier = collect_learned_by_tier(
        model,
        model.val_loader_no_swap,
        device=device,
        max_batches=args.max_batches,
    )
    for t, v in learned_by_tier.items():
        name = {1: "Core", 2: "Interface", 3: "Peripheral"}[t]
        print(f"  {name:<10s}: n={v.size:>6d}  mean={v.mean():.3f}  std={v.std():.3f}"
              if v.size > 0
              else f"  {name:<10s}: empty")

    plot_tier_distribution(learned_by_tier, args.output)


if __name__ == "__main__":
    main()
