"""Stochastic initial-guess ensemble utilities for Idea 2-E (v2).

Public surface (what `predict_ts_ensemble` and `evaluate_ensemble.py` import):

    generate_stochastic_x0_ensemble  - per-atom Gaussian perturbation of a base x0
    select_best_from_ensemble        - medoid / midpoint-distance candidate selection
    ensemble_diversity_stats         - pairwise Kabsch-RMSD distribution
    kabsch_rmsd                      - rotation-invariant pairwise RMSD

Fallback helpers (needed because branch rp-E is cut from reactot-halo8, which
does NOT carry the Idea 1-A weighting or the Idea 2-A IDPP+clash code; the
markdown calls those out as soft dependencies and asks for a graceful path):

    compute_x0_base_or_midpoint      - returns IDPP if Idea 2-A present, else (R+P)/2
    compute_atom_weights_or_fallback - returns Idea 1-A hybrid weights if present,
                                       else a displacement-magnitude-based proxy

The vdW clash correction in `_apply_clash_correction` uses Bondi (1964) radii
extended with Mantina et al. (2009) for halogens & sulfur to cover the Halo8
elements (H/C/N/O/F/S/Cl/Br) on top of the Transition1x set (H/C/N/O).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Bondi (1964) + Mantina (2009) van der Waals radii in Angstrom.
# Covers everything the Halo8 atom mapping uses (H/C/N/O/F/S/Cl/Br) plus
# I as a future-proofing entry. Unknown Z falls through to a default 1.5 A.
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}


def _apply_clash_correction(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    f_scale: float = 0.85,
    max_iter: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Push apart atom pairs whose distance is below ``f_scale * (r_i + r_j)``.

    Symmetric pair-wise relaxation: each colliding pair is shifted apart in
    equal halves along the inter-atomic axis. Iterates up to ``max_iter`` or
    until no further moves are needed (whichever comes first). Pure NumPy so
    it stays cheap inside a Python ensemble loop.
    """
    coords = coords.copy().astype(np.float64)
    n = len(atomic_numbers)
    radii = np.array(
        [VDW_RADII.get(int(z), 1.5) for z in atomic_numbers], dtype=np.float64
    )
    for _ in range(max_iter):
        moved = False
        # Diff matrix (N, N, 3); upper triangle is enough.
        diffs = coords[None, :, :] - coords[:, None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        thresholds = f_scale * (radii[:, None] + radii[None, :])
        np.fill_diagonal(dists, np.inf)
        clash_mask = dists < thresholds
        if not clash_mask.any():
            break
        # Take only the upper triangle so each pair is processed once.
        iu, ju = np.triu_indices(n, k=1)
        for i, j in zip(iu, ju):
            d = dists[i, j]
            if d >= thresholds[i, j] or d <= eps:
                continue
            push = (thresholds[i, j] - d) / 2.0
            direction = (coords[j] - coords[i]) / d
            coords[j] += push * direction
            coords[i] -= push * direction
            moved = True
        if not moved:
            break
    return coords.astype(np.float32)


def generate_stochastic_x0_ensemble(
    x0_base: np.ndarray,
    atomic_numbers: np.ndarray,
    atom_weights: np.ndarray,
    K: int = 5,
    sigma_base: float = 0.1,
    seed: Optional[int] = None,
    apply_clash_check: bool = True,
    core_weight_clip: float = 0.9,
) -> np.ndarray:
    """Generate K candidate initial structures via per-atom Gaussian perturbation.

    Perturbation scale per atom:

        sigma_i = sigma_base * max(0, 1 - w_i)

    Atoms with ``w_i > core_weight_clip`` are clipped to ``sigma_i = 0`` so the
    reaction core is never displaced. Index k=0 is always the un-perturbed
    base structure (so the ensemble always contains the deterministic anchor).

    Args:
        x0_base: (N, 3) base initial structure (e.g. IDPP or midpoint).
        atomic_numbers: (N,) integer atomic numbers (used for clash check).
        atom_weights: (N,) continuous weights in [0, 1]; core ~ 1.0,
            peripheral ~ w_min.
        K: ensemble size; k=0 always equals the input base.
        sigma_base: base perturbation scale, Angstrom.
        seed: random seed for reproducibility.
        apply_clash_check: run vdW clash correction on each perturbed candidate.
        core_weight_clip: atoms with weight strictly greater than this value
            get sigma = 0 (reaction-coordinate preservation).

    Returns:
        x0_ensemble: (K, N, 3) float32 array of K candidate initial structures.
    """
    x0_base = np.asarray(x0_base, dtype=np.float32)
    atomic_numbers = np.asarray(atomic_numbers)
    atom_weights = np.asarray(atom_weights, dtype=np.float64)

    N = x0_base.shape[0]
    rng = np.random.RandomState(seed)

    sigma_per_atom = sigma_base * np.clip(1.0 - atom_weights, 0.0, None)
    sigma_per_atom = np.where(atom_weights > core_weight_clip, 0.0, sigma_per_atom)

    x0_ensemble = np.zeros((K, N, 3), dtype=np.float32)

    for k in range(K):
        if k == 0:
            x0_ensemble[0] = x0_base.copy()
            continue

        noise = rng.randn(N, 3)
        scaled_noise = noise * sigma_per_atom[:, None]
        candidate = x0_base + scaled_noise.astype(np.float32)

        if apply_clash_check:
            candidate = _apply_clash_correction(candidate, atomic_numbers)

        candidate = candidate - candidate.mean(axis=0, keepdims=True)
        x0_ensemble[k] = candidate.astype(np.float32)

    return x0_ensemble


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two (N, 3) structures.

    Computes the optimal rigid alignment via SVD, then returns the
    rotation/translation-invariant RMSD. Same molecule under any rigid
    transform yields ~0.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_aligned = P_c @ R.T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q_c) ** 2, axis=1))))


def select_best_from_ensemble(
    ts_candidates: np.ndarray,
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    method: str = "rmsd_consensus",
) -> Tuple[np.ndarray, int]:
    """Pick the "best" TS from K candidates (Kabsch-aligned RMSD throughout).

    Methods:
        ``midpoint_distance``: pick the candidate whose Kabsch-RMSD to
            ``(R+P)/2`` is closest to the median of all such distances --
            heuristic that the true TS is usually near (but slightly off)
            the midpoint.
        ``rmsd_consensus``: the medoid of the ensemble; the candidate whose
            mean pairwise Kabsch-RMSD to all other candidates is minimal.
            Recovers a "typical" TS without needing a confidence model.
    """
    K = ts_candidates.shape[0]

    if method == "midpoint_distance":
        midpoint = 0.5 * (np.asarray(pos_R) + np.asarray(pos_P))
        distances = np.array(
            [kabsch_rmsd(ts_candidates[k], midpoint) for k in range(K)]
        )
        median_dist = np.median(distances)
        best_idx = int(np.argmin(np.abs(distances - median_dist)))

    elif method == "rmsd_consensus":
        avg_rmsd = np.zeros(K)
        for k in range(K):
            rmsds = []
            for j in range(K):
                if j == k:
                    continue
                rmsds.append(kabsch_rmsd(ts_candidates[k], ts_candidates[j]))
            avg_rmsd[k] = float(np.mean(rmsds)) if rmsds else 0.0
        best_idx = int(np.argmin(avg_rmsd))

    else:
        raise ValueError(f"Unknown selection method: {method}")

    return ts_candidates[best_idx], best_idx


def ensemble_diversity_stats(ts_candidates: np.ndarray) -> dict:
    """Pairwise Kabsch-RMSD distribution stats (mean / std / min / max)."""
    K = ts_candidates.shape[0]
    pairwise = []
    for i in range(K):
        for j in range(i + 1, K):
            pairwise.append(kabsch_rmsd(ts_candidates[i], ts_candidates[j]))
    if not pairwise:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(pairwise)),
        "std": float(np.std(pairwise)),
        "min": float(np.min(pairwise)),
        "max": float(np.max(pairwise)),
    }


# ---------------------------------------------------------------------------
# Soft-dependency fallbacks. Idea 2-E hooks into Idea 1-A (atom weighting) and
# Idea 2-A (IDPP + clash) when those branches are merged, but reactot-halo8 by
# itself does not carry that code. The lazy try/except imports below let the
# pipeline degrade gracefully on a vanilla halo8 install.
# ---------------------------------------------------------------------------


def compute_x0_base_or_midpoint(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    """Return IDPP-optimised x0 if Idea 2-A is available, else the midpoint.

    Tries to import ``compute_idpp`` (added by branch rp-A as part of Idea 2-A).
    On ImportError or any internal failure, falls back silently to (R+P)/2 --
    the original ReactOT default -- so the ensemble pipeline stays usable on
    branches that don't yet have IDPP.
    """
    pos_R = np.asarray(pos_R, dtype=np.float32)
    pos_P = np.asarray(pos_P, dtype=np.float32)
    try:  # Idea 2-A (rp-A) entry point
        from reactot.utils.initial_guess import compute_idpp  # type: ignore

        # NB: compute_idpp lives in this same module on rp-A. The recursive
        # import resolves only if this module was overwritten by rp-A's
        # version; on rp-E (no IDPP) the import goes through to this very
        # module and compute_idpp is absent, raising AttributeError below.
        return compute_idpp(pos_R, pos_P, atomic_numbers).astype(np.float32)
    except (ImportError, AttributeError):
        pass
    except Exception as exc:  # pragma: no cover -- diagnostic path
        import warnings

        warnings.warn(
            f"compute_idpp failed ({exc!r}); falling back to (R+P)/2."
        )
    return (0.5 * (pos_R + pos_P)).astype(np.float32)


def compute_atom_weights_or_fallback(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    displacement_thr: float = 0.5,
) -> np.ndarray:
    """Return Idea 1-A hybrid weights if available, else a displacement proxy.

    Fallback: normalise the per-atom |R - P| displacement, then map it to
    [w_min, 1.0]. Atoms whose positions barely change between R and P get
    the floor weight ``w_min`` (peripheral); atoms with the largest
    displacement get 1.0 (core / reactive). This is a reasonable proxy for
    "how central is this atom in the reaction" without needing a bond graph
    or hop-distance calculation.
    """
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)

    try:  # Idea 1-A (cb-A / rp-A) entry point
        from reactot.utils.weighting import compute_hybrid_weights  # type: ignore

        return np.asarray(
            compute_hybrid_weights(pos_R, pos_P, atomic_numbers), dtype=np.float64
        )
    except (ImportError, AttributeError):
        pass

    displacements = np.linalg.norm(pos_P - pos_R, axis=1)
    if displacements.max() < displacement_thr:
        # Whole molecule moved less than displacement_thr — treat all atoms
        # as peripheral so the perturbation kicks in everywhere.
        return np.full(len(atomic_numbers), w_min, dtype=np.float64)

    norm = displacements / max(displacements.max(), 1e-9)
    return (w_min + (1.0 - w_min) * norm).astype(np.float64)
