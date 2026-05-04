"""
OT-FM 초기 구조(x_0) 생성 모듈 — Combo [CE] (vdW midpoint + Stochastic ensemble).

Pipeline: (R, P) → (R+P)/2 + vdW repulsion → K stochastic candidates.
가장 가벼운 ensemble baseline (base가 < 10ms).

References:
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
- Halo8: Lee et al., Sci. Data, 2025, DOI: 10.1038/s41597-025-05944-3
"""
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Idea 2-A v2 (def): vdW + halogen-aware clash threshold (shared with C)
# ---------------------------------------------------------------------------

VDW_RADII = {
    1:  1.20, 6:  1.70, 7:  1.55, 8:  1.52, 9:  1.47,
    16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}

DEFAULT_CLASH_SCALES = {
    'default': 0.70,
    'halogen_halogen': 0.80,
    'halogen_H': 0.65,
    'halogen_heavy': 0.75,
}

HALOGENS = {9, 17, 35, 53}


def is_halogen(z: int) -> bool:
    return int(z) in HALOGENS


def get_clash_threshold(z_i: int, z_j: int,
                        clash_scales: Optional[dict] = None) -> float:
    scales = clash_scales or DEFAULT_CLASH_SCALES
    sigma = VDW_RADII.get(int(z_i), 1.70) + VDW_RADII.get(int(z_j), 1.70)
    is_h_i, is_h_j = is_halogen(z_i), is_halogen(z_j)
    is_proton_i = (int(z_i) == 1)
    is_proton_j = (int(z_j) == 1)
    if is_h_i and is_h_j:
        f_scale = scales.get('halogen_halogen', 0.80)
    elif (is_h_i and is_proton_j) or (is_proton_i and is_h_j):
        f_scale = scales.get('halogen_H', 0.65)
    elif is_h_i or is_h_j:
        f_scale = scales.get('halogen_heavy', 0.75)
    else:
        f_scale = scales.get('default', 0.70)
    return f_scale * sigma


# ---------------------------------------------------------------------------
# Idea 2-C v2: vdW corrected midpoint
# ---------------------------------------------------------------------------

def compute_vdw_corrected_midpoint(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    n_iter: int = 50,
    recovery_rate: float = 1.0,
    clash_scales: Optional[dict] = None,
    rng_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    (R+P)/2 + iterative soft-sphere repulsion.

    v2 변경: ``recovery_rate`` semantics — one step의 violation 회복 비율.
    """
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)
    N = pos_R.shape[0]
    x0 = 0.5 * (pos_R + pos_P).copy()
    rng = np.random.RandomState(rng_seed)

    r_clash_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            r_clash_matrix[i, j] = get_clash_threshold(
                int(atomic_numbers[i]), int(atomic_numbers[j]),
                clash_scales,
            )
            r_clash_matrix[j, i] = r_clash_matrix[i, j]

    for _ in range(n_iter):
        has_clash = False
        for i in range(N):
            for j in range(i + 1, N):
                diff = x0[i] - x0[j]
                r_ij = float(np.linalg.norm(diff))
                r_min = r_clash_matrix[i, j]
                if r_ij < 1e-8:
                    direction = rng.randn(3)
                    direction /= np.linalg.norm(direction) + 1e-12
                    half = 0.5 * r_min
                    x0[i] += half * direction
                    x0[j] -= half * direction
                    has_clash = True
                    continue
                if r_ij < r_min:
                    has_clash = True
                    direction = diff / r_ij
                    violation = r_min - r_ij
                    half_push = 0.5 * recovery_rate * violation
                    x0[i] += half_push * direction
                    x0[j] -= half_push * direction
        if not has_clash:
            break

    x0 -= x0.mean(axis=0)
    return x0


def count_clashes(positions: np.ndarray, atomic_numbers: np.ndarray,
                  clash_scales: Optional[dict] = None) -> dict:
    """Diagnostic: clash count + worst violation."""
    N = positions.shape[0]
    clash_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = float(np.linalg.norm(positions[i] - positions[j]))
            r_clash = get_clash_threshold(
                int(atomic_numbers[i]), int(atomic_numbers[j]),
                clash_scales,
            )
            if r_ij < r_clash:
                clash_pairs.append((i, j, r_ij, r_clash))
    violations = [r_clash - r_ij for _, _, r_ij, r_clash in clash_pairs]
    return {
        'n_clashes': len(clash_pairs),
        'worst_violation': max(violations) if violations else 0.0,
        'clash_pairs': clash_pairs,
    }


# ---------------------------------------------------------------------------
# Shared clash correction (used by ensemble post-perturbation)
# ---------------------------------------------------------------------------

def _apply_clash_correction(x0, atomic_numbers, n_iter=50):
    x0 = np.asarray(x0, dtype=np.float64)
    N = len(x0)
    for _ in range(n_iter):
        has_clash = False
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = float(np.linalg.norm(x0[i] - x0[j]))
                r_min = get_clash_threshold(
                    int(atomic_numbers[i]), int(atomic_numbers[j])
                )
                if r_ij < r_min and r_ij > 1e-6:
                    has_clash = True
                    direction = (x0[i] - x0[j]) / r_ij
                    push = 0.5 * (r_min - r_ij)
                    x0[i] += push * direction
                    x0[j] -= push * direction
        if not has_clash:
            break
    return x0


# ---------------------------------------------------------------------------
# Idea 2-E v2: Stochastic ensemble
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
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
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q_c)**2, axis=1))))


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
    x0_base = np.asarray(x0_base, dtype=np.float64)
    atom_weights = np.asarray(atom_weights, dtype=np.float64)
    N = x0_base.shape[0]
    rng = np.random.RandomState(seed)

    sigma_per_atom = sigma_base * np.clip(1.0 - atom_weights, 0.0, None)
    sigma_per_atom = np.where(atom_weights > core_weight_clip, 0.0, sigma_per_atom)

    x0_ensemble = np.zeros((K, N, 3))
    for k in range(K):
        if k == 0:
            x0_ensemble[0] = x0_base.copy()
            continue
        noise = rng.randn(N, 3) * sigma_per_atom[:, None]
        candidate = x0_base + noise
        if apply_clash_check:
            candidate = _apply_clash_correction(candidate, atomic_numbers)
        candidate -= candidate.mean(axis=0)
        x0_ensemble[k] = candidate
    return x0_ensemble


def select_best_from_ensemble(
    ts_candidates: np.ndarray,
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    method: str = 'rmsd_consensus',
) -> Tuple[np.ndarray, int]:
    K = ts_candidates.shape[0]
    if method == 'midpoint_distance':
        midpoint = 0.5 * (pos_R + pos_P)
        distances = np.array([
            kabsch_rmsd(ts_candidates[k], midpoint) for k in range(K)
        ])
        median_dist = np.median(distances)
        best_idx = int(np.argmin(np.abs(distances - median_dist)))
    elif method == 'rmsd_consensus':
        avg_rmsd = np.zeros(K)
        for k in range(K):
            rmsds = [kabsch_rmsd(ts_candidates[k], ts_candidates[j])
                     for j in range(K) if j != k]
            avg_rmsd[k] = float(np.mean(rmsds)) if rmsds else 0.0
        best_idx = int(np.argmin(avg_rmsd))
    else:
        raise ValueError(f"Unknown selection method: {method}")
    return ts_candidates[best_idx], best_idx


def ensemble_diversity_stats(ts_candidates: np.ndarray) -> dict:
    K = ts_candidates.shape[0]
    pairwise = []
    for i in range(K):
        for j in range(i + 1, K):
            pairwise.append(kabsch_rmsd(ts_candidates[i], ts_candidates[j]))
    if not pairwise:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(pairwise)),
        'std':  float(np.std(pairwise)),
        'min':  float(np.min(pairwise)),
        'max':  float(np.max(pairwise)),
    }


def _fallback_hop_weights(pos_R, pos_P, atomic_numbers,
                          w_min: float = 0.1, decay: float = 0.7):
    N = len(atomic_numbers)
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)
    d_R = cdist(pos_R, pos_R)
    d_P = cdist(pos_P, pos_P)
    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39}

    core = set()
    for i in range(N):
        for j in range(i + 1, N):
            r_cov = (covalent_radii.get(int(atomic_numbers[i]), 0.77) +
                     covalent_radii.get(int(atomic_numbers[j]), 0.77))
            in_bond = (d_R[i, j] < r_cov * 1.3) or (d_P[i, j] < r_cov * 1.3)
            if in_bond and abs(d_R[i, j] - d_P[i, j]) > 0.5:
                core.add(i); core.add(j)

    weights = np.full(N, w_min, dtype=np.float64)
    for c in core:
        weights[c] = 1.0
    if not core:
        return weights

    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            r_cov = (covalent_radii.get(int(atomic_numbers[i]), 0.77) +
                     covalent_radii.get(int(atomic_numbers[j]), 0.77))
            if d_R[i, j] < r_cov * 1.3:
                adj[i].append(j); adj[j].append(i)

    hop = {c: 0 for c in core}
    q = deque(core)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in hop:
                hop[v] = hop[u] + 1
                q.append(v)

    for i in range(N):
        h = hop.get(i, 99)
        weights[i] = max(w_min, decay ** h)
    return weights


# ---------------------------------------------------------------------------
# Combo [CE] wrapper
# ---------------------------------------------------------------------------

def compute_x0_CE(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    K: int = 5,
    sigma_base: float = 0.1,
    seed: Optional[int] = None,
    vdw_kwargs: Optional[dict] = None,
    atom_weights: Optional[np.ndarray] = None,
    core_weight_clip: float = 0.9,
) -> np.ndarray:
    """
    Combo [CE] Stage 1+2: vdW-corrected midpoint + K stochastic candidates.

    매력: base가 < 10ms — ensemble의 효과만 분리해 평가하는 가장 깨끗한
    baseline. 반응당 비용은 K * (ODE solving) 만 추가.
    """
    vdw_kw = vdw_kwargs or {'n_iter': 50, 'recovery_rate': 1.0, 'rng_seed': 42}
    x0_base = compute_vdw_corrected_midpoint(
        pos_R, pos_P, atomic_numbers, **vdw_kw,
    )

    if atom_weights is None:
        atom_weights = _fallback_hop_weights(pos_R, pos_P, atomic_numbers)

    return generate_stochastic_x0_ensemble(
        x0_base, atomic_numbers, atom_weights,
        K=K, sigma_base=sigma_base, seed=seed,
        apply_clash_check=True, core_weight_clip=core_weight_clip,
    )
