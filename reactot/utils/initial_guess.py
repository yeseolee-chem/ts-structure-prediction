"""
OT-FM 초기 구조(x_0) 생성 모듈 — Combo [FE] (Learned x_0 + Stochastic ensemble).

Pipeline: (R, P) → EGNN x_0 predictor → K stochastic candidates.

References:
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
- Halo8: Lee et al., Sci. Data, 2025, DOI: 10.1038/s41597-025-05944-3
"""
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple
from collections import deque

import torch


# ---------------------------------------------------------------------------
# vdW + halogen-aware clash threshold (shared)
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
# Idea 2-F v2: Learned x_0 predictor (with process-level model cache)
# ---------------------------------------------------------------------------

# Halo8 mapping: H/C/N/O/F/S=Cl/Br ordering used during training. Update
# this if the project uses a different mapping.
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 5, 35: 6}


_X0_PREDICTOR_CACHE: dict = {}


def _get_or_load_x0_predictor(model_checkpoint: str, device: str = 'cpu'):
    """Process-level cache for the EGNN x_0 predictor."""
    key = (model_checkpoint, device)
    if key in _X0_PREDICTOR_CACHE:
        return _X0_PREDICTOR_CACHE[key]

    from reactot.model.x0_predictor import X0PredictorEGNN
    model = X0PredictorEGNN(n_atom_types=7, hidden_dim=64, n_layers=3)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    _X0_PREDICTOR_CACHE[key] = model
    return model


def compute_learned_x0(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    model_checkpoint: str,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Predict x_0 with the learned EGNN model.
    Process-level cached so repeated calls don't re-load the checkpoint.
    """
    model = _get_or_load_x0_predictor(model_checkpoint, device)

    atom_types = np.array(
        [HALO_ATOM_MAPPING.get(int(z), 0) for z in atomic_numbers]
    )

    pos_R_t = torch.tensor(pos_R, dtype=torch.float32, device=device)
    pos_P_t = torch.tensor(pos_P, dtype=torch.float32, device=device)
    atom_types_t = torch.tensor(atom_types, dtype=torch.long, device=device)

    with torch.no_grad():
        x0_pred = model(pos_R_t, pos_P_t, atom_types_t)

    return x0_pred.cpu().numpy()


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
# Combo [FE] wrapper
# ---------------------------------------------------------------------------

def compute_x0_FE(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    learned_x0_checkpoint: str,
    K: int = 5,
    sigma_base: float = 0.1,
    seed: Optional[int] = None,
    device: str = 'cpu',
    atom_weights: Optional[np.ndarray] = None,
    core_weight_clip: float = 0.9,
) -> np.ndarray:
    """
    Combo [FE] Stage 1+2: Learned base + K stochastic candidates.

    F의 핵심 한계는 OOD 일반화 약점. Ensemble의 perturbation이 OOD 영역에서
    multiple modes를 탐색하여 robustness ↑.
    """
    x0_base = compute_learned_x0(
        pos_R, pos_P, atomic_numbers,
        model_checkpoint=learned_x0_checkpoint,
        device=device,
    )

    if atom_weights is None:
        atom_weights = _fallback_hop_weights(pos_R, pos_P, atomic_numbers)

    return generate_stochastic_x0_ensemble(
        x0_base, atomic_numbers, atom_weights,
        K=K, sigma_base=sigma_base, seed=seed,
        apply_clash_check=True, core_weight_clip=core_weight_clip,
    )
