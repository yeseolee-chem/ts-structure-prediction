"""
OT-FM 초기 구조(x_0) 생성 모듈 — Combo [BDE] (full pipeline, B-base).

Pipeline: (R, P) → IC interp → xTB short opt → K stochastic candidates.

본 모듈은 idea2 조합 [BDE]를 위해 다음을 통합한다:
- Idea 2-A v2: IDPP (B fallback)
- Idea 2-B v2: Internal coordinate interpolation
- Idea 2-D v2: GFN2-xTB short refinement
- Idea 2-E v2: Stochastic perturbation ensemble
- Combo [BD] / [BE] / [BDE] wrappers

References:
- IDPP: Smidstrup et al., J. Chem. Phys., 2014, DOI: 10.1063/1.4878664
- GSM (IC interp 영감): Zimmerman, JCTC 2013, DOI: 10.1021/ct400319w
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
- GFN2-xTB: Bannwarth, Ehlert, Grimme, JCTC 2019, DOI: 10.1021/acs.jctc.8b01176
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize
from typing import Optional, List, Tuple
from collections import deque

from reactot.utils.internal_coords import (
    get_connectivity, get_angles, get_dihedrals,
    compute_bond_length, compute_angle, compute_dihedral,
    circular_interpolate,
)


# ---------------------------------------------------------------------------
# Shared (Idea 2-A): vdW + halogen-aware clash threshold + IDPP
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


def compute_idpp(pos_R: np.ndarray, pos_P: np.ndarray,
                 atomic_numbers: Optional[np.ndarray] = None,
                 max_iter: int = 200, tol: float = 0.01,
                 lr: float = 0.01,
                 use_clash_penalty: bool = True,
                 clash_kappa: float = 10.0) -> np.ndarray:
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)
    N = pos_R.shape[0]

    d_R = squareform(pdist(pos_R))
    d_P = squareform(pdist(pos_P))
    d_target = 0.5 * (d_R + d_P)

    with np.errstate(divide='ignore'):
        w = np.where(d_target > 1e-6, d_target**(-4), 0.0)
    np.fill_diagonal(w, 0.0)

    r_clash = np.zeros((N, N))
    if use_clash_penalty and atomic_numbers is not None:
        for i in range(N):
            for j in range(i + 1, N):
                r_clash[i, j] = get_clash_threshold(
                    int(atomic_numbers[i]), int(atomic_numbers[j])
                )
                r_clash[j, i] = r_clash[i, j]

    x = 0.5 * (pos_R + pos_P).copy()
    for _ in range(max_iter):
        diff = x[:, None, :] - x[None, :, :]
        d_curr = np.linalg.norm(diff, axis=-1)
        d_curr_safe = d_curr + np.eye(N) * 1e-10
        unit_vec = diff / d_curr_safe[:, :, None]
        residual = d_curr_safe - d_target
        grad_idpp = np.sum(
            2.0 * w[:, :, None] * residual[:, :, None] * unit_vec, axis=1
        )
        grad_clash = np.zeros_like(x)
        if use_clash_penalty and atomic_numbers is not None:
            for i in range(N):
                for j in range(i + 1, N):
                    r_ij = d_curr_safe[i, j]
                    if r_ij < r_clash[i, j]:
                        violation = r_clash[i, j] - r_ij
                        grad_pair = -clash_kappa * violation * unit_vec[i, j]
                        grad_clash[i] += grad_pair
                        grad_clash[j] -= grad_pair
        grad = grad_idpp + grad_clash
        if np.max(np.linalg.norm(grad, axis=1)) < tol:
            break
        x -= lr * grad

    x -= x.mean(axis=0)
    return x


# ---------------------------------------------------------------------------
# Shared clash correction
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
# Idea 2-B v2: IC interpolation
# ---------------------------------------------------------------------------

def _reconstruct_from_ic(x0_init, bonds, bl_target, angles, ang_target,
                         dihedrals, dih_target,
                         max_iter=100,
                         w_bond=1.0, w_angle=0.5, w_dihedral=0.3):
    def objective(x_flat):
        x = x_flat.reshape(-1, 3)
        cost = 0.0
        for idx, (i, j) in enumerate(bonds):
            bl = float(np.linalg.norm(x[i] - x[j]))
            cost += w_bond * (bl - bl_target[idx])**2
        for idx, (i, j, k) in enumerate(angles):
            v1 = x[i] - x[j]
            v2 = x[k] - x[j]
            cos_a = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
            )
            a = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            cost += w_angle * (a - ang_target[idx])**2
        for idx, (i, j, k, l) in enumerate(dihedrals):
            dih = compute_dihedral(x, i, j, k, l)
            diff = dih - dih_target[idx]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            cost += w_dihedral * diff**2
        return cost

    result = minimize(
        objective, np.asarray(x0_init, dtype=np.float64).flatten(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-6}
    )
    return result.x.reshape(-1, 3)


def compute_ic_interpolation(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    alpha: float = 0.5,
    apply_clash_check: bool = True,
    max_lbfgs_iter: int = 100,
    weight_bond: float = 1.0,
    weight_angle: float = 0.5,
    weight_dihedral: float = 0.3,
) -> np.ndarray:
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)

    bonds = get_connectivity(pos_R, atomic_numbers)
    angles = get_angles(bonds)
    dihedrals_list = get_dihedrals(bonds)

    bl_R = [compute_bond_length(pos_R, i, j) for i, j in bonds]
    bl_P = [compute_bond_length(pos_P, i, j) for i, j in bonds]
    ang_R = [compute_angle(pos_R, i, j, k) for i, j, k in angles]
    ang_P = [compute_angle(pos_P, i, j, k) for i, j, k in angles]
    dih_R = [compute_dihedral(pos_R, i, j, k, l) for i, j, k, l in dihedrals_list]
    dih_P = [compute_dihedral(pos_P, i, j, k, l) for i, j, k, l in dihedrals_list]

    bl_mid = [(1 - alpha) * r + alpha * p for r, p in zip(bl_R, bl_P)]
    ang_mid = [(1 - alpha) * r + alpha * p for r, p in zip(ang_R, ang_P)]
    dih_mid = [circular_interpolate(r, p, alpha) for r, p in zip(dih_R, dih_P)]

    x0_init = 0.5 * (pos_R + pos_P)
    x0 = _reconstruct_from_ic(
        x0_init, bonds, bl_mid, angles, ang_mid, dihedrals_list, dih_mid,
        max_iter=max_lbfgs_iter,
        w_bond=weight_bond, w_angle=weight_angle, w_dihedral=weight_dihedral,
    )

    if apply_clash_check:
        x0 = _apply_clash_correction(x0, atomic_numbers)
    x0 -= x0.mean(axis=0)
    return x0


# ---------------------------------------------------------------------------
# Idea 2-D v2: xTB
# ---------------------------------------------------------------------------

def compute_xtb_refined_x0(
    x0_initial: np.ndarray,
    atomic_numbers: np.ndarray,
    core_atom_pairs: Optional[List[Tuple[int, int]]] = None,
    max_steps: int = 20,
    fmax: float = 0.5,
    max_step_size: float = 0.05,
    method: str = 'GFN2-xTB',
    charge: int = 0,
    multiplicity: int = 1,
    fallback_on_error: bool = True,
) -> np.ndarray:
    try:
        from ase import Atoms
        from ase.optimize import LBFGS
        from ase.constraints import FixBondLengths
    except ImportError:
        if fallback_on_error:
            return np.asarray(x0_initial, dtype=np.float64).copy()
        raise

    XTB = None
    try:
        from xtb.ase.calculator import XTB as _XTB
        XTB = _XTB
    except ImportError:
        try:
            from ase.calculators.xtb import XTB as _XTB
            XTB = _XTB
        except ImportError:
            if fallback_on_error:
                print("WARNING: xTB not available, returning original structure")
                return np.asarray(x0_initial, dtype=np.float64).copy()
            raise

    atoms = Atoms(
        numbers=np.asarray(atomic_numbers).astype(int),
        positions=np.asarray(x0_initial, dtype=np.float64).copy(),
    )
    try:
        calc = XTB(method=method, charge=charge, uhf=multiplicity - 1)
    except TypeError:
        calc = XTB(method=method, charge=charge)
    atoms.calc = calc

    if core_atom_pairs:
        atoms.set_constraint(
            FixBondLengths([list(pair) for pair in core_atom_pairs])
        )

    try:
        opt = LBFGS(atoms, maxstep=max_step_size, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        if fallback_on_error:
            print(f"WARNING: xTB optimization failed ({e}), returning original")
            return np.asarray(x0_initial, dtype=np.float64).copy()
        raise

    x0_refined = atoms.get_positions()
    x0_refined -= x0_refined.mean(axis=0)
    return x0_refined


def _fallback_core_pairs(pos_R: np.ndarray, pos_P: np.ndarray,
                         atomic_numbers: np.ndarray,
                         distance_change_thr: float = 0.5
                         ) -> List[Tuple[int, int]]:
    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39}
    N = len(atomic_numbers)
    d_R = cdist(pos_R, pos_R)
    d_P = cdist(pos_P, pos_P)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r_cov = (covalent_radii.get(int(atomic_numbers[i]), 0.77) +
                     covalent_radii.get(int(atomic_numbers[j]), 0.77))
            is_bond = (d_R[i, j] < r_cov * 1.3) or (d_P[i, j] < r_cov * 1.3)
            if is_bond and abs(d_R[i, j] - d_P[i, j]) < distance_change_thr:
                pairs.append((i, j))
    return pairs


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
# Combo wrappers: BD, BE, BDE
# ---------------------------------------------------------------------------

def compute_x0_BD(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    ic_kwargs: Optional[dict] = None,
    xtb_kwargs: Optional[dict] = None,
    use_idea1_core: bool = False,
    charge: int = 0,
    fallback_to_idpp: bool = True,
) -> np.ndarray:
    """Combo [BD]: IC interpolation -> xTB short refinement."""
    pos_R = np.asarray(pos_R, dtype=np.float64)
    pos_P = np.asarray(pos_P, dtype=np.float64)

    ic_kw = ic_kwargs or {'alpha': 0.5, 'apply_clash_check': True,
                          'max_lbfgs_iter': 100}
    xtb_kw = dict(xtb_kwargs or {'max_steps': 20, 'fmax': 0.5})
    xtb_kw.setdefault('charge', charge)

    try:
        x0_ic = compute_ic_interpolation(pos_R, pos_P, atomic_numbers, **ic_kw)
        if (np.any(np.isnan(x0_ic)) or
            np.any(np.linalg.norm(x0_ic - x0_ic.mean(axis=0), axis=1) > 50.0)):
            raise RuntimeError("IC interpolation produced invalid structure")
    except Exception as e:
        if fallback_to_idpp:
            print(f"INFO: IC failed ({e}), fallback to IDPP")
            x0_ic = compute_idpp(pos_R, pos_P, atomic_numbers,
                                 max_iter=200, use_clash_penalty=True)
        else:
            raise

    if use_idea1_core:
        try:
            from reactot.utils.reactive_core import (
                find_reactive_core_from_positions,
                build_adjacency_matrix,
            )
            core_set = find_reactive_core_from_positions(
                pos_R, pos_P, atomic_numbers
            )
            adj = build_adjacency_matrix(pos_R, atomic_numbers)
            core_pairs = [(i, j) for i in sorted(core_set)
                          for j in sorted(core_set)
                          if j > i and adj[i, j]]
        except ImportError:
            core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)
    else:
        core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)

    return compute_xtb_refined_x0(
        x0_ic, atomic_numbers,
        core_atom_pairs=core_pairs if core_pairs else None,
        **xtb_kw,
    )


def compute_x0_BE(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    K: int = 5,
    sigma_base: float = 0.1,
    seed: Optional[int] = None,
    ic_kwargs: Optional[dict] = None,
    atom_weights: Optional[np.ndarray] = None,
    core_weight_clip: float = 0.9,
    fallback_to_idpp: bool = True,
) -> np.ndarray:
    """Combo [BE] Stage 1+2: IC base + K stochastic candidates."""
    ic_kw = ic_kwargs or {'alpha': 0.5, 'apply_clash_check': True,
                          'max_lbfgs_iter': 100}
    try:
        x0_base = compute_ic_interpolation(
            pos_R, pos_P, atomic_numbers, **ic_kw
        )
        if (np.any(np.isnan(x0_base)) or
            np.any(np.linalg.norm(x0_base - x0_base.mean(axis=0), axis=1) > 50.0)):
            raise RuntimeError("IC produced invalid structure")
    except Exception as e:
        if fallback_to_idpp:
            print(f"INFO: IC failed ({e}), fallback to IDPP")
            x0_base = compute_idpp(pos_R, pos_P, atomic_numbers,
                                   max_iter=200, use_clash_penalty=True)
        else:
            raise

    if atom_weights is None:
        atom_weights = _fallback_hop_weights(pos_R, pos_P, atomic_numbers)

    return generate_stochastic_x0_ensemble(
        x0_base, atomic_numbers, atom_weights,
        K=K, sigma_base=sigma_base, seed=seed,
        apply_clash_check=True, core_weight_clip=core_weight_clip,
    )


def compute_x0_BDE(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    K: int = 5,
    sigma_base: float = 0.1,
    seed: Optional[int] = None,
    ic_kwargs: Optional[dict] = None,
    xtb_kwargs: Optional[dict] = None,
    atom_weights: Optional[np.ndarray] = None,
    core_weight_clip: float = 0.9,
    use_idea1_core: bool = False,
    charge: int = 0,
    fallback_to_idpp: bool = True,
) -> np.ndarray:
    """
    Combo [BDE] Stage 1-3: IC -> xTB -> K candidates (perturbed on the
    refined base; D-before-E to keep xTB cost O(1) per reaction).
    """
    x0_refined = compute_x0_BD(
        pos_R, pos_P, atomic_numbers,
        ic_kwargs=ic_kwargs,
        xtb_kwargs=xtb_kwargs,
        use_idea1_core=use_idea1_core,
        charge=charge,
        fallback_to_idpp=fallback_to_idpp,
    )

    if atom_weights is None:
        atom_weights = _fallback_hop_weights(pos_R, pos_P, atomic_numbers)

    return generate_stochastic_x0_ensemble(
        x0_refined, atomic_numbers, atom_weights,
        K=K, sigma_base=sigma_base, seed=seed,
        apply_clash_check=True, core_weight_clip=core_weight_clip,
    )
