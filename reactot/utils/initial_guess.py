"""
OT-FM 초기 구조(x_0) 생성 모듈.
(R+P)/2의 원자 겹침 문제를 해결하는 물리적 초기 구조를 생성한다.

본 모듈은 idea2 조합 [AD] (= IDPP + GFN2-xTB refinement)를 위해
다음 컴포넌트를 한 파일에 통합한다:
- Idea 2-A v2: IDPP + halogen-aware clash penalty (compute_idpp)
- Idea 2-D v2: GFN2-xTB short refinement (compute_xtb_refined_x0)
- Combo [AD] wrapper: compute_x0_AD

References:
- IDPP: Smidstrup et al., J. Chem. Phys., 2014, DOI: 10.1063/1.4878664
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
- GFN2-xTB: Bannwarth, Ehlert, Grimme, J. Chem. Theory Comput., 2019,
  DOI: 10.1021/acs.jctc.8b01176
- Halo8 (xTB optimization in reactant prep): Lee et al., Sci. Data, 2025,
  DOI: 10.1038/s41597-025-05944-3
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Idea 2-A v2: vdW + halogen-aware clash threshold
# ---------------------------------------------------------------------------

# Bondi vdW 반지름 (Å)
VDW_RADII = {
    1:  1.20,  # H
    6:  1.70,  # C
    7:  1.55,  # N
    8:  1.52,  # O
    9:  1.47,  # F
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}

DEFAULT_CLASH_SCALES = {
    'default': 0.70,
    'halogen_halogen': 0.80,
    'halogen_H': 0.65,
    'halogen_heavy': 0.75,
}

HALOGENS = {9, 17, 35, 53}  # F, Cl, Br, I


def is_halogen(z: int) -> bool:
    """원소가 할로겐인지 판별."""
    return int(z) in HALOGENS


def get_clash_threshold(z_i: int, z_j: int,
                        clash_scales: Optional[dict] = None) -> float:
    """
    원소쌍의 clash 판정 임계 거리를 반환한다.

    r_clash = f_scale * (R_vdW_i + R_vdW_j)
    """
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
# Idea 2-A v2: IDPP + clash penalty
# ---------------------------------------------------------------------------

def compute_idpp(pos_R: np.ndarray, pos_P: np.ndarray,
                 atomic_numbers: Optional[np.ndarray] = None,
                 max_iter: int = 200, tol: float = 0.01,
                 lr: float = 0.01,
                 use_clash_penalty: bool = True,
                 clash_kappa: float = 10.0) -> np.ndarray:
    """
    IDPP + 할로겐 인지 clash penalty로 초기 구조를 생성한다.

    IDPP 목적함수:
        S(r) = Σ_{i<j} w_ij * (||r_i - r_j|| - d_ij^target)²
        w_ij = (d_ij^target)^{-4}
        d_ij^target = (d_ij^R + d_ij^P) / 2

    Clash penalty (one-sided quadratic):
        P(r) = κ * Σ_{i<j} max(0, r_clash_ij - ||r_i - r_j||)²

    Returns:
        x0: (N, 3) IDPP-optimized initial structure (COM at origin)
    """
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
            2.0 * w[:, :, None] * residual[:, :, None] * unit_vec,
            axis=1
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
        max_grad = np.max(np.linalg.norm(grad, axis=1))
        if max_grad < tol:
            break

        x -= lr * grad

    x -= x.mean(axis=0)
    return x


def compute_idpp_batch(pos_R_list, pos_P_list, atomic_numbers_list,
                       **kwargs) -> list:
    """배치 내 각 분자에 대해 IDPP를 실행한다."""
    return [
        compute_idpp(pos_R, pos_P, z, **kwargs)
        for pos_R, pos_P, z in zip(pos_R_list, pos_P_list, atomic_numbers_list)
    ]


# ---------------------------------------------------------------------------
# Idea 2-D v2: GFN2-xTB short refinement
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
    """
    GFN2-xTB로 초기 구조를 짧게 최적화한다.

    Core 원자 간 거리를 constraint로 고정하여 반응 좌표를 보존하면서,
    비결합 상호작용(vdW, 정전기)만 이완시킨다.

    Args:
        x0_initial: (N, 3) 초기 구조 (IDPP 또는 (R+P)/2)
        atomic_numbers: (N,)
        core_atom_pairs: list of (i, j) — constraint로 고정할 core 원자쌍.
                         None이면 constraint 없이 최적화.
        max_steps: 최대 최적화 step 수 (매우 짧게!)
        fmax: force 수렴 기준 (eV/Å, ASE 표준)
        max_step_size: 한 step의 최대 이동량 (Å)
        method: xTB 방법
        charge: 분자 전하
        multiplicity: 스핀 다중도
        fallback_on_error: xTB 오류 시 원본 반환 여부

    Returns:
        x0_refined: (N, 3) refined structure (COM at origin)
    """
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
            raise ImportError(
                "xtb-python이 필요합니다: conda install -c conda-forge xtb-python"
            )

    atoms = Atoms(
        numbers=np.asarray(atomic_numbers).astype(int),
        positions=np.asarray(x0_initial, dtype=np.float64).copy(),
    )

    try:
        calc = XTB(method=method, charge=charge, uhf=multiplicity - 1)
    except TypeError:
        calc = XTB(method=method, charge=charge)
    atoms.calc = calc

    if core_atom_pairs is not None and len(core_atom_pairs) > 0:
        constraints = FixBondLengths([list(pair) for pair in core_atom_pairs])
        atoms.set_constraint(constraints)

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
                         distance_change_thr: float = 0.5) -> List[Tuple[int, int]]:
    """
    Idea 1의 reactive_core 모듈이 없을 때 fallback core 결합쌍 식별.

    R↔P 결합 길이 변화가 distance_change_thr (Å) 미만인 결합을
    "반응에 직접 참여하지 않는 결합"으로 간주하고 core_pairs에 포함.
    이는 xTB constraint로 사용되어 그 결합 길이를 고정한다.
    """
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
            if not is_bond:
                continue
            if abs(d_R[i, j] - d_P[i, j]) < distance_change_thr:
                pairs.append((i, j))
    return pairs


def compute_xtb_refined_x0_with_idpp(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    idpp_kwargs: Optional[dict] = None,
    xtb_kwargs: Optional[dict] = None,
    use_idea1_core: bool = False,
) -> np.ndarray:
    """
    IDPP → xTB refinement 2단계 파이프라인.
    """
    idpp_kw = idpp_kwargs or {}
    xtb_kw = xtb_kwargs or {}

    x0_idpp = compute_idpp(pos_R, pos_P, atomic_numbers, **idpp_kw)

    core_pairs: List[Tuple[int, int]] = []
    if use_idea1_core:
        try:
            from reactot.utils.reactive_core import (
                find_reactive_core_from_positions,
                build_adjacency_matrix,
            )
            core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)
            core_list = sorted(core)
            adj = build_adjacency_matrix(pos_R, atomic_numbers)
            for i in core_list:
                for j in core_list:
                    if j > i and adj[i, j]:
                        core_pairs.append((i, j))
        except ImportError:
            core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)
    else:
        core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)

    x0_refined = compute_xtb_refined_x0(
        x0_idpp, atomic_numbers,
        core_atom_pairs=core_pairs if core_pairs else None,
        **xtb_kw,
    )
    return x0_refined


# ---------------------------------------------------------------------------
# Combo [AD]: IDPP + xTB refinement wrapper
# ---------------------------------------------------------------------------

def compute_x0_AD(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    idpp_kwargs: Optional[dict] = None,
    xtb_kwargs: Optional[dict] = None,
    use_idea1_core: bool = False,
    charge: int = 0,
) -> np.ndarray:
    """
    조합 [AD]: IDPP → xTB.

    이는 compute_xtb_refined_x0_with_idpp의 명시적 alias이며,
    charge 인자를 xtb_kwargs에 합성하여 다른 조합과 일관된 인터페이스를 제공한다.

    Args:
        pos_R, pos_P: (N, 3) Hungarian+Kabsch aligned
        atomic_numbers: (N,)
        idpp_kwargs: {'max_iter', 'tol', 'lr', 'use_clash_penalty', 'clash_kappa'}
        xtb_kwargs: {'max_steps', 'fmax', 'method', 'multiplicity'}
        use_idea1_core: Idea 1의 reactive_core 함수 사용 여부
        charge: 분자 전하 (Halo8 SN2 등 음전하 시스템 -1)

    Returns:
        x0: (N, 3) refined initial structure (COM at origin)
    """
    idpp_kw = idpp_kwargs or {'max_iter': 200, 'use_clash_penalty': True}
    xtb_kw = dict(xtb_kwargs or {'max_steps': 20, 'fmax': 0.5})
    # charge는 명시적 인자로 전달되므로 xtb_kw에 합성
    xtb_kw.setdefault('charge', charge)

    return compute_xtb_refined_x0_with_idpp(
        pos_R, pos_P, atomic_numbers,
        idpp_kwargs=idpp_kw,
        xtb_kwargs=xtb_kw,
        use_idea1_core=use_idea1_core,
    )
