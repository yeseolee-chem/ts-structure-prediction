"""
OT-FM 초기 구조(x_0) 생성 모듈 — Idea 2-B (내부 좌표 Geodesic 보간).

Cartesian 좌표의 산술 평균 (R+P)/2 대신, 내부 좌표(결합 길이/각/이면각)에서
보간한 뒤 Cartesian으로 역변환하여 화학적으로 합리적인 초기 구조를 생성한다.
이면각의 주기성을 올바르게 처리하여 잘못된 conformer 생성을 방지한다.

References:
- Zimmerman, J. Chem. Theory Comput., 2013, DOI: 10.1021/ct400319w (GSM)
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
"""
import numpy as np
from scipy.optimize import minimize
from typing import Optional

from reactot.utils.internal_coords import (
    get_connectivity, get_angles, get_dihedrals,
    compute_bond_length, compute_angle, compute_dihedral,
    circular_interpolate,
)


# ----------------------------------------------------------------------------
# Clash threshold (Idea 2-A 의존성).
# Idea 2-B의 `_apply_clash_correction`이 이 함수를 요구한다 (md 명시).
# ----------------------------------------------------------------------------

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

# 원소쌍별 clash scale factor
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


# ----------------------------------------------------------------------------
# Idea 2-B: 내부 좌표 Geodesic 보간
# ----------------------------------------------------------------------------


def compute_ic_interpolation(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    alpha: float = 0.5,
    apply_clash_check: bool = False,
    max_lbfgs_iter: int = 20,
    weight_bond: float = 1.0,
    weight_angle: float = 0.5,
    weight_dihedral: float = 0.3,
) -> np.ndarray:
    """
    내부 좌표에서 보간하여 초기 구조를 생성한다.

    1. R 기준 토폴로지 추출 (bonds, angles, dihedrals)
    2. R, P의 내부 좌표 계산
    3. 내부 좌표 보간 (이면각은 circular_interpolate)
    4. (R+P)/2를 시작점으로 L-BFGS로 내부 좌표 constraint 만족 좌표 탐색
    5. (선택) vdW clash 검사 및 보정

    Args:
        pos_R: (N, 3) reactant
        pos_P: (N, 3) product (원자 순서 일치)
        atomic_numbers: (N,)
        alpha: 보간 계수 (0=R, 1=P, 0.5=중점)
        apply_clash_check: clash 검사 적용 여부

    Returns:
        x0: (N, 3) interpolated structure
    """
    # 1. 연결 정보 (R 기준; v2: get_dihedrals는 bonds만 받음)
    bonds = get_connectivity(pos_R, atomic_numbers)
    angles = get_angles(bonds)
    dihedrals_list = get_dihedrals(bonds)

    # 2. R, P의 내부 좌표 계산
    bl_R = [compute_bond_length(pos_R, i, j) for i, j in bonds]
    bl_P = [compute_bond_length(pos_P, i, j) for i, j in bonds]

    ang_R = [compute_angle(pos_R, i, j, k) for i, j, k in angles]
    ang_P = [compute_angle(pos_P, i, j, k) for i, j, k in angles]

    dih_R = [compute_dihedral(pos_R, i, j, k, l) for i, j, k, l in dihedrals_list]
    dih_P = [compute_dihedral(pos_P, i, j, k, l) for i, j, k, l in dihedrals_list]

    # 3. 보간
    bl_mid = [(1 - alpha) * r + alpha * p for r, p in zip(bl_R, bl_P)]
    ang_mid = [(1 - alpha) * r + alpha * p for r, p in zip(ang_R, ang_P)]
    dih_mid = [circular_interpolate(r, p, alpha) for r, p in zip(dih_R, dih_P)]

    # 4. Cartesian 재구성: (R+P)/2를 시작점으로 L-BFGS optimization
    x0_init = 0.5 * (pos_R + pos_P)
    x0 = _reconstruct_from_ic(
        x0_init, bonds, bl_mid, angles, ang_mid, dihedrals_list, dih_mid,
        max_iter=max_lbfgs_iter,
        w_bond=weight_bond, w_angle=weight_angle, w_dihedral=weight_dihedral,
    )

    # 5. Clash 보정
    if apply_clash_check:
        x0 = _apply_clash_correction(x0, atomic_numbers)

    # 6. Center-of-mass 제거
    x0 -= x0.mean(axis=0)

    return x0


def _reconstruct_from_ic(x0_init, bonds, bl_target, angles, ang_target,
                         dihedrals, dih_target,
                         max_iter=100,
                         w_bond=1.0, w_angle=0.5, w_dihedral=0.3):
    """
    내부 좌표 목표값에 맞게 Cartesian 좌표를 재구성한다 (L-BFGS-B).

    Note: dihedral cost는 주기적 거리(wrap into [-π, π])로 계산.
    """
    def objective(x_flat):
        x = x_flat.reshape(-1, 3)
        cost = 0.0

        # Bond length cost
        for idx, (i, j) in enumerate(bonds):
            bl = float(np.linalg.norm(x[i] - x[j]))
            cost += w_bond * (bl - bl_target[idx])**2

        # Angle cost
        for idx, (i, j, k) in enumerate(angles):
            v1 = x[i] - x[j]
            v2 = x[k] - x[j]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            a = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            cost += w_angle * (a - ang_target[idx])**2

        # Dihedral cost (주기적 거리)
        for idx, (i, j, k, l) in enumerate(dihedrals):
            dih = compute_dihedral(x, i, j, k, l)
            diff = dih - dih_target[idx]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
            cost += w_dihedral * diff**2

        return cost

    result = minimize(
        objective, x0_init.flatten(), method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-6}
    )

    return result.x.reshape(-1, 3)


def _apply_clash_correction(x0, atomic_numbers, n_iter=50):
    """
    간단한 soft-sphere repulsion으로 clash 제거.
    **v2**: get_clash_threshold가 같은 파일에 정의되어 있음을 명시.
    """
    # get_clash_threshold는 이 파일(initial_guess.py)에 정의되어 있다고 가정
    # (Idea 2-A의 코드 참고)
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
                    push_amount = 0.5 * (r_min - r_ij)
                    x0[i] += push_amount * direction
                    x0[j] -= push_amount * direction
        if not has_clash:
            break
    return x0
