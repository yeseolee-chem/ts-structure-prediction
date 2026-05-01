"""
OT-FM 초기 구조(x_0) 생성 모듈.
(R+P)/2의 원자 겹침 문제를 해결하는 물리적 초기 구조를 생성한다.

References:
- IDPP: Smidstrup et al., J. Chem. Phys., 2014, DOI: 10.1063/1.4878664
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Optional


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
    53: 1.98,  # I  (확장성을 위해 추가)
}

# 원소쌍별 clash scale factor
# 할로겐 쌍은 더 보수적(큰 f_scale)으로 설정
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

    # v2 수정: 연산자 우선순위 모호성 제거를 위해 명시적 괄호 추가
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
    """
    IDPP + 할로겐 인지 clash penalty로 초기 구조를 생성한다.

    IDPP 목적함수:
        S(r) = Σ_{i<j} w_ij * (||r_i - r_j|| - d_ij^target)²
        w_ij = (d_ij^target)^{-4}
        d_ij^target = (d_ij^R + d_ij^P) / 2

    Clash penalty (one-sided quadratic):
        P(r) = κ * Σ_{i<j} max(0, r_clash_ij - ||r_i - r_j||)²

    Args:
        pos_R: (N, 3) reactant positions (Hungarian + Kabsch 적용 후)
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) — clash penalty에 필요 (None이면 penalty 미적용)
        max_iter: 최대 반복 횟수
        tol: 수렴 기울기 임계값 (Å, 원자별 gradient norm 최댓값 기준)
        lr: 학습률
        use_clash_penalty: clash penalty 적용 여부
        clash_kappa: clash penalty 강도

    Returns:
        x0: (N, 3) IDPP-optimized initial structure
    """
    N = pos_R.shape[0]

    # 1. 목표 쌍거리
    d_R = squareform(pdist(pos_R))
    d_P = squareform(pdist(pos_P))
    d_target = 0.5 * (d_R + d_P)

    # 2. IDPP 가중치 (가까운 쌍에 더 큰 가중치)
    with np.errstate(divide='ignore'):
        w = np.where(d_target > 1e-6, d_target**(-4), 0.0)
    np.fill_diagonal(w, 0.0)

    # 3. Clash 임계값 행렬
    r_clash = np.zeros((N, N))
    if use_clash_penalty and atomic_numbers is not None:
        for i in range(N):
            for j in range(i + 1, N):
                r_clash[i, j] = get_clash_threshold(
                    int(atomic_numbers[i]), int(atomic_numbers[j])
                )
                r_clash[j, i] = r_clash[i, j]

    # 4. 초기값: (R+P)/2
    x = 0.5 * (pos_R + pos_P).copy()

    # 5. 최적화 루프 (gradient descent)
    for iteration in range(max_iter):
        # 현재 쌍거리
        diff = x[:, None, :] - x[None, :, :]                # (N, N, 3), diff[i,j] = x_i - x_j
        d_curr = np.linalg.norm(diff, axis=-1)              # (N, N)
        d_curr_safe = d_curr + np.eye(N) * 1e-10
        unit_vec = diff / d_curr_safe[:, :, None]           # (N, N, 3), unit_vec[i,j] = (x_i-x_j)/||·||

        # IDPP gradient: ∂S/∂x_i = Σ_j 2 * w_ij * (||r_i-r_j|| - d_target_ij) * unit_vec[i,j]
        residual = d_curr_safe - d_target                   # (N, N)
        grad_idpp = np.sum(
            2.0 * w[:, :, None] * residual[:, :, None] * unit_vec,
            axis=1
        )                                                    # (N, 3)

        # Clash penalty gradient
        # P_ij = κ * (r_clash - r_ij)²  if r_ij < r_clash, else 0
        # ∂P_ij/∂x_i = -2κ * (r_clash - r_ij) * unit_vec[i,j]
        # → x_i 업데이트(x_i -= lr * grad)는 결과적으로 unit_vec[i,j] 방향(j로부터 멀어지는 방향)으로 이동
        grad_clash = np.zeros_like(x)
        if use_clash_penalty and atomic_numbers is not None:
            for i in range(N):
                for j in range(i + 1, N):
                    r_ij = d_curr_safe[i, j]
                    if r_ij < r_clash[i, j]:
                        violation = r_clash[i, j] - r_ij
                        # 부호: ∂P/∂x_i = -2κ * violation * unit_vec[i,j]
                        # (factor 2 생략, kappa로 흡수)
                        grad_pair = -clash_kappa * violation * unit_vec[i, j]
                        grad_clash[i] += grad_pair          # ∂P/∂x_i
                        grad_clash[j] -= grad_pair          # ∂P/∂x_j = -∂P/∂x_i (Newton's 3rd law)

        # 전체 gradient
        grad = grad_idpp + grad_clash

        # 수렴 판정
        max_grad = np.max(np.linalg.norm(grad, axis=1))
        if max_grad < tol:
            break

        # 업데이트
        x -= lr * grad

    # 6. Center-of-mass 제거 (equivariance)
    x -= x.mean(axis=0)

    return x


def compute_idpp_batch(pos_R_list, pos_P_list, atomic_numbers_list,
                       **kwargs) -> list:
    """
    배치 내 각 분자에 대해 IDPP를 실행한다.

    Args:
        pos_R_list: list of (N_i, 3) arrays
        pos_P_list: list of (N_i, 3) arrays
        atomic_numbers_list: list of (N_i,) arrays

    Returns:
        x0_list: list of (N_i, 3) arrays
    """
    x0_list = []
    for pos_R, pos_P, z in zip(pos_R_list, pos_P_list, atomic_numbers_list):
        x0 = compute_idpp(pos_R, pos_P, z, **kwargs)
        x0_list.append(x0)
    return x0_list
