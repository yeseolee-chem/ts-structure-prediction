"""
OT-FM 초기 구조(x_0) 생성 모듈.
(R+P)/2의 원자 겹침 문제를 해결하는 물리적 초기 구조를 생성한다.

Idea 2-C (v2): vdW 인지형 Midpoint + Repulsive Correction.

References:
- vdW radii: Bondi, J. Phys. Chem., 1964, DOI: 10.1021/j100785a001
"""
import numpy as np
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
    (R+P)/2를 계산한 뒤, vdW clash를 iterative repulsion으로 제거한다.

    알고리즘:
    1. x_0 = (R + P) / 2
    2. 모든 원자쌍 (i,j)에 대해, r_clash = f_scale * (R_vdW_i + R_vdW_j)일 때
       r_ij < r_clash이면 두 원자를 양쪽으로 동등하게 push.
       한 step에서 분리 거리 증가량 = recovery_rate * (r_clash - r_ij)
       → recovery_rate=1.0이면 한 step에서 r_ij ← r_clash로 정확히 회복
       → recovery_rate=0.5이면 한 step에서 violation의 절반만 회복(보수적)
    3. 수렴(=clash 없음)까지 최대 n_iter회 반복
    4. Center-of-mass 제거

    **v2 변경**: 기존 `push_fraction` 파라미터 의미가 모호했음
    (실제 적용량 = push_fraction × violation × 0.5 × 2). 이를 `recovery_rate`로
    이름 변경 + 동작을 직관적으로 재정의.

    Args:
        pos_R: (N, 3) reactant positions (aligned)
        pos_P: (N, 3) product positions
        atomic_numbers: (N,)
        n_iter: 최대 반복 횟수
        recovery_rate: 한 step에서 violation을 회복하는 비율 (0~1].
                       값이 클수록 빠른 수렴, 작을수록 부드러운 보정.
        clash_scales: 원소쌍별 f_scale 딕셔너리
        rng_seed: 완전 겹침(r≈0) 처리 시 random push 방향의 seed (재현성)

    Returns:
        x0: (N, 3) clash-corrected midpoint
    """
    N = pos_R.shape[0]
    x0 = 0.5 * (pos_R + pos_P).copy()
    rng = np.random.RandomState(rng_seed)

    # 원소쌍별 clash 임계값 사전 계산 (대칭 행렬)
    r_clash_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            r_clash_matrix[i, j] = get_clash_threshold(
                int(atomic_numbers[i]), int(atomic_numbers[j]),
                clash_scales,
            )
            r_clash_matrix[j, i] = r_clash_matrix[i, j]

    # Iterative repulsion
    for _ in range(n_iter):
        has_clash = False

        for i in range(N):
            for j in range(i + 1, N):
                diff = x0[i] - x0[j]
                r_ij = float(np.linalg.norm(diff))
                r_min = r_clash_matrix[i, j]

                if r_ij < 1e-8:
                    # 완전 겹침 → 랜덤 방향으로 r_min만큼 분리
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
                    # 한 step의 총 분리 증가량 = recovery_rate * violation
                    # 두 원자에 동등하게 분배 → 각 원자는 0.5 * recovery_rate * violation 이동
                    violation = r_min - r_ij
                    half_push = 0.5 * recovery_rate * violation
                    x0[i] += half_push * direction
                    x0[j] -= half_push * direction

        if not has_clash:
            break

    # Center-of-mass 제거
    x0 -= x0.mean(axis=0)

    return x0


def count_clashes(positions: np.ndarray, atomic_numbers: np.ndarray,
                  clash_scales: Optional[dict] = None) -> dict:
    """
    구조의 vdW clash 수를 세고 통계를 반환한다. 디버깅 및 비교 분석용.

    Returns:
        stats: {
            'n_clashes': int,
            'worst_violation': float (Å),
            'clash_pairs': list of (i, j, r_ij, r_clash),
        }
    """
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
