"""
내부 좌표(Internal Coordinates) ↔ Cartesian 좌표 변환 유틸리티.

내부 좌표: {결합 길이(bond), 결합각(angle), 이면각(dihedral)}

Dihedral convention: IUPAC (Praxeolitic formula)
  - Looking along bond j→k, dihedral = angle from i to l, measured CCW
  - Range: [-π, π]
"""
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional


def get_connectivity(positions: np.ndarray, atomic_numbers: np.ndarray,
                     scale: float = 1.3) -> List[Tuple[int, int]]:
    """
    공유결합 반지름 기반 연결 정보 추출.

    Returns:
        bonds: list of (i, j) pairs with i < j
    """
    from scipy.spatial.distance import cdist

    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39}

    N = len(positions)
    dist = cdist(positions, positions)
    bonds = []

    for i in range(N):
        for j in range(i + 1, N):
            r_cov = (covalent_radii.get(int(atomic_numbers[i]), 0.77) +
                     covalent_radii.get(int(atomic_numbers[j]), 0.77))
            if dist[i, j] < r_cov * scale:
                bonds.append((i, j))

    return bonds


def build_neighbor_dict(bonds: List[Tuple[int, int]]) -> dict:
    """결합 목록으로부터 인접 정보 딕셔너리를 구성."""
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def get_angles(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """
    결합 정보로부터 결합각(i-j-k) 목록을 추출한다. j가 중심 원자.
    중복 방지를 위해 i < k.
    """
    neighbors = build_neighbor_dict(bonds)

    angles = []
    for j in neighbors:
        nb_list = sorted(neighbors[j])
        for idx_a in range(len(nb_list)):
            for idx_b in range(idx_a + 1, len(nb_list)):
                i = nb_list[idx_a]
                k = nb_list[idx_b]
                angles.append((i, j, k))

    return angles


def get_dihedrals(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    이면각(i-j-k-l) 목록을 추출한다. **v2 수정**: 모든 j-k 중심 결합에 대해
    완전한 조합을 한 번씩만 enumerate.

    각 결합 (a, b)를 j-k 중심 결합으로 보고:
        i ∈ N(a) \\ {b}, l ∈ N(b) \\ {a}
    중복 방지를 위해 i < l (대안: 결합 (a,b) 자체에 a < b 가정).

    이전 버그: angles에서 시작하면 일부 dihedral 누락 가능
        (예: 결합 j-k 중심에서 i ∈ N(j)에 i > k인 원자가 있는 경우)
    """
    neighbors = build_neighbor_dict(bonds)

    dihedrals = []
    seen = set()

    for a, b in bonds:                # bonds는 a < b 보장
        # j=a, k=b 방향
        for i in neighbors[a]:
            if i == b:
                continue
            for l in neighbors[b]:
                if l == a or l == i:
                    continue
                # 중복 방지: (i,j,k,l)과 (l,k,j,i)는 같은 dihedral
                key = tuple(sorted([(i, a), (l, b)]))
                if key in seen:
                    continue
                seen.add(key)
                dihedrals.append((i, a, b, l))

    return dihedrals


def compute_bond_length(pos: np.ndarray, i: int, j: int) -> float:
    """결합 길이 계산."""
    return float(np.linalg.norm(pos[i] - pos[j]))


def compute_angle(pos: np.ndarray, i: int, j: int, k: int) -> float:
    """결합각 계산 (라디안). j가 중심 원자."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def compute_dihedral(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """
    이면각 i-j-k-l 계산 (라디안, [-π, π]). Praxeolitic formula.

    b1 = j - i,  b2 = k - j,  b3 = l - k
    n1 = b1 × b2 (plane(i,j,k) normal),  n2 = b2 × b3 (plane(j,k,l) normal)
    """
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_unit = b2 / (np.linalg.norm(b2) + 1e-10)

    n1_norm = np.linalg.norm(n1) + 1e-10
    n2_norm = np.linalg.norm(n2) + 1e-10
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2_unit)         # 단위 벡터(n1과 b2 모두에 직교)

    x = float(np.dot(n1, n2))           # cos(dihedral)
    y = float(np.dot(m1, n2))           # sin(dihedral)
    return float(np.arctan2(y, x))


def circular_interpolate(angle_R: float, angle_P: float,
                         alpha: float = 0.5) -> float:
    """
    이면각의 주기적 보간 (slerp on the unit circle).

    **v2 수정**: alpha 파라미터 지원 (기존 circular_mean은 alpha=0.5만 가능했음).
        φ_α = atan2( (1-α)·sin(φ_R) + α·sin(φ_P),
                     (1-α)·cos(φ_R) + α·cos(φ_P) )

    예: φ_R = 10°, φ_P = 350°, α=0.5 → φ_α ≈ 0° (not 180°)

    Args:
        angle_R, angle_P: 라디안
        alpha: 보간 계수 (0=R, 1=P, 0.5=중점)
    """
    sin_mix = (1.0 - alpha) * np.sin(angle_R) + alpha * np.sin(angle_P)
    cos_mix = (1.0 - alpha) * np.cos(angle_R) + alpha * np.cos(angle_P)
    return float(np.arctan2(sin_mix, cos_mix))
