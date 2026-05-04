"""
Internal coordinate utilities (Idea 2-B v2).

Bond / angle / dihedral helpers for the IC interpolation pathway and a
``circular_interpolate`` for periodic dihedral averaging.

Dihedral convention: IUPAC (Praxeolitic). Looking along bond j→k, the
dihedral is the angle from i to l measured CCW; range [-π, π].
"""
import numpy as np
from collections import defaultdict
from typing import List, Tuple


def get_connectivity(positions: np.ndarray, atomic_numbers: np.ndarray,
                     scale: float = 1.3) -> List[Tuple[int, int]]:
    """Covalent-radius based bond extraction. Returns ``[(i, j) ...]`` with i<j."""
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
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def get_angles(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Bond angles (i, j, k) with j as central atom and i < k."""
    neighbors = build_neighbor_dict(bonds)
    angles = []
    for j in neighbors:
        nb_list = sorted(neighbors[j])
        for idx_a in range(len(nb_list)):
            for idx_b in range(idx_a + 1, len(nb_list)):
                angles.append((nb_list[idx_a], j, nb_list[idx_b]))
    return angles


def get_dihedrals(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Dihedrals (i, j, k, l). v2: enumerate over j-k center bonds so no
    dihedral is missed in branched topologies.
    """
    neighbors = build_neighbor_dict(bonds)
    dihedrals = []
    seen = set()
    for a, b in bonds:  # bonds enforce a < b
        for i in neighbors[a]:
            if i == b:
                continue
            for l in neighbors[b]:
                if l == a or l == i:
                    continue
                key = tuple(sorted([(i, a), (l, b)]))
                if key in seen:
                    continue
                seen.add(key)
                dihedrals.append((i, a, b, l))
    return dihedrals


def compute_bond_length(pos: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pos[i] - pos[j]))


def compute_angle(pos: np.ndarray, i: int, j: int, k: int) -> float:
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_angle = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    )
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def compute_dihedral(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral i-j-k-l in radians, [-π, π]. Praxeolitic formula."""
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_unit = b2 / (np.linalg.norm(b2) + 1e-10)

    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    m1 = np.cross(n1, b2_unit)

    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def circular_interpolate(angle_R: float, angle_P: float,
                         alpha: float = 0.5) -> float:
    """
    Periodic dihedral interpolation (slerp on the unit circle):
        φ_α = atan2((1-α)·sin φ_R + α·sin φ_P,
                    (1-α)·cos φ_R + α·cos φ_P)

    Example: φ_R=10°, φ_P=350°, α=0.5 → ≈ 0° (not 180°).
    """
    sin_mix = (1.0 - alpha) * np.sin(angle_R) + alpha * np.sin(angle_P)
    cos_mix = (1.0 - alpha) * np.cos(angle_R) + alpha * np.cos(angle_P)
    return float(np.arctan2(sin_mix, cos_mix))
