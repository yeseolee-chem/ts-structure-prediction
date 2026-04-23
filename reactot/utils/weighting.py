"""
Reactive core identification and continuous weighting functions.
References:
- FragmentFlow (2025): WLN ∪ Bemis-Murcko for reactive core
- Pipeline v4: exponential decay weighting
"""
import numpy as np
from collections import deque
from typing import Set, Optional
import torch

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    _RDKIT_AVAILABLE = False


def find_reactive_core_from_smiles(smiles_R: str, smiles_P: str) -> Set[int]:
    """
    R과 P의 SMILES에서 결합 변화(symmetric difference)를 탐지하여
    reactive core 원자 인덱스를 반환한다.

    Returns:
        core_atoms: set of atom indices that participate in bond changes
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError("rdkit is required for find_reactive_core_from_smiles")

    mol_R = Chem.MolFromSmiles(smiles_R)
    mol_P = Chem.MolFromSmiles(smiles_P)

    if mol_R is None or mol_P is None:
        # fallback: 모든 원자를 core로 취급
        return set(range(max(
            mol_R.GetNumAtoms() if mol_R else 0,
            mol_P.GetNumAtoms() if mol_P else 0
        )))

    def get_bonds(mol):
        bonds = set()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.add((min(i, j), max(i, j)))
        return bonds

    bonds_R = get_bonds(mol_R)
    bonds_P = get_bonds(mol_P)
    changed = bonds_R.symmetric_difference(bonds_P)

    core = set()
    for i, j in changed:
        core.add(i)
        core.add(j)

    return core


def find_reactive_core_from_positions(pos_R: np.ndarray, pos_P: np.ndarray,
                                       atomic_numbers: np.ndarray,
                                       bond_change_threshold: float = 0.3) -> Set[int]:
    """
    3D 좌표 기반 reactive core 식별 (SMILES 없을 때 fallback).
    R과 P에서 결합 길이 변화가 threshold 이상인 원자쌍의 원자를 core로 식별.

    Args:
        pos_R: (N, 3) reactant positions
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) atomic numbers
        bond_change_threshold: Å, 결합 길이 변화 임계값

    Returns:
        core_atoms: set of atom indices
    """
    from scipy.spatial.distance import cdist

    N = pos_R.shape[0]
    dist_R = cdist(pos_R, pos_R)
    dist_P = cdist(pos_P, pos_P)

    # 공유결합 반지름 기반 결합 판정 (1.3배 이내)
    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 1.02, 35: 1.20}

    core = set()
    for i in range(N):
        for j in range(i + 1, N):
            r_cov = covalent_radii.get(int(atomic_numbers[i]), 0.77) + \
                    covalent_radii.get(int(atomic_numbers[j]), 0.77)

            bonded_R = dist_R[i, j] < r_cov * 1.3
            bonded_P = dist_P[i, j] < r_cov * 1.3

            # 결합이 생기거나 끊어진 경우
            if bonded_R != bonded_P:
                core.add(i)
                core.add(j)
            # 결합 길이가 크게 변한 경우
            elif bonded_R and abs(dist_R[i, j] - dist_P[i, j]) > bond_change_threshold:
                core.add(i)
                core.add(j)

    # fallback: core가 비어있으면 가장 큰 변위를 보인 원자 3개를 core로
    if len(core) == 0:
        displacements = np.linalg.norm(pos_R - pos_P, axis=1)
        top_k = min(3, N)
        top_indices = np.argsort(displacements)[-top_k:]
        core = set(int(i) for i in top_indices.tolist())

    return core


def graph_distances_to_core(adj_matrix: np.ndarray, core_atoms: Set[int]) -> np.ndarray:
    """
    BFS로 각 원자에서 가장 가까운 core 원자까지의 그래프 거리를 계산한다.

    Args:
        adj_matrix: (N, N) adjacency matrix (binary)
        core_atoms: set of core atom indices

    Returns:
        dist: (N,) graph distances. core 원자는 0.
    """
    N = adj_matrix.shape[0]
    dist = np.full(N, np.inf)
    queue = deque()

    for c in core_atoms:
        dist[c] = 0
        queue.append(c)

    while queue:
        node = queue.popleft()
        for nb in range(N):
            if adj_matrix[node, nb] and dist[nb] > dist[node] + 1:
                dist[nb] = dist[node] + 1
                queue.append(nb)

    # inf 처리 (연결되지 않은 원자)
    max_dist = np.max(dist[dist < np.inf]) if np.any(dist < np.inf) else 0
    dist[dist == np.inf] = max_dist + 1

    return dist


def build_adjacency_matrix(positions: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    """
    공유결합 반지름 기반 인접 행렬 생성.

    Args:
        positions: (N, 3)
        atomic_numbers: (N,)

    Returns:
        adj: (N, N) binary adjacency matrix
    """
    from scipy.spatial.distance import cdist

    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 1.02, 35: 1.20}

    N = positions.shape[0]
    dist = cdist(positions, positions)
    adj = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            r_cov = covalent_radii.get(int(atomic_numbers[i]), 0.77) + \
                    covalent_radii.get(int(atomic_numbers[j]), 0.77)
            if dist[i, j] < r_cov * 1.3:
                adj[i, j] = 1
                adj[j, i] = 1

    return adj


def compute_continuous_weights(graph_dist: np.ndarray,
                                w_min: float = 0.1,
                                lambda_decay: float = 2.0) -> np.ndarray:
    """
    그래프 거리 기반 지수 감쇠 연속 가중치.

    w_i = w_min + (1 - w_min) * exp(-d_i / lambda)

    Args:
        graph_dist: (N,) graph distances to core
        w_min: minimum weight for most distant atoms (default 0.1)
        lambda_decay: decay length scale (default 2.0)

    Returns:
        weights: (N,) continuous weights in [w_min, 1.0]
    """
    return w_min + (1.0 - w_min) * np.exp(-graph_dist / lambda_decay)


def compute_weights_for_batch(pos_R: np.ndarray, pos_P: np.ndarray,
                               atomic_numbers: np.ndarray,
                               w_min: float = 0.1,
                               lambda_decay: float = 2.0) -> np.ndarray:
    """
    전체 파이프라인: 좌표 → reactive core → 그래프 거리 → 가중치.

    Args:
        pos_R: (N, 3) reactant positions
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) atomic numbers

    Returns:
        weights: (N,) continuous weights
    """
    # 1. Reactive core 식별
    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)

    # 2. 인접 행렬 (R 기준)
    adj = build_adjacency_matrix(pos_R, atomic_numbers)

    # 3. 그래프 거리
    graph_dist = graph_distances_to_core(adj, core)

    # 4. 연속 가중치
    weights = compute_continuous_weights(graph_dist, w_min, lambda_decay)

    return weights


# --- Torch 버전 (학습 시 GPU에서 사용) ---

def compute_weights_torch(graph_dist_tensor: torch.Tensor,
                           w_min: float = 0.1,
                           lambda_decay: float = 2.0) -> torch.Tensor:
    """
    Torch 텐서 버전의 연속 가중치.

    Args:
        graph_dist_tensor: (N,) or (B, N) graph distances

    Returns:
        weights: same shape as input
    """
    return w_min + (1.0 - w_min) * torch.exp(-graph_dist_tensor / lambda_decay)


# --- Idea 1-C: 3-Tier Hierarchical Weighting + Bond-angle correction ---
#
# Reference:
# - FragmentFlow (2025): KL divergence 분해로 core/attachment 오류 분리
# - ASM (Vermeeren et al., Nat. Protoc., 2020; DOI: 10.1038/s41596-019-0265-0):
#   interface 원자의 결합각 변화 → strain 기여

def classify_atoms_3tier(graph_dist: np.ndarray,
                          interface_max_hop: int = 2) -> np.ndarray:
    """
    원자를 3개 계층으로 분류한다.

    - Tier 1 (Core): graph_dist == 0 → 결합 변화에 직접 관여
    - Tier 2 (Interface): 0 < graph_dist <= interface_max_hop → core 인접
    - Tier 3 (Peripheral): graph_dist > interface_max_hop → 먼 치환기

    Args:
        graph_dist: (N,) graph distances to core
        interface_max_hop: Interface 계층의 최대 hop 수 (기본 2)

    Returns:
        tier: (N,) integer array, 값은 1, 2, 3
    """
    tier = np.full(graph_dist.shape, 3, dtype=int)
    tier[graph_dist == 0] = 1
    tier[(graph_dist > 0) & (graph_dist <= interface_max_hop)] = 2
    return tier


def compute_bond_angle_changes(pos_R: np.ndarray, pos_P: np.ndarray,
                                adj_matrix: np.ndarray) -> np.ndarray:
    """
    각 원자에 대해, 해당 원자를 꼭짓점으로 하는 모든 결합각의
    R→P 최대 변화량(degree)을 계산한다.

    Args:
        pos_R: (N, 3)
        pos_P: (N, 3)
        adj_matrix: (N, N) binary

    Returns:
        max_angle_change: (N,) degrees, 각 원자의 최대 결합각 변화
    """
    N = pos_R.shape[0]
    max_angle_change = np.zeros(N)

    for i in range(N):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        if len(neighbors) < 2:
            continue

        max_delta = 0.0
        # 모든 이웃 쌍에 대해 결합각 계산
        for k_idx in range(len(neighbors)):
            for l_idx in range(k_idx + 1, len(neighbors)):
                j = neighbors[k_idx]
                k = neighbors[l_idx]

                # R에서의 결합각 j-i-k
                v1_R = pos_R[j] - pos_R[i]
                v2_R = pos_R[k] - pos_R[i]
                cos_R = np.dot(v1_R, v2_R) / (np.linalg.norm(v1_R) * np.linalg.norm(v2_R) + 1e-10)
                angle_R = np.degrees(np.arccos(np.clip(cos_R, -1, 1)))

                # P에서의 결합각 j-i-k
                v1_P = pos_P[j] - pos_P[i]
                v2_P = pos_P[k] - pos_P[i]
                cos_P = np.dot(v1_P, v2_P) / (np.linalg.norm(v1_P) * np.linalg.norm(v2_P) + 1e-10)
                angle_P = np.degrees(np.arccos(np.clip(cos_P, -1, 1)))

                delta = abs(angle_R - angle_P)
                max_delta = max(max_delta, delta)

        max_angle_change[i] = max_delta

    return max_angle_change


def compute_hierarchical_weights(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    beta_angle: float = 1.0,
    interface_max_hop: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """
    3-Tier 계층적 가중치 + Interface 결합각 보정.

    - Core (Tier 1): w = 1.0
    - Interface (Tier 2): w = w_distance * (1 + β * |Δθ_max| / 180°)
    - Peripheral (Tier 3): w = w_min

    Args:
        pos_R, pos_P: (N, 3) reactant/product positions
        atomic_numbers: (N,) atomic numbers
        w_min: peripheral weight
        lambda_decay: distance decay for interface
        beta_angle: 결합각 보정 계수 (0이면 보정 없음)
        interface_max_hop: interface의 최대 hop 수
        normalize: per-molecule 정규화

    Returns:
        weights: (N,) hierarchical weights
    """
    # 1. Core 식별
    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)

    # 2. 인접 행렬 & 그래프 거리
    adj = build_adjacency_matrix(pos_R, atomic_numbers)
    graph_dist = graph_distances_to_core(adj, core)

    # 3. 3-Tier 분류. Core가 비어있으면(없음) 모든 원자를 Interface로 처리.
    if len(core) == 0:
        tier = np.full(graph_dist.shape, 2, dtype=int)
    else:
        tier = classify_atoms_3tier(graph_dist, interface_max_hop)

    # 4. 거리 기반 가중치
    distance_weights = compute_continuous_weights(graph_dist, w_min, lambda_decay)

    # 5. 결합각 변화 계산
    angle_changes = compute_bond_angle_changes(pos_R, pos_P, adj)

    # 6. 계층별 가중치 조합
    N = len(atomic_numbers)
    weights = np.zeros(N)

    for i in range(N):
        if tier[i] == 1:  # Core
            weights[i] = 1.0
        elif tier[i] == 2:  # Interface
            angle_factor = 1.0 + beta_angle * (angle_changes[i] / 180.0)
            weights[i] = distance_weights[i] * angle_factor
        else:  # Peripheral
            weights[i] = w_min

    # 7. 정규화 (per-molecule mean을 1로)
    if normalize:
        weights = weights / (weights.mean() + 1e-8)

    return weights


def compute_hierarchical_weights_for_batch(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    전체 파이프라인 래퍼. Idea 1-A의 compute_weights_for_batch()를 대체한다.
    """
    return compute_hierarchical_weights(pos_R, pos_P, atomic_numbers, **kwargs)


def weighted_rmsd(pos_TS: np.ndarray, pos_R_aligned: np.ndarray,
                   weights: np.ndarray) -> float:
    """
    Stage 5 Proxy_strain과 Stage 2 FM loss가 동일 가중치를 공유하도록
    설계된 weighted RMSD.

    Args:
        pos_TS: (N, 3)
        pos_R_aligned: (N, 3), Hungarian/Kabsch로 정렬된 R 좌표
        weights: (N,) per-atom weights

    Returns:
        rmsd: scalar, weighted RMSD
    """
    sq_dist = np.sum((pos_TS - pos_R_aligned) ** 2, axis=1)
    return float(np.sqrt(np.sum(weights * sq_dist) / (np.sum(weights) + 1e-8)))
