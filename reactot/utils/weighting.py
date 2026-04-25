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


# ============================================================================
# Idea 1-B: Element-Aware Weighting
# ============================================================================
# Extends Idea 1-A by multiplying the graph-distance weight with a
# per-element coefficient α_Z, reflecting that atoms of different elements
# have different steric/electronic impact on TS geometry even at the same
# graph distance.
#
# References:
#   - Bondi, J. Phys. Chem. 1964, DOI: 10.1021/j100785a001 (vdW radii)
#   - Pearson, Inorg. Chem. 1988, DOI: 10.1021/ic00281a023 (HSAB / polarizability)

# 원소별 중요도 계수 α_Z
# 근거: vdW 반지름과 분극율에 비례하도록 설정
# H는 자유도가 높지만 TS 정확도 기여가 작으므로 감쇄
# 할로겐은 분극율 순서대로 Br > Cl > F
ELEMENT_IMPORTANCE = {
    1:  0.5,   # H  — 회전 자유도 높고 TS 기여 낮음
    6:  1.0,   # C  — 기준 원소
    7:  1.1,   # N  — lone pair, 전기음성도
    8:  1.1,   # O  — lone pair, 전기음성도
    9:  1.2,   # F  — 높은 전기음성도, 작은 크기
    16: 1.3,   # S  — 큰 원자, d-orbital 참여
    17: 1.3,   # Cl — 중간 분극율 (Halo8에서 Cl은 Z=17)
    35: 1.4,   # Br — 가장 큰 분극율, steric effect
}

# 주의: HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6, 35: 7}
# 이 매핑은 모델 입력용이고, weighting에서는 실제 원자번호(Z)를 사용


def get_element_importance(atomic_number: int) -> float:
    """원자번호로부터 중요도 계수를 반환한다."""
    return ELEMENT_IMPORTANCE.get(int(atomic_number), 1.0)


def compute_element_aware_weights(
    graph_dist: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    normalize: bool = True,
    custom_alpha: Optional[dict] = None,
) -> np.ndarray:
    """
    그래프 거리 + 원소별 계수를 결합한 가중치.

    w_i = [w_min + (1 - w_min) * exp(-d_i / λ)] * α_{Z_i}

    Args:
        graph_dist: (N,) graph distances to core
        atomic_numbers: (N,) atomic numbers
        w_min: minimum weight
        lambda_decay: decay length scale
        normalize: True이면 평균으로 나누어 scale 통일
        custom_alpha: 사용자 정의 α_Z (grid search 시 사용)

    Returns:
        weights: (N,) element-aware weights
    """
    # 1. 거리 기반 가중치 (Idea 1-A)
    distance_weights = compute_continuous_weights(graph_dist, w_min, lambda_decay)

    # 2. 원소별 계수
    alpha_dict = custom_alpha if custom_alpha is not None else ELEMENT_IMPORTANCE
    alpha = np.array(
        [alpha_dict.get(int(z), 1.0) for z in atomic_numbers],
        dtype=np.float64,
    )

    # 3. 결합
    weights = distance_weights * alpha

    # 4. 정규화: 분자 간 scale 통일
    if normalize:
        weights = weights / (weights.mean() + 1e-8)

    return weights


def compute_element_weights_for_batch(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    custom_alpha: Optional[dict] = None,
) -> np.ndarray:
    """
    전체 파이프라인: 좌표 → core → 거리 → 원소 가중치.

    Idea 1-A의 compute_weights_for_batch()를 대체한다.

    Args:
        pos_R: (N, 3) reactant positions
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) atomic numbers
        w_min: minimum weight
        lambda_decay: decay length scale
        custom_alpha: 사용자 정의 α_Z (grid search 시 사용)

    Returns:
        weights: (N,) element-aware continuous weights (normalized to mean ~1.0)
    """
    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)
    adj = build_adjacency_matrix(pos_R, atomic_numbers)
    graph_dist = graph_distances_to_core(adj, core)
    weights = compute_element_aware_weights(
        graph_dist, atomic_numbers,
        w_min=w_min, lambda_decay=lambda_decay,
        normalize=True, custom_alpha=custom_alpha,
    )
    return weights


# ============================================================================
# Idea 1-AB Hybrid: graph-distance + element-aware default weighting
# ============================================================================
# Baseline weighting for all subsequent experiments (1-C, 1-D compare against
# this). Identical math to compute_element_weights_for_batch but exposes a
# dedicated `compute_hybrid_weights` entry point with metadata return so
# Stage 5 (Weighted RMSD) can reuse the exact same weights that trained
# Stage 2 — pipeline consistency is the whole point of the hybrid.
#
# DEFAULT_ELEMENT_ALPHA mirrors ELEMENT_IMPORTANCE. It is a separate constant
# so hybrid-specific tuning does not accidentally shift 1-B's alpha values.

DEFAULT_ELEMENT_ALPHA = {
    1:  0.5,   # H
    6:  1.0,   # C (reference)
    7:  1.1,   # N
    8:  1.1,   # O
    9:  1.2,   # F
    16: 1.3,   # S
    17: 1.3,   # Cl
    35: 1.4,   # Br
}


def compute_hybrid_weights(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    element_alpha: Optional[dict] = None,
    normalize: bool = True,
    return_metadata: bool = False,
):
    """
    그래프 거리 + 원소 인지형 통합 가중치.

        w_i = [w_min + (1 - w_min) * exp(-d_i / λ)] * α_{Z_i}
        정규화: w̃_i = w_i / mean(w)

    Stage 2 FM loss와 Stage 5 Weighted RMSD에서 동일한 함수를 재사용한다.

    Args:
        pos_R: (N, 3) reactant positions
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) atomic numbers
        w_min: minimum weight for distant atoms
        lambda_decay: exponential decay scale (hops)
        element_alpha: custom element importance dict
            (None → DEFAULT_ELEMENT_ALPHA)
        normalize: per-molecule normalization (mean → 1.0)
        return_metadata: if True, return (weights, metadata_dict)

    Returns:
        weights: (N,) normalized hybrid weights
        metadata (optional): dict with core_atoms, graph_dist, w_distance,
            alpha, weights_raw
    """
    alpha_dict = element_alpha if element_alpha is not None else DEFAULT_ELEMENT_ALPHA

    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)
    adj = build_adjacency_matrix(pos_R, atomic_numbers)
    graph_dist = graph_distances_to_core(adj, core)

    w_distance = compute_continuous_weights(graph_dist, w_min, lambda_decay)
    alpha = np.array(
        [alpha_dict.get(int(z), 1.0) for z in atomic_numbers],
        dtype=np.float64,
    )

    weights_raw = w_distance * alpha
    weights = weights_raw.copy()
    if normalize and weights.mean() > 0:
        weights = weights / weights.mean()

    if return_metadata:
        metadata = {
            "core_atoms": sorted(int(i) for i in core),
            "graph_dist": graph_dist,
            "w_distance": w_distance,
            "alpha": alpha,
            "weights_raw": weights_raw,
        }
        return weights, metadata
    return weights


def compute_hybrid_weights_batch_torch(
    pos_R_batch: torch.Tensor,
    pos_P_batch: torch.Tensor,
    atomic_numbers_batch: torch.Tensor,
    batch_mask: torch.Tensor,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    element_alpha: Optional[dict] = None,
) -> torch.Tensor:
    """
    배치 내 모든 분자에 대해 하이브리드 가중치를 계산한다.

    학습 시 dataloader에서 사전 계산하는 것이 효율적이지만,
    on-the-fly 계산이 필요할 때 이 함수를 사용한다.

    Args:
        pos_R_batch: (total_atoms, 3)
        pos_P_batch: (total_atoms, 3)
        atomic_numbers_batch: (total_atoms,)
        batch_mask: (total_atoms,) molecule index per atom
        w_min, lambda_decay, element_alpha: forwarded to
            compute_hybrid_weights

    Returns:
        weights: (total_atoms,) float32 tensor on the input device
    """
    device = pos_R_batch.device
    n_molecules = int(batch_mask.max().item()) + 1 if batch_mask.numel() else 0

    chunks = []
    for mol_idx in range(n_molecules):
        mask = batch_mask == mol_idx
        pos_R = pos_R_batch[mask].detach().cpu().numpy().astype(np.float64)
        pos_P = pos_P_batch[mask].detach().cpu().numpy().astype(np.float64)
        z = (
            atomic_numbers_batch[mask]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.int64)
        )
        w = compute_hybrid_weights(
            pos_R, pos_P, z,
            w_min=w_min, lambda_decay=lambda_decay,
            element_alpha=element_alpha, normalize=True,
        )
        chunks.append(torch.tensor(w, dtype=torch.float32))

    if not chunks:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(chunks).to(device)


# ============================================================================
# Idea 1-C: 3-Tier Hierarchical Weighting + Bond-angle correction
# ============================================================================
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
    """
    N = pos_R.shape[0]
    max_angle_change = np.zeros(N)

    for i in range(N):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        if len(neighbors) < 2:
            continue

        max_delta = 0.0
        for k_idx in range(len(neighbors)):
            for l_idx in range(k_idx + 1, len(neighbors)):
                j = neighbors[k_idx]
                k = neighbors[l_idx]

                v1_R = pos_R[j] - pos_R[i]
                v2_R = pos_R[k] - pos_R[i]
                cos_R = np.dot(v1_R, v2_R) / (
                    np.linalg.norm(v1_R) * np.linalg.norm(v2_R) + 1e-10)
                angle_R = np.degrees(np.arccos(np.clip(cos_R, -1, 1)))

                v1_P = pos_P[j] - pos_P[i]
                v2_P = pos_P[k] - pos_P[i]
                cos_P = np.dot(v1_P, v2_P) / (
                    np.linalg.norm(v1_P) * np.linalg.norm(v2_P) + 1e-10)
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
    3-Tier 계층적 가중치 + Interface 결합각 보정 (Idea 1-C).

    - Core (Tier 1): w = 1.0
    - Interface (Tier 2): w = w_distance * (1 + β * |Δθ_max| / 180°)
    - Peripheral (Tier 3): w = w_min
    """
    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)
    adj = build_adjacency_matrix(pos_R, atomic_numbers)
    graph_dist = graph_distances_to_core(adj, core)

    if len(core) == 0:
        tier = np.full(graph_dist.shape, 2, dtype=int)
    else:
        tier = classify_atoms_3tier(graph_dist, interface_max_hop)

    distance_weights = compute_continuous_weights(graph_dist, w_min, lambda_decay)
    angle_changes = compute_bond_angle_changes(pos_R, pos_P, adj)

    N = len(atomic_numbers)
    weights = np.zeros(N)
    for i in range(N):
        if tier[i] == 1:
            weights[i] = 1.0
        elif tier[i] == 2:
            angle_factor = 1.0 + beta_angle * (angle_changes[i] / 180.0)
            weights[i] = distance_weights[i] * angle_factor
        else:
            weights[i] = w_min

    if normalize:
        weights = weights / (weights.mean() + 1e-8)
    return weights


# ============================================================================
# Idea 1-BC: Tier-Aware × Element-Aware Weighting
# ============================================================================
# B (element-aware α_Z) × C (3-tier + bond-angle) hybrid. A is already used
# inside C — do NOT multiply A again.
#
# Formula:
#   Core (tier=1):       w_i = 1.0 * α_{Z_i}
#   Interface (tier=2):  w_i = w_dist(d_i) * angle_factor(Δθ_i) * α_{Z_i}
#   Peripheral (tier=3): w_i = w_min * α_{Z_i}
#
# References:
#   - Bondi vdW radii (J. Phys. Chem. 1964, DOI: 10.1021/j100785a001)
#   - Pearson HSAB (Inorg. Chem. 1988, DOI: 10.1021/ic00281a023)
#   - ASM (Vermeeren et al., Nat. Protoc. 2020, DOI: 10.1038/s41596-019-0265-0)


def compute_bc_weights(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    interface_max_hop: int = 2,
    beta_angle: float = 1.0,
    element_alpha: Optional[dict] = None,
    normalize: bool = True,
    return_metadata: bool = False,
):
    """
    Tier 분류(C) × 원소 계수(B) 결합 가중치.

        Core:       w_i = 1.0 * α_{Z_i}
        Interface:  w_i = w_dist(d_i) * (1 + β * |Δθ_i|/180°) * α_{Z_i}
        Peripheral: w_i = w_min * α_{Z_i}

    Args:
        pos_R: (N, 3) reactant positions
        pos_P: (N, 3) product positions
        atomic_numbers: (N,) atomic numbers
        w_min: minimum (peripheral) base weight
        lambda_decay: exponential decay scale for interface (hops)
        interface_max_hop: tier-2 maximum hop distance
        beta_angle: bond-angle correction strength (0 → C without angle)
        element_alpha: per-element α_Z dict (None → DEFAULT_ELEMENT_ALPHA)
        normalize: per-molecule normalization (mean → 1.0)
        return_metadata: if True, return (weights, metadata_dict)

    Returns:
        weights: (N,) BC weights
        metadata (optional): debug dict with tier, alpha, base_weights, etc.
    """
    alpha_dict = element_alpha if element_alpha is not None else DEFAULT_ELEMENT_ALPHA

    # 1. Reactive core
    core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)

    # 2. Adjacency & graph distances
    adj = build_adjacency_matrix(pos_R, atomic_numbers)
    graph_dist = graph_distances_to_core(adj, core)

    # 3. C: 3-Tier (fallback to all-Interface when core is empty)
    if len(core) == 0:
        tier = np.full(graph_dist.shape, 2, dtype=int)
    else:
        tier = classify_atoms_3tier(graph_dist, interface_max_hop)

    # 4. C: distance & bond-angle (interface only)
    distance_weights = compute_continuous_weights(graph_dist, w_min, lambda_decay)
    angle_changes = compute_bond_angle_changes(pos_R, pos_P, adj)

    # 5. B: per-element α_Z
    alpha = np.array(
        [alpha_dict.get(int(z), 1.0) for z in atomic_numbers],
        dtype=np.float64,
    )

    # 6. Tier-wise base weight (C logic)
    N = len(atomic_numbers)
    base_weights = np.zeros(N)
    for i in range(N):
        if tier[i] == 1:
            base_weights[i] = 1.0
        elif tier[i] == 2:
            angle_factor = 1.0 + beta_angle * (angle_changes[i] / 180.0)
            base_weights[i] = distance_weights[i] * angle_factor
        else:
            base_weights[i] = w_min

    # 7. B: multiply by α_Z
    weights_raw = base_weights * alpha
    weights = weights_raw.copy()

    # 8. Normalize
    if normalize and weights.mean() > 1e-8:
        weights = weights / weights.mean()

    if return_metadata:
        metadata = {
            "core_atoms": sorted(int(i) for i in core),
            "graph_dist": graph_dist,
            "tier": tier,
            "distance_weights": distance_weights,
            "angle_changes": angle_changes,
            "alpha": alpha,
            "base_weights": base_weights,
            "weights_raw": weights_raw,
        }
        return weights, metadata

    return weights


def compute_bc_weights_for_batch(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    전체 파이프라인 래퍼. compute_hybrid_weights와 동일 시그니처(반환값,
    normalize 동작)이므로 dataset/trainer는 weighting_scheme 변수만 바꾸면
    즉시 교체 가능.
    """
    return compute_bc_weights(pos_R, pos_P, atomic_numbers, **kwargs)


def compute_bc_weights_batch_torch(
    pos_R_batch: torch.Tensor,
    pos_P_batch: torch.Tensor,
    atomic_numbers_batch: torch.Tensor,
    batch_mask: torch.Tensor,
    w_min: float = 0.1,
    lambda_decay: float = 2.0,
    interface_max_hop: int = 2,
    beta_angle: float = 1.0,
    element_alpha: Optional[dict] = None,
) -> torch.Tensor:
    """배치 내 모든 분자에 대해 BC 가중치를 계산 (compute_hybrid_weights_batch_torch와 동일 패턴)."""
    device = pos_R_batch.device
    n_molecules = int(batch_mask.max().item()) + 1 if batch_mask.numel() else 0

    chunks = []
    for mol_idx in range(n_molecules):
        mask = batch_mask == mol_idx
        pos_R = pos_R_batch[mask].detach().cpu().numpy().astype(np.float64)
        pos_P = pos_P_batch[mask].detach().cpu().numpy().astype(np.float64)
        z = (
            atomic_numbers_batch[mask]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.int64)
        )
        w = compute_bc_weights(
            pos_R, pos_P, z,
            w_min=w_min, lambda_decay=lambda_decay,
            interface_max_hop=interface_max_hop, beta_angle=beta_angle,
            element_alpha=element_alpha, normalize=True,
        )
        chunks.append(torch.tensor(w, dtype=torch.float32))

    if not chunks:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(chunks).to(device)


# ============================================================================
# Idea 1-BCD: prior dispatcher + BC-aware wrapper for D's KL regularization
# ============================================================================
# BCD combines the BC prior (tier × element + bond-angle) with D's learnable
# importance head. The BC formula is already defined above; the only new
# surface is a dispatcher so the dataset / trainer can flip between A, AB,
# B, C, BC priors without touching call sites.

def compute_hierarchical_weights_for_batch(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """C prior pipeline wrapper used by the dataset."""
    return compute_hierarchical_weights(pos_R, pos_P, atomic_numbers, **kwargs)


def compute_BC_prior_with_tier(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    **kwargs,
):
    """BC prior plus tier + alpha + base_weight metadata for decomposition.

    Used by scripts/analyze_bcd.py to decompose the learned importance into
    tier / element / residual components.

    Returns:
        (weights, metadata_dict). metadata_dict keys: ``tier`` (N,), ``alpha``
        (N,), ``base_weights`` (N,), ``graph_dist`` (N,).
    """
    weights, meta = compute_bc_weights(
        pos_R, pos_P, atomic_numbers, return_metadata=True, **kwargs
    )
    return weights, meta


def get_prior_weights(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    prior_scheme: str = "BC",
    **kwargs,
):
    """Dispatcher for per-atom prior schemes used by Idea 1-D / ABD / CD / BCD.

    Args:
        prior_scheme: one of
            - ``"A"``  : Idea 1-A   — graph-distance exp-decay only.
            - ``"AB"`` : Idea 1-AB  — A times per-element alpha_Z.
            - ``"B"``  : Idea 1-B   — element-aware alias of AB (same math).
            - ``"C"``  : Idea 1-C   — 3-tier hierarchical + bond-angle.
            - ``"BC"`` : Idea 1-BC  — tier × element + bond-angle (BCD default).

    Returns:
        weights: (N,) array. Normalized per-molecule to mean ~1.0 by default.
    """
    scheme = prior_scheme.upper()
    if scheme == "A":
        return compute_weights_for_batch(pos_R, pos_P, atomic_numbers, **kwargs)
    if scheme in ("AB", "A+B"):
        return compute_hybrid_weights(pos_R, pos_P, atomic_numbers, **kwargs)
    if scheme == "B":
        # B on its own collapses to AB with the default alpha_Z dict, since
        # the graph-distance kernel is shared by all schemes.
        return compute_hybrid_weights(pos_R, pos_P, atomic_numbers, **kwargs)
    if scheme == "C":
        return compute_hierarchical_weights(
            pos_R, pos_P, atomic_numbers, **kwargs
        )
    if scheme in ("BC", "B+C"):
        return compute_bc_weights(pos_R, pos_P, atomic_numbers, **kwargs)
    raise ValueError(f"Unknown prior_scheme: {prior_scheme!r}")


def weighted_rmsd(
    pos_TS: np.ndarray, pos_R_aligned: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted RMSD used by Stage 5 so the same weights train and evaluate."""
    sq_dist = np.sum((pos_TS - pos_R_aligned) ** 2, axis=1)
    return float(np.sqrt(np.sum(weights * sq_dist) / (np.sum(weights) + 1e-8)))
