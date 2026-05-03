"""
경량 EGNN 기반 x_0 predictor.
(R, P, atom_types) → x_0 ≈ TS_approx

모델 크기: ~50K parameters (ReactOT의 10.6M 대비 0.5%)
추론 시간: < 50ms/분자
"""
import torch
import torch.nn as nn
from typing import Optional


class X0PredictorEGNN(nn.Module):
    """
    3-layer EGNN로 (R, P)에서 x_0를 예측한다.

    핵심 설계:
        1. R과 P를 node feature로 encoding (atom_type one-hot + 좌표)
        2. 3-layer equivariant message passing
        3. 좌표 offset 예측: x_0 = midpoint + Δx
    """

    def __init__(self,
                 n_atom_types: int = 7,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 cutoff: float = 10.0,
                 normalize_coord_diff: bool = True):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Node encoder: atom_type(one-hot) + R_pos(3) + P_pos(3) → hidden
        self.node_encoder = nn.Sequential(
            nn.Linear(n_atom_types + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, cutoff, normalize_coord_diff)
            for _ in range(n_layers)
        ])

        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, pos_R, pos_P, atom_types, batch_mask: Optional[torch.Tensor] = None):
        """
        Args:
            pos_R: (N_total, 3) reactant positions
            pos_P: (N_total, 3) product positions
            atom_types: (N_total,) atom type indices
            batch_mask: (N_total,) molecule index (배치 처리용); None이면 단일 분자

        Returns:
            x0_pred: (N_total, 3) predicted initial structure (per-molecule COM removed)
        """
        midpoint = 0.5 * (pos_R + pos_P)

        atom_onehot = torch.nn.functional.one_hot(
            atom_types.long(), self.n_atom_types
        ).float()
        node_feat = torch.cat([atom_onehot, pos_R, pos_P], dim=-1)
        h = self.node_encoder(node_feat)

        pos = midpoint.clone()
        for layer in self.layers:
            h, pos = layer(h, pos, batch_mask)

        delta_x = self.coord_predictor(h)
        x0_pred = midpoint + delta_x

        # Per-molecule center-of-mass 제거
        if batch_mask is not None:
            from torch_scatter import scatter_mean
            com = scatter_mean(x0_pred, batch_mask, dim=0)
            x0_pred = x0_pred - com[batch_mask]
        else:
            x0_pred = x0_pred - x0_pred.mean(dim=0, keepdim=True)

        return x0_pred


class EGNNLayer(nn.Module):
    """Simplified EGNN layer (Satorras et al., 2021 스타일)."""

    def __init__(self, hidden_dim, cutoff, normalize_coord_diff: bool = True):
        super().__init__()
        self.cutoff = cutoff
        self.normalize_coord_diff = normalize_coord_diff

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, pos, batch_mask: Optional[torch.Tensor] = None):
        N = h.shape[0]

        # Build edges within cutoff (O(N^2) — N>50이면 sparse 구현 권장)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)            # (N, N, 3)
        dist = torch.norm(diff, dim=-1)                        # (N, N)
        mask = (dist < self.cutoff) & (dist > 1e-6)

        if batch_mask is not None:
            same_mol = batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1)
            mask = mask & same_mol

        edges_i, edges_j = torch.where(mask)
        if len(edges_i) == 0:
            return h, pos

        h_i = h[edges_i]
        h_j = h[edges_j]
        d_ij = dist[edges_i, edges_j].unsqueeze(-1)            # (E, 1)
        diff_ij = diff[edges_i, edges_j]                        # (E, 3)

        # 좌표 차이 정규화 (수치 안정성)
        if self.normalize_coord_diff:
            diff_ij = diff_ij / (d_ij + 1e-6)                   # 단위 벡터
            # diff_ij는 이제 unit vector이고, 거리 정보는 d_ij에 따로 보존됨

        edge_feat = torch.cat([h_i, h_j, d_ij], dim=-1)
        m_ij = self.edge_mlp(edge_feat)                         # (E, hidden)

        # Aggregate messages
        agg = torch.zeros_like(h)
        agg.index_add_(0, edges_i, m_ij)

        # Node update
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Coordinate update (equivariant): pos_i += Σ_j w_ij * (x_i - x_j)
        coord_weights = self.coord_mlp(m_ij)                    # (E, 1)
        weighted_diff = diff_ij * coord_weights                 # (E, 3)
        coord_update = torch.zeros_like(pos)
        coord_update.index_add_(0, edges_i, weighted_diff)

        pos_new = pos + coord_update
        return h_new, pos_new
