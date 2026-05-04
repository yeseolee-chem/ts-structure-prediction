"""
Lightweight EGNN-based x_0 predictor (Idea 2-F v2).

(R, P, atom_types) → x_0 ≈ TS_approx.
Model size ~50K parameters, inference < 50 ms / molecule.
"""
import torch
import torch.nn as nn
from typing import Optional


class X0PredictorEGNN(nn.Module):
    """
    3-layer EGNN that predicts x_0 = midpoint + Δx from (R, P).
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

    def forward(self, pos_R, pos_P, atom_types,
                batch_mask: Optional[torch.Tensor] = None):
        """
        Returns (N_total, 3) predicted initial structure (CoG removed
        per-molecule when batch_mask is supplied).
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

        if batch_mask is not None:
            try:
                from torch_scatter import scatter_mean
                com = scatter_mean(x0_pred, batch_mask, dim=0)
                x0_pred = x0_pred - com[batch_mask]
            except ImportError:  # pragma: no cover
                # Manual per-molecule mean removal (fallback when
                # torch_scatter is unavailable).
                unique = torch.unique(batch_mask)
                for u in unique:
                    mask = batch_mask == u
                    x0_pred[mask] = x0_pred[mask] - x0_pred[mask].mean(0, keepdim=True)
        else:
            x0_pred = x0_pred - x0_pred.mean(dim=0, keepdim=True)

        return x0_pred


class EGNNLayer(nn.Module):
    """Simplified EGNN layer (Satorras et al., 2021)."""

    def __init__(self, hidden_dim, cutoff,
                 normalize_coord_diff: bool = True):
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

        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = (dist < self.cutoff) & (dist > 1e-6)

        if batch_mask is not None:
            same_mol = batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1)
            mask = mask & same_mol

        edges_i, edges_j = torch.where(mask)
        if len(edges_i) == 0:
            return h, pos

        h_i = h[edges_i]
        h_j = h[edges_j]
        d_ij = dist[edges_i, edges_j].unsqueeze(-1)
        diff_ij = diff[edges_i, edges_j]

        if self.normalize_coord_diff:
            diff_ij = diff_ij / (d_ij + 1e-6)

        edge_feat = torch.cat([h_i, h_j, d_ij], dim=-1)
        m_ij = self.edge_mlp(edge_feat)

        agg = torch.zeros_like(h)
        agg.index_add_(0, edges_i, m_ij)

        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        coord_weights = self.coord_mlp(m_ij)
        weighted_diff = diff_ij * coord_weights
        coord_update = torch.zeros_like(pos)
        coord_update.index_add_(0, edges_i, weighted_diff)

        pos_new = pos + coord_update
        return h_new, pos_new
