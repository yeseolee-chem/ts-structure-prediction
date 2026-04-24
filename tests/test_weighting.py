"""
Weighted FM loss pipeline tests (Idea 1-AB hybrid).

Run: pytest tests/test_weighting.py -v
"""
import numpy as np
import pytest

from reactot.utils.weighting import (
    find_reactive_core_from_positions,
    build_adjacency_matrix,
    graph_distances_to_core,
    compute_continuous_weights,
    compute_hybrid_weights,
    DEFAULT_ELEMENT_ALPHA,
)
from reactot.analyze.rmsd import weighted_rmsd


class TestReactiveCoreIdentification:
    """Reactive core 식별 테스트."""

    def test_ethane_to_ethylene(self):
        """에탄 → 에틸렌: C-H 결합 절단, core = {C0, C1, H_leaving}."""
        pos_R = np.array([
            [0.0, 0.0, 0.0],   # C0
            [1.54, 0.0, 0.0],  # C1
            [0.0, 1.09, 0.0],  # H
            [0.0, -0.5, 0.9],  # H
            [0.0, -0.5, -0.9], # H (leaving)
            [1.54, 1.09, 0.0], # H
            [1.54, -0.5, 0.9], # H
            [1.54, -0.5, -0.9],# H (leaving)
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [0.0, 1.09, 0.0],
            [0.0, -1.09, 0.0],
            [5.0, 5.0, 5.0],   # H 떠남
            [1.34, 1.09, 0.0],
            [1.34, -1.09, 0.0],
            [5.0, 5.0, -5.0],  # H 떠남
        ])
        z = np.array([6, 6, 1, 1, 1, 1, 1, 1])

        core = find_reactive_core_from_positions(pos_R, pos_P, z)

        assert 0 in core or 1 in core, "Carbon atoms should be in core"
        assert len(core) >= 2, "Core should have at least 2 atoms"

    def test_halogen_substitution(self):
        """C-Cl → C-F 치환: Cl과 C가 core."""
        pos_R = np.array([
            [0.0, 0.0, 0.0],   # C
            [1.78, 0.0, 0.0],  # Cl
            [0.0, 1.09, 0.0],  # H
            [0.0, -0.5, 0.9],  # H
            [0.0, -0.5, -0.9], # H
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],   # C
            [1.39, 0.0, 0.0],  # F 자리에 온 원자 (결합 길이 변화)
            [0.0, 1.09, 0.0],  # H
            [0.0, -0.5, 0.9],  # H
            [0.0, -0.5, -0.9], # H
        ])
        z_R = np.array([6, 17, 1, 1, 1])

        core = find_reactive_core_from_positions(pos_R, pos_P, z_R)
        assert 0 in core, "C should be in core"
        assert 1 in core, "Halogen should be in core"


class TestGraphDistance:
    """BFS 거리 계산 테스트."""

    def test_linear_chain(self):
        """선형 사슬: 0-1-2-3, core={0}."""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ])
        core = {0}
        dist = graph_distances_to_core(adj, core)
        np.testing.assert_array_equal(dist, [0, 1, 2, 3])

    def test_multiple_core_atoms(self):
        """Multiple core atoms → BFS from any core atom."""
        adj = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ])
        core = {0, 4}
        dist = graph_distances_to_core(adj, core)
        # nearest to either endpoint
        np.testing.assert_array_equal(dist, [0, 1, 2, 1, 0])


class TestContinuousWeights:
    """지수 감쇠 가중치 테스트."""

    def test_core_gets_max_weight(self):
        dist = np.array([0, 1, 2, 3])
        w = compute_continuous_weights(dist, w_min=0.1, lambda_decay=2.0)
        assert w[0] == pytest.approx(1.0)
        assert all(w[i] > w[i + 1] for i in range(len(w) - 1))
        assert all(w >= 0.1)

    def test_w_min_bound(self):
        dist = np.array([100.0])
        w = compute_continuous_weights(dist, w_min=0.1)
        assert w[0] == pytest.approx(0.1, abs=0.01)


class TestHybridWeights:
    """Hybrid A+B integration 테스트."""

    def test_normalization(self):
        """정규화 후 mean ≈ 1.0."""
        rng = np.random.default_rng(0)
        pos_R = rng.standard_normal((10, 3)) * 2.0
        pos_P = pos_R + rng.standard_normal((10, 3)) * 0.5
        z = np.array([6, 6, 1, 1, 1, 1, 8, 1, 1, 1])

        w = compute_hybrid_weights(pos_R, pos_P, z, normalize=True)
        assert w.mean() == pytest.approx(1.0, abs=0.01)

    def test_halogen_gets_higher_weight_than_H(self):
        """Same graph hop: α_Br > α_H ⇒ Br 가중치 > H 가중치 (정규화 전)."""
        pos_R = np.array([
            [0.0, 0.0, 0.0],   # C (core)
            [1.9, 0.0, 0.0],   # Br (1-hop)
            [-1.09, 0.0, 0.0], # H (1-hop)
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],   # Br 멀어짐 → bond break
            [-1.09, 0.0, 0.0],
        ])
        z = np.array([6, 35, 1])

        w = compute_hybrid_weights(pos_R, pos_P, z, normalize=False)
        # α_Br(1.4) >> α_H(0.5), Br는 core일 수 있고 H는 1-hop
        assert w[1] > w[2], "Br should have higher weight than H"

    def test_metadata_return(self):
        pos_R = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
        pos_P = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        z = np.array([6, 1])

        w, meta = compute_hybrid_weights(
            pos_R, pos_P, z, return_metadata=True,
        )
        assert w.shape == (2,)
        assert set(meta.keys()) == {
            "core_atoms", "graph_dist", "w_distance", "alpha", "weights_raw",
        }
        assert meta["alpha"][0] == DEFAULT_ELEMENT_ALPHA[6]
        assert meta["alpha"][1] == DEFAULT_ELEMENT_ALPHA[1]

    def test_custom_element_alpha_overrides_default(self):
        pos_R = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
        pos_P = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        z = np.array([6, 1])

        custom = {6: 1.0, 1: 10.0}  # H heavily upweighted
        w = compute_hybrid_weights(
            pos_R, pos_P, z, element_alpha=custom, normalize=False,
        )
        _, meta = compute_hybrid_weights(
            pos_R, pos_P, z, element_alpha=custom,
            normalize=False, return_metadata=True,
        )
        assert meta["alpha"][1] == 10.0


class TestWeightedRMSD:
    """Stage 5 weighted RMSD 테스트."""

    def test_zero_deviation_zero_rmsd(self):
        pos = np.zeros((5, 3))
        w = np.ones(5)
        assert weighted_rmsd(pos, pos, w) == pytest.approx(0.0)

    def test_uniform_weights_match_unweighted(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((8, 3))
        b = rng.standard_normal((8, 3))
        w = np.ones(8)
        wr = weighted_rmsd(a, b, w)
        unweighted = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        assert wr == pytest.approx(unweighted, rel=1e-6)

    def test_weight_focuses_error(self):
        """Only one atom contributes error → weights that zero it out drop RMSD."""
        a = np.zeros((3, 3))
        b = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])  # atom 1 has the error
        w_all = np.array([1.0, 1.0, 1.0])
        w_mask = np.array([1.0, 0.0, 1.0])  # ignore atom 1
        assert weighted_rmsd(a, b, w_all) > weighted_rmsd(a, b, w_mask)
        assert weighted_rmsd(a, b, w_mask) == pytest.approx(0.0, abs=1e-6)
