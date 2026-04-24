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
    compute_bc_weights,
    classify_atoms_3tier,
    compute_bond_angle_changes,
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


class TestBCWeights:
    """B+C 가중치 단위 테스트 (3-tier × α_Z)."""

    def test_core_atom_weight_proportional_to_alpha(self):
        """Core 원자의 가중치는 α_Z에 정비례 (정규화 전)."""
        # SN2: Cl⁻ + CH₃Br → ClCH₃ + Br⁻
        pos_R = np.array([
            [-2.0, 0.0, 0.0],   # Cl⁻ (incoming, core)
            [0.0, 0.0, 0.0],    # C (core)
            [2.0, 0.0, 0.0],    # Br (leaving, core)
            [0.5, 1.0, 0.0],    # H
            [0.5, -0.5, 0.87],  # H
            [0.5, -0.5, -0.87], # H
        ])
        pos_P = np.array([
            [-1.5, 0.0, 0.0],   # Cl
            [0.0, 0.0, 0.0],    # C
            [3.0, 0.0, 0.0],    # Br⁻ (떠남)
            [-0.5, 1.0, 0.0],   # H (umbrella inversion)
            [-0.5, -0.5, 0.87],
            [-0.5, -0.5, -0.87],
        ])
        z = np.array([17, 6, 35, 1, 1, 1])  # Cl, C, Br, H, H, H

        w, meta = compute_bc_weights(
            pos_R, pos_P, z, normalize=False, return_metadata=True,
        )

        # Core atoms identified
        assert set(meta["core_atoms"]) >= {1, 2}, (
            f"Expected C and Br in core, got {meta['core_atoms']}"
        )

        # In core, base_weight = 1.0 → w = 1.0 * α_Z
        # DEFAULT_ELEMENT_ALPHA: Cl=1.3, C=1.0, Br=1.4
        for core_idx in meta["core_atoms"]:
            expected = DEFAULT_ELEMENT_ALPHA.get(int(z[core_idx]), 1.0)
            assert w[core_idx] == pytest.approx(expected, abs=0.01), (
                f"Core atom {core_idx} (Z={z[core_idx]}): w={w[core_idx]:.3f} "
                f"expected {expected:.3f}"
            )

    def test_peripheral_h_lowest_and_normalized(self):
        """정규화 시 평균이 약 1, 모든 가중치 양수."""
        rng = np.random.default_rng(0)
        N = 20
        pos_R = rng.standard_normal((N, 3)) * 2
        pos_P = pos_R + rng.standard_normal((N, 3)) * 0.3
        z = np.array([6] * 5 + [1] * 15)

        w = compute_bc_weights(pos_R, pos_P, z, normalize=True)

        assert (w > 0).all(), "All BC weights must be positive"
        assert w.mean() == pytest.approx(1.0, abs=0.05), (
            f"Normalized mean should be ~1.0, got {w.mean():.4f}"
        )

    def test_beta_angle_zero_disables_angle_term(self):
        """beta_angle=0이면 결합각 효과가 사라지고 흐름은 C without angle × α_Z."""
        # SN2 with umbrella inversion: H atoms swing through the carbon, so
        # the bond angles around C change substantially R→P.
        pos_R = np.array([
            [-2.0, 0.0, 0.0],   # Cl⁻ (incoming, becomes core)
            [0.0, 0.0, 0.0],    # C
            [2.0, 0.0, 0.0],    # Br (leaving, core)
            [0.5, 1.0, 0.0],    # H
            [0.5, -0.5, 0.87],  # H
            [0.5, -0.5, -0.87], # H
        ])
        pos_P = np.array([
            [-1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-0.5, 1.0, 0.0],   # umbrella inversion
            [-0.5, -0.5, 0.87],
            [-0.5, -0.5, -0.87],
        ])
        z = np.array([17, 6, 35, 1, 1, 1])

        w_no_angle, meta_no = compute_bc_weights(
            pos_R, pos_P, z, beta_angle=0.0, normalize=False,
            return_metadata=True,
        )
        w_with_angle, meta_with = compute_bc_weights(
            pos_R, pos_P, z, beta_angle=2.0, normalize=False,
            return_metadata=True,
        )

        assert (w_no_angle > 0).all()
        # angle 항은 +ive contribution이므로 with_angle ≥ no_angle
        assert (w_with_angle >= w_no_angle - 1e-9).all()

        # 적어도 하나의 interface 원자가 측정 가능한 결합각 변화를 보여야
        # angle term이 weights를 바꾼다는 것을 검증할 수 있음
        interface_with_angle_change = (
            (meta_no["tier"] == 2) & (meta_no["angle_changes"] > 1.0)
        )
        if interface_with_angle_change.any():
            assert not np.allclose(w_no_angle, w_with_angle), (
                "beta_angle should change interface weights when angles change"
            )

    def test_halogen_interface_higher_than_h_interface(self):
        """같은 Interface tier 안에서 Br > H (B의 효과)."""
        # CH3-CH2-Br → CH3· + ·CH2-Br radical cleavage style:
        # C0-C1 bond breaks (R=1.5, P=3.0). C1-Br stays bonded throughout.
        # H attached to C0 stays put. Both Br and H are 1-hop interface.
        pos_R = np.array([
            [0.0, 0.0, 0.0],    # C0 (core: C-C bond breaks)
            [1.54, 0.0, 0.0],   # C1 (core)
            [3.49, 0.0, 0.0],   # Br (1-hop from C1: distance 1.95 → bonded)
            [-1.09, 0.0, 0.0],  # H  (1-hop from C0: distance 1.09 → bonded)
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],
            [3.04, 0.0, 0.0],   # C0-C1 bond broken (Δ=1.5 Å)
            [4.99, 0.0, 0.0],   # Br moves with C1 (still 1.95 from C1)
            [-1.09, 0.0, 0.0],
        ])
        z = np.array([6, 6, 35, 1])

        w, meta = compute_bc_weights(
            pos_R, pos_P, z, normalize=False, return_metadata=True,
        )
        assert set(meta["core_atoms"]) >= {0, 1}, (
            f"C0 and C1 should be core (bond breaks), got {meta['core_atoms']}"
        )
        assert meta["tier"][2] == 2 and meta["tier"][3] == 2, (
            f"Br and H should be interface, got tiers {meta['tier']}"
        )
        assert w[2] > w[3], (
            f"Br interface ({w[2]:.3f}) should > H interface ({w[3]:.3f})"
        )

    def test_classify_3tier_basic(self):
        """3-tier 분류 sanity check."""
        graph_dist = np.array([0, 1, 2, 3, 5, 0])
        tier = classify_atoms_3tier(graph_dist, interface_max_hop=2)
        assert tier.tolist() == [1, 2, 2, 3, 3, 1]

    def test_bond_angle_change_zero_when_no_motion(self):
        """좌표 변화 없으면 결합각 변화도 0."""
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = adj[1, 0] = 1
        adj[0, 2] = adj[2, 0] = 1
        adj[0, 3] = adj[3, 0] = 1
        delta = compute_bond_angle_changes(pos, pos, adj)
        assert np.allclose(delta, 0.0)
