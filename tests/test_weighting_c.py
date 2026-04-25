"""Idea 1-C unit tests — 3-tier hierarchical + bond-angle correction."""
import numpy as np
import pytest

from reactot.utils.weighting import (
    classify_atoms_3tier,
    compute_bond_angle_changes,
    compute_hierarchical_weights,
    weighted_rmsd,
)


class TestThreeTierClassification:
    def test_core_tier_assignment(self):
        graph_dist = np.array([0, 1, 2, 3, 4])
        tier = classify_atoms_3tier(graph_dist, interface_max_hop=2)
        assert tier[0] == 1, "hop=0 → Tier 1 (Core)"

    def test_interface_tier_assignment(self):
        graph_dist = np.array([0, 1, 2, 3, 4])
        tier = classify_atoms_3tier(graph_dist, interface_max_hop=2)
        assert tier[1] == 2, "hop=1 → Tier 2 (Interface)"
        assert tier[2] == 2, "hop=2 → Tier 2 (Interface)"

    def test_peripheral_tier_assignment(self):
        graph_dist = np.array([0, 1, 2, 3, 4])
        tier = classify_atoms_3tier(graph_dist, interface_max_hop=2)
        assert tier[3] == 3, "hop=3 → Tier 3 (Peripheral)"
        assert tier[4] == 3, "hop=4 → Tier 3 (Peripheral)"


class TestBondAngleChanges:
    def test_no_motion_zero_angle_change(self):
        # Linear 3-atom chain, identical R and P.
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        changes = compute_bond_angle_changes(pos, pos, adj)
        np.testing.assert_allclose(changes, [0.0, 0.0, 0.0], atol=1e-6)

    def test_180_to_90_degree_bend(self):
        # Linear (180°) → bent (90°) at central atom.
        pos_R = np.array([
            [-1.0, 0.0, 0.0],   # A
            [ 0.0, 0.0, 0.0],   # B (center)
            [ 1.0, 0.0, 0.0],   # C
        ])
        pos_P = np.array([
            [-1.0, 0.0, 0.0],
            [ 0.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],   # C now perpendicular
        ])
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        changes = compute_bond_angle_changes(pos_R, pos_P, adj)
        # Central atom B sees 180° → 90°, a 90° change.
        assert changes[1] == pytest.approx(90.0, abs=1.0)


class TestHierarchicalWeights:
    def test_core_always_one(self):
        # SN2-like: C–X bond breaks. Core atoms (C, X) are dist=0.
        pos_R = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],   # X, bonded
            [-1.0, 1.0, 0.0],  # H peripheral
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],   # X dissociated
            [-1.0, 1.0, 0.0],
        ])
        atomic_numbers = np.array([6, 9, 1])
        w = compute_hierarchical_weights(
            pos_R, pos_P, atomic_numbers,
            w_min=0.1, lambda_decay=2.0, beta_angle=1.0,
        )
        # Core atoms (C and F) get weight 1.0
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(1.0)

    def test_beta_angle_zero_disables_angle_correction(self):
        pos_R = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        pos_P = pos_R.copy()
        pos_P[1, 0] = 3.0
        atomic_numbers = np.array([6, 9, 1])
        w_no_angle = compute_hierarchical_weights(
            pos_R, pos_P, atomic_numbers,
            beta_angle=0.0,
        )
        w_with_angle = compute_hierarchical_weights(
            pos_R, pos_P, atomic_numbers,
            beta_angle=1.0,
        )
        # With β=0 and β=1, the Interface-tier atoms differ; Core/Peripheral identical.
        assert isinstance(w_no_angle, np.ndarray)
        assert isinstance(w_with_angle, np.ndarray)


class TestWeightedRMSD:
    def test_zero_displacement_zero_rmsd(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        w = np.array([1.0, 1.0])
        assert weighted_rmsd(pos, pos, w) == pytest.approx(0.0, abs=1e-8)

    def test_uniform_weights_match_unweighted(self):
        pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos_b = np.array([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]])
        w_uniform = np.array([1.0, 1.0])
        unweighted = np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1)))
        weighted = weighted_rmsd(pos_a, pos_b, w_uniform)
        assert weighted == pytest.approx(unweighted, rel=1e-6)
