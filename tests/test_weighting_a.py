"""Idea 1-A unit tests — graph-distance exponential decay weighting.

Verifies the BFS-based reactive core identification, graph-distance
calculation, and continuous weight function used in cb-A.
"""
import numpy as np
import pytest

from reactot.utils.weighting import (
    find_reactive_core_from_positions,
    build_adjacency_matrix,
    graph_distances_to_core,
    compute_continuous_weights,
    compute_weights_for_batch,
)


# Ethane → Ethylene (H–H elimination on C2 backbone).
# atomic numbers: [C, C, H, H, H, H, H, H]  (ethane); ethylene loses 2 H's.
ETHANE_POS = np.array([
    [ 0.000,  0.000,  0.000],   # C0
    [ 1.540,  0.000,  0.000],   # C1
    [-0.510,  1.020,  0.000],   # H2
    [-0.510, -0.510,  0.880],   # H3
    [-0.510, -0.510, -0.880],   # H4
    [ 2.050,  1.020,  0.000],   # H5
    [ 2.050, -0.510,  0.880],   # H6
    [ 2.050, -0.510, -0.880],   # H7
])
ETHYLENE_POS = np.array([
    [ 0.000,  0.000,  0.000],   # C0
    [ 1.330,  0.000,  0.000],   # C1  (shorter C=C)
    [-0.510,  1.020,  0.000],   # H2
    [-0.510, -0.510,  0.880],   # H3
    [-0.510, -0.510, -0.880],   # H4   (this H removed in product, but kept for matched ordering)
    [ 1.840,  1.020,  0.000],   # H5
    [ 1.840, -0.510,  0.880],   # H6
    [ 1.840, -0.510, -0.880],   # H7   (this H removed in product, kept for matched ordering)
])
ATOMIC_NUMBERS_C2H6 = np.array([6, 6, 1, 1, 1, 1, 1, 1])


class TestReactiveCoreIdentification:
    def test_c2h6_to_c2h4_core_includes_carbons(self):
        core = find_reactive_core_from_positions(
            ETHANE_POS, ETHYLENE_POS, ATOMIC_NUMBERS_C2H6
        )
        # The C–C bond contracts substantially (1.54 → 1.33 Å > 0.3 Å threshold).
        assert 0 in core, "C0 must be in the reactive core"
        assert 1 in core, "C1 must be in the reactive core"

    def test_no_change_returns_displacement_fallback(self):
        # If R == P exactly, the function falls back to the top-3 displaced
        # atoms — but with zero displacement, the function still returns
        # *some* set rather than crashing.
        core = find_reactive_core_from_positions(
            ETHANE_POS, ETHANE_POS, ATOMIC_NUMBERS_C2H6
        )
        assert isinstance(core, set), "fallback must return a set"
        assert len(core) > 0, "fallback must not return empty"


class TestAdjacencyAndBFS:
    def test_adjacency_symmetric(self):
        adj = build_adjacency_matrix(ETHANE_POS, ATOMIC_NUMBERS_C2H6)
        np.testing.assert_array_equal(adj, adj.T,
            err_msg="Adjacency matrix must be symmetric")

    def test_adjacency_no_self_loops(self):
        adj = build_adjacency_matrix(ETHANE_POS, ATOMIC_NUMBERS_C2H6)
        assert np.all(np.diag(adj) == 0), "No self-loops"

    def test_bfs_from_core_zero_distance(self):
        adj = build_adjacency_matrix(ETHANE_POS, ATOMIC_NUMBERS_C2H6)
        dist = graph_distances_to_core(adj, core_atoms={0, 1})
        assert dist[0] == 0 and dist[1] == 0, "Core atoms have distance 0"
        # All H atoms are 1 hop from one of the carbons.
        for i in range(2, 8):
            assert dist[i] == 1, f"H{i} should be 1 hop from core"


class TestContinuousWeights:
    def test_core_atoms_get_max_weight(self):
        dist = np.array([0, 0, 1, 2, 3, 4])
        w = compute_continuous_weights(dist, w_min=0.1, lambda_decay=2.0)
        np.testing.assert_allclose(w[:2], [1.0, 1.0])

    def test_w_min_floor_respected(self):
        # As distance → ∞, weight should approach (but not undershoot) w_min.
        dist = np.array([0, 1, 2, 5, 10, 100])
        w = compute_continuous_weights(dist, w_min=0.1, lambda_decay=2.0)
        assert np.all(w >= 0.1 - 1e-12), "Weights must respect w_min floor"
        assert w[-1] == pytest.approx(0.1, abs=1e-3), "Far atoms approach w_min"

    def test_monotone_decreasing(self):
        dist = np.arange(10).astype(float)
        w = compute_continuous_weights(dist, w_min=0.1, lambda_decay=2.0)
        assert np.all(np.diff(w) <= 1e-12), "Weights must monotonically decrease"

    def test_lambda_controls_decay_rate(self):
        dist = np.array([0, 1, 2, 3, 4])
        w_fast = compute_continuous_weights(dist, w_min=0.1, lambda_decay=1.0)
        w_slow = compute_continuous_weights(dist, w_min=0.1, lambda_decay=4.0)
        # Slower decay → larger weights at non-zero distances.
        assert np.all(w_slow[1:] >= w_fast[1:])


class TestEndToEndPipeline:
    def test_compute_weights_for_batch_shape(self):
        w = compute_weights_for_batch(
            ETHANE_POS, ETHYLENE_POS, ATOMIC_NUMBERS_C2H6,
            w_min=0.1, lambda_decay=2.0,
        )
        assert w.shape == (8,), f"Expected shape (8,), got {w.shape}"
        assert np.all(w >= 0.1) and np.all(w <= 1.0), "Weights in [w_min, 1]"

    def test_carbons_outweigh_hydrogens_for_cc_change(self):
        w = compute_weights_for_batch(
            ETHANE_POS, ETHYLENE_POS, ATOMIC_NUMBERS_C2H6,
            w_min=0.1, lambda_decay=2.0,
        )
        # Carbons are core (weight=1), Hs are 1 hop away (weight<1).
        assert w[0] > w[2], "C0 (core) > H2 (1-hop)"
        assert w[1] > w[5], "C1 (core) > H5 (1-hop)"
