"""Idea 1-B unit tests — element-aware (per-Z polarizability) weighting."""
import numpy as np
import pytest

from reactot.utils.weighting import (
    ELEMENT_IMPORTANCE,
    get_element_importance,
    compute_element_aware_weights,
    compute_element_weights_for_batch,
)


class TestElementImportanceTable:
    def test_carbon_is_reference(self):
        assert ELEMENT_IMPORTANCE[6] == 1.0, "C must be the reference α=1.0"

    def test_polarizability_ordering_halogens(self):
        # Polarizability: Br > Cl > F (well-established trend)
        assert ELEMENT_IMPORTANCE[35] > ELEMENT_IMPORTANCE[17], "Br > Cl"
        assert ELEMENT_IMPORTANCE[17] > ELEMENT_IMPORTANCE[9], "Cl > F"
        assert ELEMENT_IMPORTANCE[9] > ELEMENT_IMPORTANCE[6], "F > C"

    def test_hydrogen_lowest(self):
        # H rotation is irrelevant for TS; should get the smallest weight.
        assert ELEMENT_IMPORTANCE[1] == min(ELEMENT_IMPORTANCE.values())

    def test_unknown_element_defaults_to_one(self):
        # Z=999 doesn't exist; lookup must not crash.
        assert get_element_importance(999) == 1.0


class TestElementAwareCombination:
    def test_combines_distance_with_alpha(self):
        # Two atoms: both at hop 0 (core), but H vs Br.
        graph_dist = np.array([0, 0])
        atomic_numbers = np.array([1, 35])
        w = compute_element_aware_weights(
            graph_dist, atomic_numbers,
            w_min=0.1, lambda_decay=2.0, normalize=False,
        )
        # Distance weight is 1.0 for both; α(H)=0.5, α(Br)=1.4
        np.testing.assert_allclose(w, [0.5, 1.4], rtol=1e-6)

    def test_normalization_preserves_relative_ratio(self):
        graph_dist = np.array([0, 1, 2])
        atomic_numbers = np.array([6, 6, 6])
        w_unnorm = compute_element_aware_weights(
            graph_dist, atomic_numbers,
            w_min=0.1, lambda_decay=2.0, normalize=False,
        )
        w_norm = compute_element_aware_weights(
            graph_dist, atomic_numbers,
            w_min=0.1, lambda_decay=2.0, normalize=True,
        )
        # Mean of normalized weights ≈ 1.0
        assert w_norm.mean() == pytest.approx(1.0, abs=1e-6)
        # Relative ordering preserved.
        np.testing.assert_array_equal(np.argsort(w_unnorm), np.argsort(w_norm))


class TestBatchPipeline:
    def test_halogen_substituted_methane_br_dominates(self):
        # CH3Br: C at origin, 3 H's, 1 Br.
        pos_R = np.array([
            [0.0, 0.0, 0.0],   # C
            [1.09, 0.0, 0.0],  # H
            [-0.36, 1.02, 0.0],  # H
            [-0.36, -0.51, 0.88],  # H
            [-1.94, 0.0, 0.0],  # Br (typical C–Br bond ~1.94 Å)
        ])
        pos_P = pos_R.copy()
        pos_P[4, 0] = -2.40  # Br dissociates a bit
        atomic_numbers = np.array([6, 1, 1, 1, 35])
        w = compute_element_weights_for_batch(
            pos_R, pos_P, atomic_numbers,
            w_min=0.1, lambda_decay=2.0,
        )
        # Br should have higher weight than any H.
        assert w[4] > max(w[1], w[2], w[3]), "Br weight must exceed all H weights"
