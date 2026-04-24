"""Unit tests for Idea 1-ABD prior dispatcher and hybrid weights."""
import numpy as np
import pytest
import torch

from reactot.utils.weighting import (
    compute_continuous_weights,
    compute_element_aware_weights,
    compute_hybrid_weights,
    compute_weights_for_batch,
    find_reactive_core_from_positions,
    get_prior_weights,
    graph_distances_to_core,
    build_adjacency_matrix,
    DEFAULT_ELEMENT_ALPHA,
    ELEMENT_IMPORTANCE,
)


# ---------------------------------------------------------------------------
# Building blocks (Idea 1-A + 1-B)
# ---------------------------------------------------------------------------


class TestReactiveCore:
    def test_core_identified_on_bond_break(self):
        # Linear C-H …H  → C … H-H (H atoms 1 and 2 become bonded)
        pos_R = np.array([
            [0.0, 0.0, 0.0],   # C
            [1.1, 0.0, 0.0],   # H (bonded to C)
            [3.0, 0.0, 0.0],   # H (far)
        ])
        pos_P = np.array([
            [0.0, 0.0, 0.0],   # C
            [2.0, 0.0, 0.0],   # H (moved away from C)
            [2.8, 0.0, 0.0],   # H (close to other H now)
        ])
        z = np.array([6, 1, 1])
        core = find_reactive_core_from_positions(pos_R, pos_P, z)
        assert 0 in core or len(core) > 0


class TestContinuousWeights:
    def test_w_min_floor(self):
        d = np.array([0.0, 100.0])
        w = compute_continuous_weights(d, w_min=0.1, lambda_decay=2.0)
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], 0.1, atol=1e-6)

    def test_monotone_decay(self):
        d = np.array([0.0, 1.0, 2.0, 3.0])
        w = compute_continuous_weights(d, w_min=0.05, lambda_decay=2.0)
        assert np.all(np.diff(w) < 0)


# ---------------------------------------------------------------------------
# Idea 1-ABD dispatcher — the point of this branch
# ---------------------------------------------------------------------------


class TestABDPrior:
    """A+B hybrid is D's KL prior. Dispatcher + math sanity checks."""

    def _cf3_geometry(self):
        """CF3-H reactant with one H poised to leave."""
        pos_R = np.array([
            [0.00, 0.00, 0.00],  # C
            [1.35, 0.00, 0.00],  # F
            [-0.65, 1.15, 0.00], # F
            [-0.65, -1.15, 0.00],# F
            [0.00, 0.00, 1.10],  # H (leaving)
        ])
        pos_P = pos_R.copy()
        pos_P[4, 2] = 2.50  # H pulled away → bond broken
        z = np.array([6, 9, 9, 9, 1])
        return pos_R, pos_P, z

    def test_dispatcher_returns_AB(self):
        """prior_scheme='AB' must match compute_hybrid_weights directly."""
        pos_R, pos_P, z = self._cf3_geometry()
        w_via_dispatcher = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        w_direct = compute_hybrid_weights(pos_R, pos_P, z)
        np.testing.assert_allclose(w_via_dispatcher, w_direct)

    def test_dispatcher_returns_A(self):
        """prior_scheme='A' must match compute_weights_for_batch (cb-D original)."""
        pos_R, pos_P, z = self._cf3_geometry()
        w_via_dispatcher = get_prior_weights(pos_R, pos_P, z, prior_scheme="A")
        w_direct = compute_weights_for_batch(pos_R, pos_P, z)
        np.testing.assert_allclose(w_via_dispatcher, w_direct)

    def test_dispatcher_unknown_scheme_raises(self):
        pos_R, pos_P, z = self._cf3_geometry()
        with pytest.raises(ValueError, match="Unknown prior_scheme"):
            get_prior_weights(pos_R, pos_P, z, prior_scheme="XYZ")

    def test_AB_prior_differs_from_A_prior(self):
        """A prior vs AB prior must differ once ≥2 elements are present."""
        pos_R, pos_P, z = self._cf3_geometry()
        w_A = get_prior_weights(pos_R, pos_P, z, prior_scheme="A")
        w_AB = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        assert not np.allclose(w_A, w_AB), \
            "A and AB priors should differ when multiple elements are present"

    def test_AB_prior_respects_element_ordering(self):
        """At equal graph distance, F must outweigh H in AB prior."""
        # All atoms 1-hop from core → same graph distance, only α_Z differs.
        pos_R = np.array([
            [0.0, 0.0, 0.0],  # C (core)
            [1.35, 0.0, 0.0], # F (1-hop)
            [-1.10, 0.0, 0.0],# H (1-hop)
        ])
        pos_P = pos_R.copy()
        pos_P[1, 0] = 2.5  # F leaves → makes C/F a reactive pair
        z = np.array([6, 9, 1])
        w_AB = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        # F (idx 1) should outweigh H (idx 2) because α_F=1.2 > α_H=0.5.
        assert w_AB[1] > w_AB[2], \
            f"F should outweigh H in AB prior (got F={w_AB[1]:.3f}, H={w_AB[2]:.3f})"

    def test_AB_prior_normalized(self):
        pos_R, pos_P, z = self._cf3_geometry()
        w = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        # Mean ≈ 1.0 (per-molecule normalization)
        assert np.isclose(w.mean(), 1.0, atol=1e-6)

    def test_AB_prior_kwargs_forwarded(self):
        """w_min / lambda_decay / element_alpha reach compute_hybrid_weights."""
        pos_R, pos_P, z = self._cf3_geometry()
        custom_alpha = {1: 2.0, 6: 1.0, 9: 0.5}  # inverted so H >> F
        w_custom = get_prior_weights(
            pos_R, pos_P, z, prior_scheme="AB", element_alpha=custom_alpha,
        )
        w_default = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        assert not np.allclose(w_custom, w_default), \
            "Custom element_alpha should change the weights"

    def test_KL_finite_with_AB_prior(self):
        """Mid-training learned≈sigmoid(0) against AB prior: KL is finite, ≥0."""
        N = 10
        rng = np.random.default_rng(0)
        pos_R = rng.normal(size=(N, 3))
        pos_P = pos_R + rng.normal(scale=0.3, size=(N, 3))
        z = rng.choice([1, 6, 8, 9, 17, 35], size=N)

        prior_np = get_prior_weights(pos_R, pos_P, z, prior_scheme="AB")
        prior = torch.tensor(prior_np, dtype=torch.float32)
        learned = torch.full((N,), 0.55)  # sigmoid(x)·0.9+0.1 ≈ 0.55 when x=0

        p = learned / learned.sum()
        q = prior / prior.sum()
        kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum()

        assert torch.isfinite(kl), "KL should be finite"
        assert kl.item() >= 0.0, "KL is non-negative"

    def test_default_alpha_matches_ELEMENT_IMPORTANCE(self):
        """ABD uses DEFAULT_ELEMENT_ALPHA which mirrors ELEMENT_IMPORTANCE."""
        for z, v in ELEMENT_IMPORTANCE.items():
            assert DEFAULT_ELEMENT_ALPHA.get(z) == v


# ---------------------------------------------------------------------------
# Hybrid returns metadata for Stage 5 Weighted RMSD reuse
# ---------------------------------------------------------------------------


class TestHybridMetadata:
    def test_metadata_keys(self):
        pos_R = np.array([
            [0.00, 0.00, 0.00],
            [1.10, 0.00, 0.00],
            [2.20, 0.00, 0.00],
        ])
        pos_P = pos_R + np.array([[0, 0, 0], [0.1, 0, 0], [0.5, 0, 0]])
        z = np.array([6, 1, 1])
        w, meta = compute_hybrid_weights(pos_R, pos_P, z, return_metadata=True)
        assert set(meta.keys()) >= {
            "core_atoms", "graph_dist", "w_distance", "alpha", "weights_raw",
        }
        assert meta["graph_dist"].shape == (3,)
        assert meta["alpha"].shape == (3,)
