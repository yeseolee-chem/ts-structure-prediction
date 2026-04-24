"""Unit tests for Idea 1-CD: dispatcher, C prior + tier cache, angle effect.

Run:  pytest tests/test_weighting.py -v
"""
import numpy as np
import pytest

from reactot.utils.weighting import (
    get_prior_weights,
    compute_hierarchical_weights,
    compute_C_prior_with_tier,
    compute_weights_for_batch,
)


class TestCDPrior:
    """CD prior = compute_hierarchical_weights driven via the dispatcher."""

    def test_dispatcher_returns_C(self):
        """prior_scheme='C' is equivalent to compute_hierarchical_weights."""
        rng = np.random.default_rng(0)
        pos_R = rng.standard_normal((8, 3))
        pos_P = pos_R + 0.5 * rng.standard_normal((8, 3))
        z = np.array([6, 6, 1, 1, 1, 1, 8, 1])

        w_via_dispatcher = get_prior_weights(pos_R, pos_P, z, prior_scheme="C")
        w_direct = compute_hierarchical_weights(pos_R, pos_P, z)

        np.testing.assert_allclose(w_via_dispatcher, w_direct)

    def test_dispatcher_returns_A(self):
        """prior_scheme='A' delegates to the flat exp-decay prior."""
        rng = np.random.default_rng(1)
        pos_R = rng.standard_normal((6, 3))
        pos_P = pos_R + 0.3 * rng.standard_normal((6, 3))
        z = np.array([6, 6, 1, 1, 1, 1])

        w_dispatch = get_prior_weights(pos_R, pos_P, z, prior_scheme="A")
        w_direct = compute_weights_for_batch(pos_R, pos_P, z)

        np.testing.assert_allclose(w_dispatch, w_direct)

    def test_dispatcher_rejects_unknown_scheme(self):
        pos_R = np.zeros((3, 3))
        pos_P = np.zeros((3, 3))
        z = np.array([6, 1, 1])
        with pytest.raises(ValueError, match="Unknown prior_scheme"):
            get_prior_weights(pos_R, pos_P, z, prior_scheme="ZZZ")

    def test_compute_C_prior_with_tier_returns_tier(self):
        """CD helper hands back (weights, tier) with tier in {1, 2, 3}."""
        rng = np.random.default_rng(2)
        pos_R = rng.standard_normal((10, 3))
        pos_P = pos_R + 0.4 * rng.standard_normal((10, 3))
        z = np.array([6, 6, 6, 1, 1, 1, 1, 8, 1, 1])

        w, tier = compute_C_prior_with_tier(pos_R, pos_P, z)
        assert w.shape == (10,)
        assert tier.shape == (10,)
        assert set(np.unique(tier).tolist()).issubset({1, 2, 3})

    def test_C_prior_peripheral_equals_w_min(self):
        """Without normalize, all Tier-3 atoms sit at exactly w_min."""
        N = 20
        # Make a sparse pair: only atoms 0 and 1 move, rest identical → most
        # atoms end up peripheral for large interface_max_hop boundaries.
        rng = np.random.default_rng(3)
        pos_R = 5.0 * rng.standard_normal((N, 3))
        pos_P = pos_R.copy()
        pos_P[0] += 1.5
        pos_P[1] += 1.5
        z = np.full(N, 6, dtype=np.int64)

        w, meta = compute_hierarchical_weights(
            pos_R, pos_P, z, normalize=False, return_metadata=True
        )
        tier = meta["tier"]

        peripheral_mask = tier == 3
        if peripheral_mask.sum() > 1:
            peripheral_w = w[peripheral_mask]
            assert np.allclose(peripheral_w, peripheral_w[0]), (
                "All peripheral atoms should have identical w_min without normalization"
            )

    def test_C_prior_angle_correction_has_effect(self):
        """beta_angle > 0 changes the weights vs beta_angle = 0.

        Construction: C0 moves while C1, C2 are stationary. C0-C1 bond
        length changes enough (>0.3 Å) to put {0, 1} in the reactive core,
        while C1-C2 stays fixed. C2 therefore ends up in the Interface tier
        (hop = 1). H3 is the only neighbor of C2 besides C1, and H3 is
        displaced so its distance to C2 changes by < 0.3 Å (keeping C2
        *not* in core) but the C1-C2-H3 angle changes substantially. The
        interface atom's weight must therefore respond to beta_angle.
        """
        pos_R = np.array(
            [
                [0.0, 0.0, 0.0],   # C0 — will move (becomes core)
                [1.5, 0.0, 0.0],   # C1 — stationary (becomes core via bond to C0)
                [3.0, 0.0, 0.0],   # C2 — stationary, interface
                [3.0, 1.0, 0.0],   # H3 — pendant on C2, angle-only change
            ],
            dtype=np.float64,
        )
        pos_P = np.array(
            [
                [0.5, 0.0, 0.0],   # C0-C1 bond: 1.5 → 1.0 (Δ=0.5, flags core)
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],   # C1-C2 bond unchanged → C2 stays out of core
                [3.5, 1.0, 0.2],   # C2-H3 bond: 1.0 → ~1.14 (Δ<0.3, stays interface)
            ],
            dtype=np.float64,
        )
        z = np.array([6, 6, 6, 1])

        w_with = compute_hierarchical_weights(
            pos_R, pos_P, z, beta_angle=2.0, normalize=False
        )
        w_without = compute_hierarchical_weights(
            pos_R, pos_P, z, beta_angle=0.0, normalize=False
        )

        assert not np.allclose(w_with, w_without), (
            f"beta_angle > 0 should perturb interface weights. "
            f"w_with={w_with}, w_without={w_without}"
        )
