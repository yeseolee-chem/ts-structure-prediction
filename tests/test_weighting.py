"""Unit tests for reactot.utils.weighting (Idea 1-BCD)."""
import numpy as np
import pytest

from reactot.utils import (
    compute_BC_prior_with_tier,
    compute_bc_weights,
    get_prior_weights,
)


def _cnh3cl_displaced():
    """CH3Cl -> CH3 + Cl dissociation along the C-Cl axis.

    Returns pos_R, pos_P, atomic_numbers. The Cl atom translates 1.2 A farther
    from C in the product, so the reactive core should pick up {C, Cl}.
    """
    pos_R = np.array(
        [
            [0.0, 0.0, 0.0],    # C
            [1.08, 0.0, 0.0],   # H
            [-0.54, 0.94, 0.0], # H
            [-0.54, -0.94, 0.0],# H
            [0.0, 0.0, 1.78],   # Cl
        ]
    )
    pos_P = pos_R.copy()
    pos_P[4, 2] += 1.2  # Cl moves away
    z = np.array([6, 1, 1, 1, 17])
    return pos_R, pos_P, z


class TestBCDPrior:
    def test_returns_weights_and_metadata(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        w, meta = compute_BC_prior_with_tier(pos_R, pos_P, z)
        assert w.shape == (5,)
        assert np.all(np.isfinite(w))
        for key in ("tier", "alpha", "base_weights", "graph_dist"):
            assert key in meta, f"missing metadata key {key}"
            assert meta[key].shape[0] == 5

    def test_tier_labels_in_valid_range(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        _, meta = compute_BC_prior_with_tier(pos_R, pos_P, z)
        assert set(np.unique(meta["tier"]).tolist()).issubset({1, 2, 3})

    def test_reactive_atoms_get_core_tier(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        _, meta = compute_BC_prior_with_tier(pos_R, pos_P, z)
        tier = meta["tier"]
        # C (idx 0) and Cl (idx 4) are the bond-breaking pair — both Core.
        assert tier[0] == 1
        assert tier[4] == 1

    def test_dispatcher_routes_to_bc(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        w_direct = compute_bc_weights(pos_R, pos_P, z)
        w_dispatch = get_prior_weights(pos_R, pos_P, z, prior_scheme="BC")
        np.testing.assert_allclose(w_direct, w_dispatch)

    def test_dispatcher_rejects_unknown_scheme(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        with pytest.raises(ValueError):
            get_prior_weights(pos_R, pos_P, z, prior_scheme="Q")

    def test_alpha_follows_element_lookup(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        _, meta = compute_BC_prior_with_tier(pos_R, pos_P, z)
        # α for H < α for C < α for Cl (softer / more important atoms rank
        # higher in Pearson HSAB × Bondi). We don't assert exact values here;
        # just that the per-element lookup is monotone-ish.
        alpha = meta["alpha"]
        assert alpha[1] <= alpha[0] <= alpha[4]

    def test_weights_mean_is_normalized(self):
        pos_R, pos_P, z = _cnh3cl_displaced()
        w = compute_bc_weights(pos_R, pos_P, z, normalize=True)
        # per-molecule mean should be ~1 after normalization
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-6)
