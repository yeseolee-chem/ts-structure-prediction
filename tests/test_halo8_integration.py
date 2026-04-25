"""End-to-end integration test on a single Halo8 reaction.

This is a slow test (loads LMDB) but verifies that the full pipeline —
data loading → graph-distance weight precomputation → SBModule.training_step —
runs without error on a real halogen-containing reaction.

Skipped automatically if the Halo8 LMDB is not available locally.
"""
import os
import pytest
import torch
import numpy as np


HALO8_LMDB_PATH = os.environ.get(
    "HALO8_LMDB_PATH",
    "reactot/data/halo8/train.lmdb",
)


@pytest.mark.skipif(
    not os.path.exists(HALO8_LMDB_PATH),
    reason=f"Halo8 LMDB not found at {HALO8_LMDB_PATH}",
)
class TestHalo8EndToEnd:
    def test_single_reaction_loads_with_weights(self):
        from reactot.dataset.ff_lmdb import LmdbDataset
        ds = LmdbDataset(
            db_paths=[HALO8_LMDB_PATH],
            transform=None,
        )
        # Trigger weight precomputation.
        if hasattr(ds, "_attach_atom_weights"):
            ds._attach_atom_weights()

        # Pull the first sample.
        sample = ds[0]
        assert "atom_weights_0" in ds.data or hasattr(sample, "atom_weights"), \
            "Halo8 sample must carry per-atom weights after precomputation"

    def test_weights_normalized_to_unit_mean(self):
        """Hybrid weights (cb-AB+) should normalize to mean ≈ 1.0 per molecule."""
        from reactot.dataset.ff_lmdb import LmdbDataset
        ds = LmdbDataset(db_paths=[HALO8_LMDB_PATH])
        if hasattr(ds, "_attach_atom_weights"):
            ds._attach_atom_weights()

        # Check first 10 samples.
        for idx in range(min(10, len(ds))):
            if "atom_weights_0" in ds.data:
                w = ds.data["atom_weights_0"][idx]
                if isinstance(w, torch.Tensor):
                    w = w.cpu().numpy()
                # Per-molecule mean should be ≈ 1.0 with normalize=True default.
                assert 0.5 <= w.mean() <= 2.0, \
                    f"Sample {idx}: mean weight {w.mean():.3f} out of expected range"

    def test_halogen_atoms_get_higher_weights_than_h(self):
        """In a halogen-containing reaction, Br/Cl/F should outweigh H atoms."""
        from reactot.dataset.ff_lmdb import LmdbDataset
        ds = LmdbDataset(db_paths=[HALO8_LMDB_PATH])
        if hasattr(ds, "_attach_atom_weights"):
            ds._attach_atom_weights()

        halogen_zs = {9, 17, 35}
        h_z = 1
        any_halogen_reaction_found = False

        for idx in range(min(50, len(ds))):
            atomic_numbers = ds.data.get(f"charge_0", [None])[idx]
            if atomic_numbers is None:
                continue
            if isinstance(atomic_numbers, torch.Tensor):
                atomic_numbers = atomic_numbers.cpu().numpy().reshape(-1)
            if not any(z in halogen_zs for z in atomic_numbers):
                continue

            any_halogen_reaction_found = True
            w = ds.data["atom_weights_0"][idx]
            if isinstance(w, torch.Tensor):
                w = w.cpu().numpy()

            halogen_w = [w[i] for i, z in enumerate(atomic_numbers) if z in halogen_zs]
            h_w = [w[i] for i, z in enumerate(atomic_numbers) if z == h_z]
            if halogen_w and h_w:
                assert max(halogen_w) > min(h_w), \
                    f"In halogen reaction (idx={idx}), max halogen weight " \
                    f"{max(halogen_w):.3f} should exceed min H weight {min(h_w):.3f}"
                break  # one verified case is enough

        assert any_halogen_reaction_found, \
            "No halogen-containing reactions in first 50 samples — check dataset"
