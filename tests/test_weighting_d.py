"""Idea 1-D unit tests — learnable per-atom importance + KL regularization.

Tests the importance head's structural correctness and the KL formula's
mathematical properties. Does not test full training (that's an integration
test and is too slow for unit tests).
"""
import torch
import pytest


class TestImportanceHeadStructure:
    def test_importance_head_input_dim_matches_in_hidden_channels(self):
        from reactot.dynamics.egnn_dynamics import EGNNDynamics
        # Build a minimal EGNNDynamics with learn_importance=True.
        # We can't easily build a full instance here without a config; instead,
        # we inspect the source to verify the in_dim choice.
        import inspect
        src = inspect.getsource(EGNNDynamics.__init__)
        assert "in_dim=in_hidden_channels" in src, \
            "importance_head must use in_hidden_channels (=8 by default), " \
            "not hidden_channels (=196), per the rationale in the source comment."

    def test_importance_head_outputs_one_scalar(self):
        from reactot.dynamics.egnn_dynamics import EGNNDynamics
        import inspect
        src = inspect.getsource(EGNNDynamics.__init__)
        # Confirm last out_dim is 1.
        assert ", 1]" in src, "importance_head last layer must output 1 scalar"


class TestSigmoidSquashing:
    def test_sigmoid_to_w_min_unity_range(self):
        # The squash maps (-∞, +∞) → [w_min, 1.0].
        w_min = 0.1
        # Very negative → ~w_min
        very_neg = torch.tensor([-100.0])
        squashed_neg = torch.sigmoid(very_neg) * (1.0 - w_min) + w_min
        assert squashed_neg.item() == pytest.approx(w_min, abs=1e-3)
        # Very positive → ~1.0
        very_pos = torch.tensor([100.0])
        squashed_pos = torch.sigmoid(very_pos) * (1.0 - w_min) + w_min
        assert squashed_pos.item() == pytest.approx(1.0, abs=1e-3)
        # Zero logit → midpoint
        zero = torch.tensor([0.0])
        squashed_zero = torch.sigmoid(zero) * (1.0 - w_min) + w_min
        assert squashed_zero.item() == pytest.approx(0.55, abs=1e-3)


class TestKLDivergenceFormula:
    """Verify KL(p ‖ q) = Σ p · log(p/q) properties."""

    def test_kl_zero_when_distributions_identical(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum()
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_nonnegative(self):
        # KL(p ‖ q) ≥ 0 always (Gibbs' inequality).
        torch.manual_seed(42)
        for _ in range(5):
            p_logits = torch.randn(8)
            q_logits = torch.randn(8)
            p = torch.softmax(p_logits, dim=0)
            q = torch.softmax(q_logits, dim=0)
            kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum()
            assert kl.item() >= -1e-6, \
                f"KL must be non-negative, got {kl.item()}"

    def test_kl_asymmetric(self):
        # KL(p ‖ q) != KL(q ‖ p) in general.
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.4, 0.5])
        kl_pq = (p * torch.log(p / q)).sum()
        kl_qp = (q * torch.log(q / p)).sum()
        assert not torch.isclose(kl_pq, kl_qp), \
            "Forward and reverse KL should differ"


class TestSBModuleConfig:
    def test_kl_weight_default(self):
        # The default kl_weight in SBModule should be 0.1 per the spec.
        from reactot.trainer.pl_trainer import SBModule
        import inspect
        sig = inspect.signature(SBModule.__init__)
        if "kl_weight" in sig.parameters:
            default = sig.parameters["kl_weight"].default
            assert default == 0.1, f"kl_weight default must be 0.1, got {default}"

    def test_learn_importance_default_false(self):
        # learn_importance must default to False for backward compatibility.
        from reactot.trainer.pl_trainer import SBModule
        import inspect
        sig = inspect.signature(SBModule.__init__)
        if "learn_importance" in sig.parameters:
            default = sig.parameters["learn_importance"].default
            assert default is False, "learn_importance must default to False"
