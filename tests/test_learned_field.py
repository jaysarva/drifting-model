from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from learned_field import LearnedFieldBank, SetAttentionField


def _assert_antisymmetry(use_projection: bool):
    torch.manual_seed(0)
    field = SetAttentionField(
        input_dim=16,
        hidden_dim=32,
        projection_dim=24,
        score_hidden_dim=20,
        use_projection=use_projection,
    )

    x = torch.randn(7, 16)
    set_p = torch.randn(9, 16)
    set_q = torch.randn(11, 16)

    v_pq = field.compute_V_learned(x, set_p, set_q, mask_self_neg=False)
    v_qp = field.compute_V_learned(x, set_q, set_p, mask_self_neg=False)

    assert torch.allclose(v_pq, -v_qp, atol=1e-6, rtol=1e-5)


def test_learned_field_antisymmetry_with_projection():
    _assert_antisymmetry(use_projection=True)


def test_learned_field_antisymmetry_without_projection():
    _assert_antisymmetry(use_projection=False)


def test_learned_field_mask_self_singleton_is_finite():
    torch.manual_seed(1)
    field = SetAttentionField(
        input_dim=8,
        hidden_dim=16,
        projection_dim=12,
        score_hidden_dim=16,
        use_projection=True,
    )
    x = torch.randn(1, 8)
    v = field.compute_V_learned(x, x, x, mask_self_neg=True)
    assert torch.isfinite(v).all()


def test_learned_field_bank_scale_shapes():
    torch.manual_seed(2)
    bank = LearnedFieldBank(
        feature_dims=[8, 12],
        hidden_dim=16,
        projection_dim=10,
        score_hidden_dim=16,
        use_projection=True,
    )
    v0 = bank.compute_V_learned(
        scale_idx=0,
        feat_x=torch.randn(4, 8),
        feat_pos=torch.randn(6, 8),
        feat_neg=torch.randn(4, 8),
        mask_self_neg=True,
    )
    v1 = bank.compute_V_learned(
        scale_idx=1,
        feat_x=torch.randn(5, 12),
        feat_pos=torch.randn(7, 12),
        feat_neg=torch.randn(5, 12),
        mask_self_neg=True,
    )

    assert v0.shape == (4, 8)
    assert v1.shape == (5, 12)
