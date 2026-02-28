"""
Learned drifting field operators.

LDF defines an antisymmetric field by construction:
V(x; P, Q) = S(x, P) - S(x, Q)
"""

from typing import List, Optional

import torch
import torch.nn as nn


class SetAttentionField(nn.Module):
    """
    Learned set operator:
    S(x, P) = sum_{y in P} w(x, y) * (g(y) - g(x))

    with attention logits:
    a(x, y) = MLP([g(x), g(y), g(y)-g(x), ||g(y)-g(x)||]).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        projection_dim: Optional[int] = 256,
        score_hidden_dim: int = 128,
        use_projection: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.use_projection = bool(use_projection)

        if self.use_projection:
            if projection_dim is None:
                projection_dim = hidden_dim
            self.rep_dim = int(projection_dim)
            self.g = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.rep_dim),
            )
            self.out_proj = nn.Linear(self.rep_dim, self.input_dim, bias=False)
        else:
            self.rep_dim = self.input_dim
            self.g = nn.Identity()
            self.out_proj = nn.Identity()

        self.score_net = nn.Sequential(
            nn.Linear(self.rep_dim * 3 + 1, score_hidden_dim),
            nn.GELU(),
            nn.Linear(score_hidden_dim, 1),
        )

    def _set_operator(
        self,
        feat_x: torch.Tensor,
        feat_set: torch.Tensor,
        mask_self: bool = False,
    ) -> torch.Tensor:
        if feat_set.shape[0] == 0:
            return torch.zeros_like(feat_x)

        x_rep = self.g(feat_x)  # (N, R)
        y_rep = self.g(feat_set)  # (M, R)

        n, r = x_rep.shape
        m = y_rep.shape[0]

        x_exp = x_rep.unsqueeze(1).expand(n, m, r)
        y_exp = y_rep.unsqueeze(0).expand(n, m, r)
        delta = y_exp - x_exp
        delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)

        score_in = torch.cat([x_exp, y_exp, delta, delta_norm], dim=-1)
        logits = self.score_net(score_in).squeeze(-1)  # (N, M)

        if mask_self and n == m:
            if n == 1:
                return torch.zeros_like(feat_x)
            diag = torch.eye(n, device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(diag, float("-inf"))

        weights = torch.softmax(logits, dim=1)  # Softmax over set members.
        set_update_rep = torch.sum(weights.unsqueeze(-1) * delta, dim=1)
        return self.out_proj(set_update_rep)

    def compute_V_learned(
        self,
        feat_x: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg: torch.Tensor,
        mask_self_neg: bool = False,
    ) -> torch.Tensor:
        s_pos = self._set_operator(feat_x, feat_pos, mask_self=False)
        s_neg = self._set_operator(feat_x, feat_neg, mask_self=mask_self_neg)
        return s_pos - s_neg

    def forward(
        self,
        feat_x: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg: torch.Tensor,
        mask_self_neg: bool = False,
    ) -> torch.Tensor:
        return self.compute_V_learned(
            feat_x,
            feat_pos,
            feat_neg,
            mask_self_neg=mask_self_neg,
        )


def compute_V_learned(
    feat_x: torch.Tensor,
    feat_pos: torch.Tensor,
    feat_neg: torch.Tensor,
    field: SetAttentionField,
    mask_self_neg: bool = False,
) -> torch.Tensor:
    """Functional wrapper for learned field computation."""
    return field.compute_V_learned(
        feat_x,
        feat_pos,
        feat_neg,
        mask_self_neg=mask_self_neg,
    )


class LearnedFieldBank(nn.Module):
    """One learned field per feature scale."""

    def __init__(
        self,
        feature_dims: List[int],
        hidden_dim: int = 256,
        projection_dim: Optional[int] = 256,
        score_hidden_dim: int = 128,
        use_projection: bool = True,
    ):
        super().__init__()
        self.feature_dims = [int(d) for d in feature_dims]
        self.fields = nn.ModuleList([
            SetAttentionField(
                input_dim=d,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                score_hidden_dim=score_hidden_dim,
                use_projection=use_projection,
            )
            for d in self.feature_dims
        ])

    def compute_V_learned(
        self,
        scale_idx: int,
        feat_x: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg: torch.Tensor,
        mask_self_neg: bool = False,
    ) -> torch.Tensor:
        return self.fields[scale_idx].compute_V_learned(
            feat_x,
            feat_pos,
            feat_neg,
            mask_self_neg=mask_self_neg,
        )

    def forward(
        self,
        scale_idx: int,
        feat_x: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg: torch.Tensor,
        mask_self_neg: bool = False,
    ) -> torch.Tensor:
        return self.compute_V_learned(
            scale_idx,
            feat_x,
            feat_pos,
            feat_neg,
            mask_self_neg=mask_self_neg,
        )
