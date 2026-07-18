from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .._potential_ops import actualize_h, compute_h_scores


class PotentialMultiheadAttention(nn.Module):
    """Multi-head attention with parallel hypothesis-specific attention modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_potentials: int = 8,
        dropout: float = 0.0,
        batch_first: bool = True,
        bias: bool = True,
        use_prev_h: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_potentials = num_potentials
        self.batch_first = batch_first
        self.use_prev_h = use_prev_h
        self.attention_potentials = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=batch_first,
                    bias=bias,
                )
                for _ in range(num_potentials)
            ]
        )
        self.prev_h_projections = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_potentials)]
        )

    def forward(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for i, attention in enumerate(self.attention_potentials):
            out_i, _ = attention(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )
            if H is not None and self.use_prev_h:
                if H.shape[0] != self.num_potentials or H.shape[-1] != self.embed_dim:
                    raise ValueError(
                        "prev H shape is incompatible with attention layer"
                    )
                out_i = out_i + self.prev_h_projections[i](H[i])
            outputs.append(out_i)

        H_out = torch.stack(outputs, dim=0)
        scores = compute_h_scores(H_out)
        output = actualize_h(H_out, scores)
        H_out._is_potential = True
        H_out._layer = "MultiheadAttention"
        H_out._meaning = "Possible attention outputs"
        H_out._scores = scores
        return output, H_out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_potentials={self.num_potentials}, batch_first={self.batch_first}, "
            f"use_prev_h={self.use_prev_h}"
        )
