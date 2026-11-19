# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task heads.

- :class:`MLMHead` — the pretraining head: for each (masked) token it concatenates
  ``[ẑ_e(token), z_h(its event), z_h(USR)] ∈ R^{3d}``, projects ``3d → d`` and
  produces logits via the **tied** embedding weights; trained with cross-entropy
  and label smoothing 0.1.
- :class:`ClassificationHead` — a fine-tuning head on the user embedding
  ``z_h[USR]``.

Heads are instantiated directly by the benchmark runner.
"""

from __future__ import annotations

import torch
from torch import nn

from .pragmatiq import PragmaOutput


class MLMHead(nn.Module):
    """Masked-language-model head: 3d concat → ``Linear(3d→d)`` → tied logits.

    For each (masked) token it concatenates ``[ẑ_e(token), z_h(its event),
    z_h(USR)] ∈ R^{3d}``, projects ``3d → d``, and reads out logits through the
    **tied** token-embedding weights. Trained with cross-entropy and label
    smoothing 0.1.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * dim, dim)

    def _context(self, out: PragmaOutput, token_idx: torch.Tensor | None) -> torch.Tensor:
        """The ``[ẑ_e, z_h(event), z_h(USR)] ∈ R^{3d}`` context per selected token.

        Ordinary tensor indexing preserves the requested order and supports
        repeated indices. PyTorch autograd accumulates their gradients.
        """
        tok = out.token_repr if token_idx is None else out.token_repr[token_idx]
        ev = out.event_of_token if token_idx is None else out.event_of_token[token_idx]
        z_he = out.history_event_repr[ev]
        z_hu = out.user_repr[out.user_of_event[ev]]
        return torch.cat([tok, z_he, z_hu], dim=-1)

    def forward(
        self,
        out: PragmaOutput,
        embedding_weight: torch.Tensor,
        token_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits ``[M, vocab]`` for the selected (default: all) tokens."""
        h = self.proj(self._context(out, token_idx))
        return h @ embedding_weight.t()  # tied logits


class ClassificationHead(nn.Module):
    """Linear (optionally MLP) head on the user embedding ``z_h[USR]``."""

    def __init__(self, dim: int, n_classes: int = 2, hidden: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_classes = n_classes
        if hidden:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
            )
        else:
            self.net = nn.Linear(dim, n_classes)

    def forward(self, user_repr: torch.Tensor) -> torch.Tensor:
        """Logits ``[n_users, n_classes]`` from user embeddings ``[n_users, d]``."""
        return self.net(user_repr)
