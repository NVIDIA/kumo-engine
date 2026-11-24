# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRU history backbone for the flat stream (experimental baseline).

Recurrent models are the standard baseline in event-sequence benchmarks
(CoLES, EBES) and the backbone of Featurespace's NPPR. A GRU is causal by
construction: the state at position `t` depends only on tokens `0..t`, so it
satisfies the next-event objective's contract without an attention mask.
Select it with `model.backbone: gru`; it reuses the decoder's width and depth.
"""

from __future__ import annotations

from torch import Tensor, nn

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.layers.decoder import Decoder


class GRUBackbone(nn.Module):
    """Token embedding, stacked GRU, norm, and a language-model head.

    It takes no `segment_ids`, so the runtime refuses packed batches rather
    than let the recurrent state run across entities.
    """

    def __init__(self, spec: Decoder) -> None:
        super().__init__()
        if spec.depth < 1:
            raise IncompatibleCompositionError("GRUBackbone needs depth >= 1.")
        self.spec = spec
        self.hidden_size = spec.hidden_size
        self.vocab_size = spec.vocab_size
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.gru = nn.GRU(
            spec.hidden_size,
            spec.hidden_size,
            num_layers=spec.depth,
            batch_first=True,
            dropout=spec.dropout if spec.depth > 1 else 0.0,
        )
        self.final_norm = nn.LayerNorm(spec.hidden_size)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, causal: bool = True) -> Tensor:
        if not causal:
            raise IncompatibleCompositionError("GRUBackbone is causal only.")
        hidden = self.embed(input_ids) * attention_mask.unsqueeze(-1).to(self.embed.weight.dtype)
        output, _state = self.gru(hidden)
        return self.final_norm(output)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)
