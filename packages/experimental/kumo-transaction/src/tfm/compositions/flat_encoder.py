# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental bidirectional encoder over the flat field-token stream (ADR 0019)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.tokenizers import LAYOUT_FLAT_FIELDS, FieldVocabulary


@dataclass(frozen=True)
class FlatEncoder:
    """A BERT-style encoder on the same token rows `Flat(Decoder)` reads.

    Every token attends to every valid token in its row, so the encoder can only
    be pretrained by masking (`MaskedFields`). `vocabulary` locates each field's
    ids; the model adds one `<mask>` id after them, so the tokenizer is unchanged.
    """

    vocabulary: FieldVocabulary
    depth: int = 2
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.0
    max_seq_len: int = 256
    rope_base: float = 500000.0
    name: str = "flat_encoder"
    support_tier: str = "experimental"
    accepted_layouts: frozenset[str] = field(
        default_factory=lambda: frozenset({LAYOUT_FLAT_FIELDS})
    )

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise IncompatibleCompositionError("hidden_size must be divisible by num_heads.")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise IncompatibleCompositionError("Rotary positions need an even head dimension.")

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.vocab_size

    @property
    def mask_id(self) -> int:
        """The id the encoder appends after the tokenizer's vocabulary."""
        return self.vocabulary.vocab_size
