# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Blueprint v2 granularity: profile + event encoders, causal decoder (ADR 0012)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME


@dataclass(frozen=True)
class ProfileEventDecoder:
    """PRAGMA-style profile and event encoders feeding a causal history decoder.

    Each event becomes one summary token; the decoder reads `[profile, event_1,
    ..., event_N]` causally with TimeRoPE over time since the first event.
    """

    name: str = "profile_event_decoder"
    hidden_size: int = 32
    num_heads: int = 4
    profile_depth: int = 1
    event_depth: int = 2
    decoder_depth: int = 2
    vocab_size: int = 512
    dropout: float = 0.0
    rope_base: float = 10000.0
    max_position: int = 64
    varlen: bool = False
    """Skip all-padding event rows in the event encoder; the padded path is the baseline."""
    support_tier: str = "experimental"
    accepted_layouts: frozenset[str] = field(
        default_factory=lambda: frozenset({LAYOUT_KEY_VALUE_TIME})
    )

    def __post_init__(self) -> None:
        from tfm.contracts.errors import IncompatibleCompositionError

        if self.hidden_size % self.num_heads != 0:
            raise IncompatibleCompositionError("hidden_size must be divisible by num_heads.")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise IncompatibleCompositionError("TimeRoPE needs an even head dimension.")
        if min(self.profile_depth, self.event_depth, self.decoder_depth) < 1:
            raise IncompatibleCompositionError("Every component needs at least one layer.")
