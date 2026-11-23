# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental profile + event + history granularity (PRAGMA-style)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tfm.contracts.kvt import CALENDAR, CyclicTime
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME


@dataclass(frozen=True)
class ProfileHistory:
    name: str = "profile_history"
    hidden_size: int = 32
    num_heads: int = 4
    profile_depth: int = 1
    event_depth: int = 2
    history_depth: int = 1
    vocab_size: int = 512
    dropout: float = 0.0
    rope_base: float = 10000.0
    max_position: int = 64
    varlen: bool = False
    """Skip all-padding event rows and trim token width before attention.

    Arithmetic bookkeeping only: the padded path is the correctness baseline and
    the two agree to floating-point tolerance. Worth enabling when histories are
    heavy-tailed, which is when most of the padded grid carries no token.
    """
    time_features: tuple[CyclicTime, ...] = CALENDAR
    """The per-event arrays the calendar embedding reads, each as a cycle."""
    support_tier: str = "experimental"
    accepted_layouts: frozenset[str] = field(
        default_factory=lambda: frozenset({LAYOUT_KEY_VALUE_TIME})
    )

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            from tfm.contracts.errors import IncompatibleCompositionError

            raise IncompatibleCompositionError("hidden_size must be divisible by num_heads.")
