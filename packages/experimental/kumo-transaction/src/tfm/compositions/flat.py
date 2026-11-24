# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat (single-stream) composition wrapping a causal decoder."""

from __future__ import annotations

from dataclasses import dataclass, field

from tfm.contracts.tokenizers import LAYOUT_FLAT_FIELDS, LAYOUT_FLAT_KEY_VALUE
from tfm.layers.decoder import Decoder


@dataclass(frozen=True)
class Flat:
    """One token stream per entity. Accepts any layout with separators between events."""

    model: Decoder
    name: str = "flat"
    accepted_layouts: frozenset[str] = field(
        default_factory=lambda: frozenset({LAYOUT_FLAT_FIELDS, LAYOUT_FLAT_KEY_VALUE})
    )
