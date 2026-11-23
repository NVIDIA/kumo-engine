# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name-to-`AttentionSpec` selection for serialized configurations.

Configs name a kernel; composition code takes the spec object. Keeping the
mapping here means `composition.attention` is a real switch rather than a label,
and an unknown name fails loudly instead of silently falling back to eager.
"""

from __future__ import annotations

from tfm.attention.eager import EagerAttention
from tfm.attention.sdpa import SdpaAttention
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.protocols import AttentionSpec

_SPECS: dict[str, type] = {
    "eager": EagerAttention,
    "sdpa": SdpaAttention,
}


def advertised_attention() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))


def build_attention(name: str) -> AttentionSpec:
    key = name.strip().lower()
    # Configs have historically spelled the kernel either way.
    key = key.removesuffix("_attention")
    if key not in _SPECS:
        raise UnsupportedCombinationError(
            f"Unknown attention kernel {name!r}; available: {advertised_attention()}."
        )
    return _SPECS[key]()
