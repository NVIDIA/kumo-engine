# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-index rotary positions, applied inside the backbone's attention."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RoPE:
    """Rotary positions over token indices, as a Llama backbone applies them.

    The spec declares what the backbone does; it carries no module of its own.
    The eager `DecoderModule` does not implement it, so only a backbone whose
    `position` attribute is `"rope"` (the HuggingFace Llama backbone) accepts it.
    """

    name: str = "rope"

    def apply(self, hidden: Tensor) -> Tensor:
        """Identity on the backbone's output: the rotation happened inside attention."""
        return hidden
