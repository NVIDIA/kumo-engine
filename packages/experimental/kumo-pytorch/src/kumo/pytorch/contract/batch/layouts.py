# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a variable-length axis is stored.

A layout changes the *shape* of what a model receives, never its meaning: the
same values padded and packed carry the same information, at different cost.
That is the line between this module and an encoding — an encoding decides what
a value becomes, a layout decides how the result is arranged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "Nested",
    "Packed",
    "Padded",
    "Separate",
    "SequenceLayout",
]


@dataclass(frozen=True)
class Padded:
    """Rectangular sequences accompanied by a boolean validity mask.

    Padding is storage: a padded position is never a real value or event.
    """


@dataclass(frozen=True)
class Packed:
    """Flat values accompanied by offsets delimiting each sequence.

    Every packed boundary is emitted three ways: ``offsets`` (int64), and for varlen
    attention kernels ``cu_seqlens`` (the same boundaries as int32) and ``max_seqlen``
    (a host int). Under `Nested` the token level is prefixed ``field_`` and the event
    level sits under ``sequence``.
    """


SequenceLayout: TypeAlias = Padded | Packed


@dataclass(frozen=True)
class Separate:
    """Keep fixed entity fields as independently named arrays."""


@dataclass(frozen=True)
class Nested:
    """Storage choices for fields within events and events within samples.

    :param fields: layout of the tokens inside one event.
    :param events: layout of the events inside one sample.
    """

    fields: SequenceLayout
    events: SequenceLayout
