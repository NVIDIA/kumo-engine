# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which part of a built sample an input reads.

A source selects; it does not represent. Every one of these names something the
`SampleSpec` already declared, which is why adding an input never changes what
was selected from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "CandidateSource",
    "EntitySource",
    "EventSource",
    "EventsOf",
    "OccurrencesOf",
    "InputSource",
    "NeighborhoodSource",
]


@dataclass(frozen=True)
class EntitySource:
    """Read values already declared by ``SampleSpec.entity_values``.

    Dated entity values are also available to entity token inputs.
    """


@dataclass(frozen=True)
class EventSource:
    """Read ordered event components already declared by ``SampleSpec``.

    :param events: component names, merged chronologically when several.
    :param collection: read ``SampleSpec.events`` or ``current_events``.
    :param max_events: limit applied after merging; each component's own
        ``EventSpec.limit`` applies first.
    :param keep: which end survives that limit — ``"last"`` keeps the most
        recent.
    :param order: deliver the selected events oldest-first (``"ascending"``)
        or newest-first (``"descending"``), independently of ``keep``.
    """

    events: tuple[str, ...]
    collection: str = "history"
    max_events: int | None = None
    keep: str = "last"
    order: str = "ascending"

    def __post_init__(self) -> None:
        if not self.events or len(self.events) != len(set(self.events)):
            raise ValueError("an event source requires unique component names")
        if self.collection not in {"history", "current"}:
            raise ValueError("event collection must be 'history' or 'current'")
        if self.max_events is not None and self.max_events <= 0:
            raise ValueError("maximum events must be positive")
        if self.keep not in {"first", "last"}:
            raise ValueError("event selection must keep 'first' or 'last'")
        if self.order not in {"ascending", "descending"}:
            raise ValueError("event order must be 'ascending' or 'descending'")
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True)
class NeighborhoodSource:
    """Read one neighborhood already declared by ``SampleSpec``.

    Use it with a ``Graph`` representation and ``layout=None``. A batch keeps
    each sample's graph disjoint, even when two samples reach the same stored
    node.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a neighborhood source requires a name")


@dataclass(frozen=True)
class CandidateSource:
    """Read one named candidate set supplied on each sample request.

    Candidates keep the caller's order. Use columns with a ``Padded`` or
    ``Packed`` candidate layout; padding is not an unclicked candidate.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a candidate source requires a name")


@dataclass(frozen=True)
class EventsOf:
    """Read original sampled fields from one grouped input's retained events.

    The parent must be a direct grouped ``EventSource`` input, and the follower
    uses grouped ``Columns``. It reads the parent's retained rows in the same
    order, before masking, following the parent's NULL and required-field skips
    as well as its event selection; it adds no filtering or limits of its own.
    Self, chained and cyclic references are refused before reading data.

    :param input: the parent input's name.
    """

    input: str

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("EventsOf requires an input name")


@dataclass(frozen=True)
class OccurrencesOf:
    """Project typed dated facts onto an entity token input's retained positions.

    A text value's pieces repeat its occurrence fields. Ordinary undated entity
    values have NULL occurrence fields. No timestamp or age is added implicitly.

    The parent must be a direct entity token input; the projection uses grouped
    ``Raw`` columns. Learned encoders are refused, because token lengths must
    not decide what their own encoders learn.

    :param input: the parent entity token input's name.
    """

    input: str

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("OccurrencesOf requires an input name")


InputSource: TypeAlias = (
    EntitySource | EventSource | EventsOf | OccurrencesOf | NeighborhoodSource | CandidateSource
)
