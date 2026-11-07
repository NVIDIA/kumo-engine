# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row arrangements, history boundaries, and explicit ID dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from kumo.pytorch.contract.batch._common import _mapping

ROLE_ORDER = ("pad", "unknown", "missing", "mask", "entity", "event", "separator", "bos", "eos")


@dataclass(frozen=True)
class IdSpace:
    """Final numbering shared by explicitly participating encoder uses.

    Different encoders get separate allocations even when their local IDs or
    values are equal; uses of one shared encoder get one. Declared roles have
    stable IDs, allocated in ``pad``, ``unknown``, ``missing``, ``mask``,
    ``entity``, ``event``, ``separator``, ``bos``, ``eos`` order, then custom
    names lexically. Members' local missing and unknown IDs map to the matching
    final roles; ordinary entries never collide with roles. Only discrete
    encoders can join.

    :param specials: role name to its display label.
    """

    specials: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str) or not name or not isinstance(label, str) or not label
            for name, label in self.specials.items()
        ):
            raise ValueError("ID-space role names and labels must be non-empty strings")
        object.__setattr__(self, "specials", _mapping(self.specials))


@dataclass(frozen=True)
class FieldNames:
    """IDs for declared field identities, independent from value encoders.

    Identifies declared sampled fields, not the letters in their names. A field
    keeps its ID across inputs using the same dictionary, even with different
    value encoders; equal short names in different event components are
    distinct. Markers have no field identity.
    """


@dataclass(frozen=True)
class Columns:
    """Keep selected encoded fields as separate typed columns.

    Each field keeps its own array and dtype; NULL has a validity mask distinct
    from a valid zero or an unknown category. Repeated values carry occurrence
    offsets and text pieces carry piece offsets.
    """


@dataclass(frozen=True)
class Tokens:
    """Concatenate one row's encoded IDs in the declared field order.

    Every value must come from an ID-producing encoder joining the same final
    ID space; ``Raw`` and ``ZScore`` values cannot be tokens. Field IDs are
    parallel, never inserted tokens; piece positions and source-event indices
    stay explicit.

    :param max_tokens: per-row token limit.
    :param keep: which end survives the limit.
    :param missing: for a NULL declared value: ``"omit"``, ``"emit"``,
        ``"skip_event"`` or ``"error"``. A present empty text
        contributes zero pieces without removing the event.
    :param occurrence_order: entity tokens only. ``"fields"`` groups
        occurrences under each field; ``"chronological"`` emits undated values
        first in declaration order, then dated occurrences by time.
    """

    max_tokens: int | None = None
    keep: Literal["first", "last"] = "first"
    missing: Literal["omit", "emit", "skip_event", "error"] = "omit"
    occurrence_order: Literal["fields", "chronological"] = "fields"

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("row maximum tokens must be positive")
        if self.keep not in {"first", "last"}:
            raise ValueError("row truncation must keep first or last")
        if self.missing not in {"omit", "emit", "skip_event", "error"}:
            raise ValueError("token missing policy must be omit, emit, skip_event, or error")
        if self.occurrence_order not in {"fields", "chronological"}:
            raise ValueError("token occurrence_order must be fields or chronological")


@dataclass(frozen=True)
class Composite:
    """One category ID for the ordered tuple of scalar component results.

    Components keep their local encoder results; only the outer ID can join a
    final space. Fitting keeps the ``top_k`` most frequent tuples, ties broken
    by canonical typed tuple order; other tuples use the unknown ID. Text and
    repeated components are refused before reading data.

    :param required_fields: a NULL in one of these drops the event.
    :param missing: an optional NULL component stays a distinct typed missing
        value.
    :param id_space: final space for the outer ID.
    :param vocabulary: supplied tuple/ID pairs instead of fitting.
    """

    top_k: int = 100_000
    required_fields: tuple[str, ...] = ()
    missing: Literal["typed"] = "typed"
    id_space: str | None = None
    vocabulary: tuple[tuple[tuple[Any, ...], int], ...] | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("composite top_k must be positive")
        if self.missing != "typed":
            raise ValueError("composite missing policy must be typed")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("composite required fields must be unique")
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        if self.vocabulary is not None:
            entries = tuple(
                (tuple(tuple(component) for component in key), index)
                for key, index in self.vocabulary
            )
            ids = [index for _, index in entries]
            if (
                any(type(index) is not int or not 2 <= index < 2**63 for index in ids)
                or len(set(ids)) != len(ids)
                or len({key for key, _ in entries}) != len(entries)
            ):
                raise ValueError(
                    "composite keys must be unique and IDs distinct int64 values above 1"
                )
            object.__setattr__(self, "vocabulary", tuple(sorted(entries, key=lambda item: item[1])))


@dataclass(frozen=True)
class Grouped:
    """Preserve each event as its own logical sequence."""


@dataclass(frozen=True)
class Flatten:
    """One sequence per sample, with explicit markers and a total token limit.

    Markers name reserved roles of the final ID space, count toward the limit
    and have no source field or event. Sample boundaries are never removed.

    :param max_tokens: total budget per sample, markers included.
    :param keep: which end survives clipping.
    :param prefer_event_boundary: clip at complete events where possible; a
        single oversized event is clipped rather than dropping all content.
    :param before: role placed first.
    :param separator: role placed between (or after) events.
    :param separator_placement: ``"between"`` or ``"after_each"``.
    :param after: role placed last.
    """

    max_tokens: int
    keep: Literal["first", "last"] = "last"
    prefer_event_boundary: bool = True
    before: str | None = None
    separator: str | None = None
    separator_placement: Literal["between", "after_each"] = "between"
    after: str | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= int(self.before is not None) + int(self.after is not None):
            raise ValueError("history token budget must leave room beyond before/after markers")
        if self.keep not in {"first", "last"}:
            raise ValueError("history truncation must keep first or last")
        if self.separator_placement not in {"between", "after_each"}:
            raise ValueError("separator placement must be between or after_each")
