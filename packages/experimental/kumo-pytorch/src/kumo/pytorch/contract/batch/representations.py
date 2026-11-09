# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered field encodings and their logical row/history arrangements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TypeAlias

from kumo.pytorch.contract.batch._common import _mapping
from kumo.pytorch.contract.batch.encodings import (
    Raw,
    ScalarEncoding,
    Text,
    UseEncoding,
    ValueEncoding,
    ZScore,
)
from kumo.pytorch.contract.batch.tokens import Columns, Composite, Flatten, Grouped, Tokens


@dataclass(frozen=True)
class Encode:
    """Encode one sampled field, optionally translating IDs into a final space.

    :param field: sampled field name, short or qualified.
    :param encoding: an explicit encoder or a ``UseEncoding`` reference; there
        is no automatic choice.
    :param id_space: a ``BatchSpec.id_spaces`` name. Without it the use keeps
        local IDs, even when another use of the same encoder joins a space.
    """

    field: str
    encoding: ValueEncoding
    id_space: str | None = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("an encoded field requires a name")
        if not isinstance(self.encoding, (ScalarEncoding, UseEncoding)):
            raise ValueError("an encoded field requires an explicit value encoding")
        if self.id_space is not None and not self.id_space:
            raise ValueError("an ID-space reference requires a name")
        if self.id_space is not None and isinstance(self.encoding, (Raw, ZScore)):
            raise ValueError("Raw and ZScore values cannot join an ID space")


@dataclass(frozen=True)
class Fields:
    """Select ordered fields with explicit encoders, grouping, and field IDs.

    ``fields`` is an ordered, nonempty sequence; its order is part of the
    output. Each selected sampled field has exactly one encoding per input:
    selecting both a short and a qualified name is a duplicate. An unqualified
    event field selects that field in every participating component, whose
    dtypes must agree; ``events/<component>/<field>`` selects one.

    :param row: ``Columns()``, ``Tokens()`` or ``Composite()``.
    :param history: ``Grouped()`` or ``Flatten()`` for event inputs.
    :param field_names: a ``BatchSpec.field_names`` dictionary to emit parallel
        field IDs from.
    :param field_space: final space the field IDs translate into.
    :param forbidden_fields: fields this input must not select, even through an
        alias or a named encoding.
    """

    fields: tuple[Encode, ...]
    row: Columns | Tokens | Composite = dataclass_field(default_factory=Columns)
    history: Grouped | Flatten = dataclass_field(default_factory=Grouped)
    field_names: str | None = None
    field_space: str | None = None
    forbidden_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.row, (Columns, Tokens, Composite)):
            raise ValueError("Fields row requires Columns, Tokens, or Composite")
        if not isinstance(self.history, (Grouped, Flatten)):
            raise ValueError("Fields history requires Grouped or Flatten")
        if not isinstance(self.fields, (tuple, list)) or not self.fields:
            raise ValueError("Fields requires an ordered non-empty sequence of Encode declarations")
        if any(not isinstance(value, Encode) for value in self.fields):
            raise ValueError("Fields entries must be Encode declarations")
        names = tuple(value.field for value in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("an input cannot encode the same field more than once")
        if set(names) & set(self.forbidden_fields):
            raise ValueError("an input selects a forbidden field")
        if self.field_space is not None and self.field_names is None:
            raise ValueError("field_space requires a field_names dictionary")
        if isinstance(self.row, Tokens):
            if any(isinstance(value.encoding, (Raw, ZScore)) for value in self.fields):
                raise ValueError("token rows require ID-producing encoders")
            spaces = {value.id_space for value in self.fields}
            if None in spaces or len(spaces) != 1:
                raise ValueError("token rows require one explicit final ID space")
        if isinstance(self.row, Composite):
            if self.field_names is not None:
                raise ValueError("composite rows cannot emit field-name IDs")
            if any(isinstance(value.encoding, Text) for value in self.fields):
                raise ValueError("composite components must be scalar, not Text")
            if any(value.id_space is not None for value in self.fields):
                raise ValueError("composite component IDs are local; set id_space on Composite")
            if not set(self.row.required_fields) <= set(names):
                raise ValueError("composite required fields must be selected")
            if self.row.vocabulary is not None and any(
                len(key) != len(self.fields) for key, _ in self.row.vocabulary
            ):
                raise ValueError("composite vocabulary tuples must match the selected field count")
        if isinstance(self.history, Flatten) and isinstance(self.row, Columns):
            raise ValueError("flattened histories require Tokens or Composite rows")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "forbidden_fields", tuple(self.forbidden_fields))


@dataclass(frozen=True)
class Graph:
    """Node attributes use the same explicit field encodings as ordinary columns.

    :param node_fields: table name to the ``Fields`` (scalar column encodings
        of physical columns) emitted for its nodes. Field-name IDs are not
        supported on graph nodes.
    """

    node_fields: Mapping[str, Fields] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Fields) or not isinstance(value.row, Columns)
            for value in self.node_fields.values()
        ):
            raise ValueError("graph attributes require Columns rows")
        object.__setattr__(self, "node_fields", _mapping(self.node_fields))


Representation: TypeAlias = Fields | Graph
