# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named outputs, shared encoding resources, and batch delivery settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, TypeAlias

import torch

from kumo.pytorch.contract.batch._common import _mapping, _move
from kumo.pytorch.contract.batch.encodings import Raw, ScalarEncoding, Text, UseEncoding, ZScore
from kumo.pytorch.contract.batch.layouts import Nested, Packed, Padded, Separate, SequenceLayout
from kumo.pytorch.contract.batch.representations import Fields, Graph, Representation
from kumo.pytorch.contract.batch.sources import (
    CandidateSource,
    EntitySource,
    EventsOf,
    EventSource,
    InputSource,
    NeighborhoodSource,
    OccurrencesOf,
)
from kumo.pytorch.contract.batch.targets import (
    AlignedEventTarget,
    CandidateTarget,
    LabelTarget,
    MaskedValuesTarget,
    MatchingGroupTarget,
    NextTokenTarget,
    StructuredTarget,
    TargetSpec,
)
from kumo.pytorch.contract.batch.tokens import (
    Columns,
    Composite,
    FieldNames,
    Flatten,
    Grouped,
    IdSpace,
    Tokens,
)


@dataclass(frozen=True)
class Input:
    """Select sampled rows, encode their fields, and choose their storage layout.

    Valid layouts by data: entity columns use ``Separate()``; entity tokens,
    grouped event columns or composites, flattened histories and candidate
    columns use ``Padded()`` or ``Packed()``; grouped event tokens use
    ``Nested(fields=..., events=...)``; a graph uses ``None``.

    :param source: which sampled data the input reads.
    :param representation: ``Fields`` or ``Graph``.
    :param layout: storage of variable-length axes.
    """

    source: InputSource
    representation: Representation
    layout: Separate | SequenceLayout | Nested | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.source,
            (
                EntitySource,
                EventSource,
                EventsOf,
                OccurrencesOf,
                CandidateSource,
                NeighborhoodSource,
            ),
        ):
            raise ValueError("input requires a declared sampled source")
        if isinstance(self.source, NeighborhoodSource):
            if not isinstance(self.representation, Graph) or self.layout is not None:
                raise ValueError("neighborhood inputs require Graph and no layout")
            return
        if not isinstance(self.representation, Fields):
            raise ValueError("non-graph inputs require Fields")
        value = self.representation
        if isinstance(value.row, Columns) and not isinstance(self.source, EntitySource):
            reserved = {"valid", "offsets", "cu_seqlens", "max_seqlen"}
            if any(use.field in reserved for use in value.fields):
                raise ValueError(
                    "column selectors conflict with layout metadata; qualify the field"
                )
        if (
            isinstance(value.row, Tokens)
            and value.row.occurrence_order == "chronological"
            and not isinstance(self.source, EntitySource)
        ):
            raise ValueError("chronological field occurrences require an entity source")
        if isinstance(self.source, EntitySource):
            expected = (Separate,) if isinstance(value.row, Columns) else (Padded, Packed)
            if not isinstance(self.layout, expected) or not isinstance(value.history, Grouped):
                raise ValueError(
                    "entity columns require Separate; entity tokens require Padded or Packed"
                )
        elif isinstance(self.source, (EventSource, EventsOf)):
            expected = (
                (Nested,)
                if isinstance(value.row, Tokens) and isinstance(value.history, Grouped)
                else (Padded, Packed)
            )
            if not isinstance(self.layout, expected):
                raise ValueError(
                    "grouped token events require Nested; "
                    "other event outputs require Padded or Packed"
                )
            if isinstance(self.source, EventsOf) and (
                not isinstance(value.row, Columns) or not isinstance(value.history, Grouped)
            ):
                raise ValueError("EventsOf followers require grouped Columns")
        elif isinstance(self.source, OccurrencesOf):
            if (
                not isinstance(value.row, Columns)
                or not isinstance(value.history, Grouped)
                or not isinstance(self.layout, (Padded, Packed))
            ):
                raise ValueError("OccurrencesOf requires grouped Columns and Padded or Packed")
            if any(not isinstance(use.encoding, Raw) for use in value.fields):
                raise ValueError("occurrence projections require Raw encodings")
        elif isinstance(self.source, CandidateSource):
            if not isinstance(value.row, Columns) or not isinstance(self.layout, (Padded, Packed)):
                raise ValueError("candidate inputs require Columns and Padded or Packed")


InputSpec: TypeAlias = Input


@dataclass(frozen=True)
class Batching:
    """Limits on samples and materialized token slots, independent of layout.

    :param max_samples: samples per batch.
    :param max_tokens: budget of materialized slots of ``length_input``,
        padding included. An example larger than the budget is refused rather
        than exceeding it. Refused with ``world_size > 1``, where ranks could
        emit different numbers of steps.
    :param length_input: the token input whose length the budget and bucketing
        use; pick the token input itself, not an occurrence projection.
    :param bucket_by_length: group samples of similar length.
    :param drop_last: drop the final batch of a pass when it holds fewer than
        ``max_samples`` (or ``batch_size``) samples; a batch the token budget
        closed early is kept. Without a sample limit nothing is dropped.
        Inference keeps the last batch unless ``task.batches`` is given
        ``drop_last``.
    """

    max_samples: int | None = None
    max_tokens: int | None = None
    length_input: str | None = None
    bucket_by_length: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("maximum samples per batch must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("maximum tokens per batch must be positive")
        if (self.max_tokens is not None or self.bucket_by_length) and not self.length_input:
            raise ValueError("token budgets and length bucketing require length_input")


def _field_groups(value: Input):
    if isinstance(value.representation, Fields):
        yield value.representation
    else:
        yield from value.representation.node_fields.values()


@dataclass(frozen=True)
class BatchSpec:
    """Named model inputs and targets, with explicit shared mappings.

    :param inputs: named inputs, each selecting sampled data, encoding ordered
        fields and choosing a layout.
    :param targets: named targets. Encoding a target never inserts it into an
        input.
    :param encodings: named encoders shared by ``UseEncoding`` references,
        each fitted once across its uses.
    :param id_spaces: named final ID spaces that participating uses translate
        into.
    :param field_names: named field-identity dictionaries for ``Fields``.
    :param batching: sample and token limits.
    """

    inputs: Mapping[str, Input]
    targets: Mapping[str, TargetSpec] = dataclass_field(default_factory=dict)
    encodings: Mapping[str, ScalarEncoding] = dataclass_field(default_factory=dict)
    id_spaces: Mapping[str, IdSpace] = dataclass_field(default_factory=dict)
    field_names: Mapping[str, FieldNames] = dataclass_field(default_factory=dict)
    batching: Batching = dataclass_field(default_factory=Batching)

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("batch specification requires at least one input")
        for name in ("inputs", "targets", "encodings", "id_spaces", "field_names"):
            object.__setattr__(self, name, _mapping(getattr(self, name)))
        for input_name, value in self.inputs.items():
            if not isinstance(value, Input):
                raise ValueError(f"input {input_name!r} requires Input")
            if isinstance(value.source, EventsOf):
                parent = self.inputs.get(value.source.input)
                if parent is None or value.source.input == input_name:
                    raise ValueError(
                        f"input {input_name!r} has an unknown or self EventsOf reference"
                    )
                if not isinstance(parent.source, EventSource) or not isinstance(
                    parent.representation.history, Grouped
                ):
                    raise ValueError("EventsOf requires a direct grouped EventSource parent")
            if isinstance(value.source, OccurrencesOf):
                parent = self.inputs.get(value.source.input)
                if parent is None or value.source.input == input_name:
                    raise ValueError("OccurrencesOf requires another declared input")
                if not isinstance(parent.source, EntitySource) or not isinstance(
                    parent.representation.row, Tokens
                ):
                    raise ValueError("OccurrencesOf requires a direct entity token parent")
            for group in _field_groups(value):
                if group.field_names is not None and group.field_names not in self.field_names:
                    raise ValueError(
                        f"input {input_name!r} refers to unknown field-name dictionary"
                    )
                destinations = {group.field_space, *(item.id_space for item in group.fields)}
                if isinstance(group.row, Composite):
                    destinations.add(group.row.id_space)
                if destinations - {None} - set(self.id_spaces):
                    raise ValueError(f"input {input_name!r} refers to unknown ID space")
                for item in group.fields:
                    encoding = (
                        self.encodings.get(item.encoding.name)
                        if isinstance(item.encoding, UseEncoding)
                        else item.encoding
                    )
                    if encoding is None:
                        raise ValueError(f"input {input_name!r} refers to unknown encoding")
                    if isinstance(group.row, Tokens) and isinstance(encoding, (Raw, ZScore)):
                        raise ValueError("token rows require ID-producing encoders")
                    if isinstance(group.row, Composite) and isinstance(encoding, Text):
                        raise ValueError("composite components must be scalar, not Text")
                space_name = (
                    group.row.id_space
                    if isinstance(group.row, Composite)
                    else group.fields[0].id_space
                )
                roles = self.id_spaces[space_name].specials if space_name is not None else {}
                if (
                    group.field_space is not None
                    and "pad" not in self.id_spaces[group.field_space].specials
                ):
                    raise ValueError("field-ID output space requires the pad role")
                if isinstance(group.row, Tokens) and "pad" not in roles:
                    raise ValueError("token output space requires the pad role")
                if (
                    isinstance(group.row, Composite)
                    and space_name is not None
                    and isinstance(value.layout, Padded)
                    and "pad" not in roles
                ):
                    raise ValueError("padded composite output space requires the pad role")
                if isinstance(group.history, Flatten):
                    if space_name is None:
                        raise ValueError("flattened output requires an explicit final ID space")
                    for role in (
                        group.history.before,
                        group.history.separator,
                        group.history.after,
                    ):
                        if role is not None and role not in roles:
                            raise ValueError(f"flatten marker role {role!r} is not declared")
        if self.batching.length_input is not None and self.batching.length_input not in self.inputs:
            raise ValueError("batching refers to unknown length input")
        if self.batching.length_input is not None and isinstance(
            self.inputs[self.batching.length_input].source, OccurrencesOf
        ):
            raise ValueError(
                "use the parent token input as length_input, not an occurrence projection"
            )
        for name, target in self.targets.items():
            if isinstance(target, (LabelTarget, StructuredTarget)):
                continue
            if target.input not in self.inputs:
                raise ValueError(f"target {name!r} refers to unknown input")
            value = self.inputs[target.input]
            group = value.representation
            if not isinstance(group, Fields):
                raise ValueError(f"target {name!r} requires a field input")
            compatible = (
                isinstance(target, CandidateTarget)
                and isinstance(value.source, CandidateSource)
                or isinstance(target, AlignedEventTarget)
                and isinstance(value.source, (EventSource, EventsOf))
                and isinstance(group.row, Columns)
                or isinstance(target, MaskedValuesTarget)
                and isinstance(group.row, Tokens)
                or isinstance(target, NextTokenTarget)
                and isinstance(group.history, Flatten)
                or isinstance(target, MatchingGroupTarget)
                and isinstance(value.source, EntitySource)
                and isinstance(group.row, Columns)
            )
            if not compatible:
                raise ValueError(f"target {name!r} is incompatible with input {target.input!r}")
            selected = {use.field for use in group.fields}
            if isinstance(target, AlignedEventTarget) and selected & {
                use.field for use in target.fields
            }:
                raise ValueError("aligned target fields must be withheld from their event input")
            if isinstance(target, MatchingGroupTarget) and not set(target.fields) <= selected:
                raise ValueError("matching target fields must be selected by their input")
            if isinstance(target, NextTokenTarget) and target.prefix_input is not None:
                prefix = self.inputs.get(target.prefix_input)
                if prefix is None or not isinstance(prefix.source, EntitySource):
                    raise ValueError("next-token prefix must name an entity input")
            if isinstance(target, MaskedValuesTarget):
                destination = group.fields[0].id_space
                roles = self.id_spaces[destination].specials
                if "mask" not in roles or (
                    target.unknown_probability > 0 and "unknown" not in roles
                ):
                    raise ValueError("masking requires declared mask and applicable unknown roles")
        token_targets = [
            target.input
            for target in self.targets.values()
            if isinstance(target, (MaskedValuesTarget, NextTokenTarget))
        ]
        if len(token_targets) != len(set(token_targets)):
            raise ValueError("an input supports only one masked or next-token target")


@dataclass(frozen=True)
class Batch:
    """Named arrays and per-sample metadata supplied to model and loss code.

    Inputs hold what a model may observe and targets what it predicts; they are
    never mixed. Separate columns sit under their field names with ``values``
    and ``valid``; token outputs use ``value_ids``, ``position_ids``, validity
    and source-event indices, with optional ``field_ids`` and ``field_valid``.
    Raw types without a tensor representation stay Arrow arrays in inspectable
    output. Metadata carries ``sample_ids``, ``entity_ids``, ``as_of_time``,
    ``splits``, ``is_padding``, request metadata, fingerprints and candidate
    identities; it is not a route for undeclared features.
    """

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Batch:
        """Move every nested tensor to ``device``; masks, offsets and Arrow values are kept."""
        return Batch(
            inputs=_move(self.inputs, device, non_blocking),
            targets=_move(self.targets, device, non_blocking),
            metadata=_move(self.metadata, device, non_blocking),
        )

    @property
    def batch_size(self) -> int:
        """Number of samples, padding samples included."""
        return len(self.metadata.get("sample_ids", ()))
