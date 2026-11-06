# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervision, kept separate from the inputs it is aligned with.

A target never enters a model input. That separation is what makes a label
withheld from the input provably withheld, rather than withheld by convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from kumo.pytorch.contract.batch._common import _probability
from kumo.pytorch.contract.batch.encodings import ValueEncoding
from kumo.pytorch.contract.batch.representations import Encode
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import ComparisonOperation

__all__ = [
    "AlignedEventTarget",
    "CandidateTarget",
    "LabelTarget",
    "MaskedValuesTarget",
    "MatchingGroupTarget",
    "NextTokenTarget",
    "StructuredTarget",
    "TargetSpec",
    "ThresholdEventLabel",
]


@dataclass(frozen=True)
class ThresholdEventLabel:
    """A categorical event label derived from a supplied or fitted threshold.

    The fitted threshold is target state, fitted on the fit split: the median,
    the midpoint for an even-sized population, and zero for an empty one.

    :param threshold: fixed cut point; omit to fit one with ``fit_statistic``.
    :param below: label for values on the non-matching side.
    :param above: label for values satisfying ``comparison``.
    :param fit_statistic: statistic used when ``threshold`` is omitted.
    :param comparison: which side of the threshold counts as ``above``.
    """

    name: str
    field: str
    dtype: DType
    encoding: ValueEncoding
    threshold: float | None = None
    below: str = "normal"
    above: str = "abnormal"
    fit_statistic: str = "median"
    comparison: ComparisonOperation = "greater_than"

    def __post_init__(self) -> None:
        if not self.name or not self.field:
            raise ValueError("threshold label name and source field must be non-empty")
        if self.below == self.above:
            raise ValueError("threshold label categories must be different")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("threshold label cut point must be finite")
        if self.fit_statistic != "median":
            raise ValueError("only median threshold fitting is supported")
        if self.comparison not in {"greater_than", "greater_or_equal"}:
            raise ValueError("threshold comparison must be greater_than or greater_or_equal")


@dataclass(frozen=True)
class LabelTarget:
    """A label supplied beside each sample rather than inserted into its inputs.

    :param source: name of the `SampleSpec` target or request-supplied value.
    :param encoding: an explicit encoder, or a ``UseEncoding`` reference. A
        target sharing an input encoder reuses its mapping without adding
        observations; a reused final space cannot gain target members.
    :param id_space: optional final ID space.
    """

    source: str
    encoding: ValueEncoding
    id_space: str | None = None


@dataclass(frozen=True)
class StructuredTarget:
    """Independently encode named fields from a structured sample target.

    :param source: a ``RowTarget`` in ``SampleSpec.targets``.
    :param fields: the row's fields to encode; scalar values only, a repeated
        field is refused.
    """

    source: str
    fields: tuple[Encode, ...]

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("structured target source must be non-empty")
        if not self.fields:
            raise ValueError("structured targets require at least one field")
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class CandidateTarget:
    """Encode one label for every candidate emitted by a candidate input.

    Labels follow the caller's candidate order. Candidate identity, row
    validity and label validity are separate; padding is not a negative.

    :param input: the candidate input these labels stay aligned with.
    :param source: name of the per-candidate target value.
    """

    input: str
    source: str
    encoding: ValueEncoding
    id_space: str | None = None

    def __post_init__(self) -> None:
        if not self.input or not self.source:
            raise ValueError("a candidate target requires an input and source label")


@dataclass(frozen=True)
class AlignedEventTarget:
    """Fields withheld from, but aligned with, one grouped event-column input.

    The labels follow the input's retained events through filtering and
    clipping, not an independently sampled sequence. Their source fields must
    be withheld from inputs, and must be scalar. Text labels keep flat
    ``values`` and ``piece_offsets`` per real event; padding the event axis adds
    no pretend values.

    :param input: the input these labels are positionally aligned with.
    :param fields: the withheld fields, each with an explicit encoder.
    :param derived: labels computed from a field rather than read from one.
    """

    input: str
    fields: tuple[Encode, ...]
    derived: tuple[ThresholdEventLabel, ...] = ()

    def __post_init__(self) -> None:
        names = (*(value.field for value in self.fields), *(value.name for value in self.derived))
        if not names or len(names) != len(set(names)):
            raise ValueError("aligned event target names must be non-empty and unique")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "derived", tuple(self.derived))


@dataclass(frozen=True)
class MaskedValuesTarget:
    """Predict values hidden from a token input.

    Three independently seeded populations select positions — one token, one
    whole event, and every occurrence of one semantic key — and the selected
    positions are replaced in the input before the model sees it.

    :param input: the token input whose positions are masked.
    :param token_probability: chance of selecting an individual field token.
    :param event_probability: chance of selecting all value tokens in an event.
    :param key_probability: chance of selecting one semantic key everywhere it
        occurs.
    :param unknown_probability: chance a selected position is replaced by the
        unknown ID rather than the mask ID.
    :param fields: fields eligible for masking; empty means all of them.
    :param combine: ``"union"`` supervises a position any population chose;
        ``"intersection"`` requires all three.
    :param ensure_at_least_one: guarantee every non-empty sample has at least
        one supervised position, so no sample contributes zero loss.
    :param supervise_unknown: whether unknown-replaced positions are also
        predicted, or are only context corruption.

    Answers are the input's final IDs, recorded before corruption; only the
    named input changes, using the space's declared ``mask`` and ``unknown``
    roles. Selection uses deterministic sample and field coordinates. Another
    input exposing the same field raises an `InputExposureWarning`; it is not
    masked. Off in inference.
    """

    input: str
    token_probability: float
    event_probability: float
    key_probability: float
    unknown_probability: float = 0.0
    fields: tuple[str, ...] = ()
    combine: str = "union"
    ensure_at_least_one: bool = True
    supervise_unknown: bool = False

    def __post_init__(self) -> None:
        _probability("token probability", self.token_probability)
        _probability("event probability", self.event_probability)
        _probability("key probability", self.key_probability)
        _probability("unknown probability", self.unknown_probability)
        if self.fields and len(self.fields) != len(set(self.fields)):
            raise ValueError("masked fields must be unique")
        if self.combine not in {"union", "intersection"}:
            raise ValueError("mask selections must combine by 'union' or 'intersection'")
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class NextTokenTarget:
    """Shifted token targets, optionally including one target for an entity prefix.

    Shifts already encoded flattened IDs without crossing sample boundaries;
    it never fits another vocabulary. An input can carry at most one
    ``MaskedValuesTarget`` or ``NextTokenTarget``.

    :param input: the flattened token input.
    :param prefix_input: an entity input prepended to the sequence, so the first
        event token is also supervised.
    """

    input: str
    prefix_input: str | None = None


@dataclass(frozen=True)
class MatchingGroupTarget:
    """Stable group IDs derived from selected fields of an entity-value input.

    Groups compare typed raw values, not encoded IDs, so two values that both
    encode as unknown are not merged.

    :param input: the entity-value input.
    :param fields: the fields whose combination defines a group; samples sharing
        one get the same ID.
    """

    input: str
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise ValueError("matching targets require unique field names")


TargetSpec: TypeAlias = (
    LabelTarget
    | StructuredTarget
    | CandidateTarget
    | AlignedEventTarget
    | MaskedValuesTarget
    | NextTokenTarget
    | MatchingGroupTarget
)
