# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch declarations: sources, explicit encodings, arrangements, and layouts."""

from kumo.pytorch.contract.batch.encodings import (
    Buckets,
    Raw,
    ScalarEncoding,
    Text,
    UseEncoding,
    ValueEncoding,
    Vocabulary,
    ZScore,
)
from kumo.pytorch.contract.batch.layouts import Nested, Packed, Padded, Separate, SequenceLayout
from kumo.pytorch.contract.batch.representations import Encode, Fields, Graph, Representation
from kumo.pytorch.contract.batch.sources import (
    CandidateSource,
    EntitySource,
    EventsOf,
    EventSource,
    InputSource,
    NeighborhoodSource,
    OccurrencesOf,
)
from kumo.pytorch.contract.batch.spec import Batch, Batching, BatchSpec, Input, InputSpec
from kumo.pytorch.contract.batch.targets import (
    AlignedEventTarget,
    CandidateTarget,
    LabelTarget,
    MaskedValuesTarget,
    MatchingGroupTarget,
    NextTokenTarget,
    StructuredTarget,
    TargetSpec,
    ThresholdEventLabel,
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

__all__ = [
    "AlignedEventTarget",
    "Batch",
    "Batching",
    "BatchSpec",
    "Buckets",
    "CandidateSource",
    "CandidateTarget",
    "Columns",
    "Composite",
    "Encode",
    "EntitySource",
    "EventSource",
    "EventsOf",
    "FieldNames",
    "Fields",
    "Flatten",
    "Graph",
    "Grouped",
    "IdSpace",
    "Input",
    "InputSource",
    "InputSpec",
    "LabelTarget",
    "MaskedValuesTarget",
    "MatchingGroupTarget",
    "NeighborhoodSource",
    "OccurrencesOf",
    "Nested",
    "NextTokenTarget",
    "Packed",
    "Padded",
    "Raw",
    "Representation",
    "ScalarEncoding",
    "Separate",
    "SequenceLayout",
    "StructuredTarget",
    "TargetSpec",
    "Text",
    "ThresholdEventLabel",
    "Tokens",
    "UseEncoding",
    "ValueEncoding",
    "Vocabulary",
    "ZScore",
]
