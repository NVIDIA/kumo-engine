# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, batched source values for the model execution path.

Unlike :class:`BuiltSample`, these containers do not create one Python object
per row or field.  Engines may keep values in their native columnar buffers;
kumo-pytorch does the later encoding into framework tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSequence:
    """One event component flattened across anchors and delimited by offsets."""

    offsets: Any
    timestamps: Any
    fields: Mapping[str, Any]
    order: str = "ascending"


@dataclass(frozen=True)
class ModelRowTarget:
    """One optional structured row per anchor."""

    timestamps: Any
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class ModelSourceBatch:
    """Raw source values retained in typed buffers until model encoding."""

    sample_ids: tuple[str | int, ...]
    entity_ids: tuple[Any, ...]
    evaluation_times: tuple[int, ...]
    entity_type: str
    entity_values: Mapping[str, Any]
    events: Mapping[str, ModelSequence]
    targets: Mapping[str, Any]
    row_targets: Mapping[str, ModelRowTarget]
    splits: tuple[str | None, ...]
    request_metadata: tuple[Mapping[str, Any], ...]
    ordered_events: Mapping[tuple[str, str], ModelSequence] = field(default_factory=dict)
    # Optional native permutation backend; values stay in these Arrow buffers.
    merge_event_streams: Any = None
    # Iterator-produced ordering; keys describe the declared component selection.
    merged_events: Mapping[Any, Any] = field(default_factory=dict)
    runtime_batch: Any = None

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)
