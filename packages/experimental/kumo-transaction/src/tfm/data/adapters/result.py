# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter load result."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tfm.contracts.relational import RelationalDataset
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransferEdge
from tfm.data.splits.temporal import SplitAssignment

if TYPE_CHECKING:
    from tfm.data.adapters.pragmatiq import LabelRow
    from tfm.data.tasks.identity_theft import IdentityTheftSplice


@dataclass(frozen=True)
class AdapterResult:
    """Events plus the optional context a book carries.

    `profiles` are entity-level state (slot 0). `labels` are label tables other
    than the event-level fraud labels already attached to events, keyed by task
    (for example `default_12m`); each row carries its own cutoff.
    `fraud_splice`, when set, holds the fraud task's spliced victim histories;
    `events` stay unspliced (dataset strategy).
    """

    events: tuple[TransactionEvent, ...] = ()
    splits: SplitAssignment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    profiles: tuple[EntityProfile, ...] = ()
    labels: Mapping[str, tuple[LabelRow, ...]] = field(default_factory=dict)
    fraud_splice: IdentityTheftSplice | None = None
    transfers: tuple[TransferEdge, ...] = ()
    relational: RelationalDataset | None = None
    """Optional shared tables/keys/clocks declaration, independent of recipe mode."""
