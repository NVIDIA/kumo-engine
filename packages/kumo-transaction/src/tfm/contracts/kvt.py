# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout-specific tensors for key-value-time tokenization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME
from tfm.contracts.training import TrainableBatch


@dataclass(frozen=True)
class EncodedKVTRecord:
    """One entity's key-value-time encoding (unpadded)."""

    entity_id: str
    sequence_id: str
    key_ids: tuple[int, ...]
    value_ids: tuple[int, ...]
    positions: tuple[int, ...]
    event_offsets: tuple[int, ...]
    event_ts_us: tuple[int, ...]
    time_log: tuple[float, ...]
    events: Mapping[str, tuple[int, ...]]
    """Per-event integer arrays by name (the calendar's `hour`, `dow`, `dom`), one value
    per event."""
    prof_key_ids: tuple[int, ...]
    prof_value_ids: tuple[int, ...]
    prof_positions: tuple[int, ...]
    prof_offsets: tuple[int, ...]
    prof_time_log: tuple[float, ...]
    layout: str = LAYOUT_KEY_VALUE_TIME


@dataclass
class KeyValueTimeBatch(TrainableBatch):
    """Padded KVT tensors. Not a TransactionBatch; do not pass to TrxFM."""

    # Graph execution views retain raw probe features/duplicate labels on CPU.
    execution_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "key_ids",
            "value_ids",
            "positions",
            "token_mask",
            "event_mask",
            "time_log",
            "events",
            "prof_key_ids",
            "prof_value_ids",
            "prof_positions",
            "prof_mask",
            "prof_time_log",
            "mlm_labels",
            "elapsed_log",
        }
    )

    key_ids: Tensor
    value_ids: Tensor
    positions: Tensor
    token_mask: Tensor
    event_mask: Tensor
    time_log: Tensor
    events: dict[str, Tensor]
    """Per-event arrays by name, each `[B, N]` int64, 0 on padded events. A model
    reads them with `event_array`."""
    prof_key_ids: Tensor
    prof_value_ids: Tensor
    prof_positions: Tensor
    prof_mask: Tensor
    prof_time_log: Tensor
    entity_ids: tuple[str, ...] = ()
    sequence_ids: tuple[str, ...] = ()
    fraud_labels: Tensor | None = None
    numeric_features: Tensor | None = None
    mlm_labels: Tensor | None = None
    elapsed_log: Tensor | None = None
    """`[B, N]` log-time since the sequence's first event, `8·ln(1 + Δt/8)`.

    `time_log` is measured back from the last event, so under a causal mask
    event `i`'s value would reveal the gap to later events. `elapsed_log`
    depends only on events up to `i` and is what a causal consumer uses.
    """

    @property
    def batch_size(self) -> int:
        return int(self.key_ids.shape[0])

    def event_array(self, name: str) -> Tensor:
        """The per-event array `name`; refuses a batch without it."""
        array = self.events.get(name)
        if array is None:
            raise IncompatibleCompositionError(
                f"The model reads the per-event array {name!r}, but this batch carries "
                f"{sorted(self.events)}. Use a tokenizer that emits it."
            )
        return array

    def check_event_arrays(self, names: Sequence[str]) -> None:
        """Raise unless the batch carries every per-event array in `names`."""
        if missing := [name for name in names if name not in self.events]:
            raise IncompatibleCompositionError(
                f"The model reads the per-event arrays {missing}, which this tokenizer does "
                f"not emit (it emits {sorted(self.events)}). Use a tokenizer that emits them, "
                "or a model that does not read them."
            )

    def to(self, device: torch.device) -> KeyValueTimeBatch:
        def _move(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return KeyValueTimeBatch(
            key_ids=self.key_ids.to(device),
            value_ids=self.value_ids.to(device),
            positions=self.positions.to(device),
            token_mask=self.token_mask.to(device),
            event_mask=self.event_mask.to(device),
            time_log=self.time_log.to(device),
            events={name: array.to(device) for name, array in self.events.items()},
            prof_key_ids=self.prof_key_ids.to(device),
            prof_value_ids=self.prof_value_ids.to(device),
            prof_positions=self.prof_positions.to(device),
            prof_mask=self.prof_mask.to(device),
            prof_time_log=self.prof_time_log.to(device),
            entity_ids=self.entity_ids,
            sequence_ids=self.sequence_ids,
            fraud_labels=_move(self.fraud_labels),
            numeric_features=_move(self.numeric_features),
            mlm_labels=_move(self.mlm_labels),
            elapsed_log=_move(self.elapsed_log),
        )


@dataclass(frozen=True)
class CyclicTime:
    """A per-event array read as a point on a circle: angle 2π (value − origin) / period."""

    array: str
    period: float
    origin: float = 0.0

    def __post_init__(self) -> None:
        if not self.period > 0:
            raise IncompatibleCompositionError(
                f"time feature {self.array!r} needs a positive period, not {self.period}."
            )


CALENDAR = (CyclicTime("hour", 24.0), CyclicTime("dow", 7.0), CyclicTime("dom", 31.0, 1.0))
"""Hour of day, day of week (Monday 0) and day of month (from 1), as the key-value-time
tokenizer emits them."""


@runtime_checkable
class EventArrayReader(Protocol):
    """A model that reads per-event arrays from `KeyValueTimeBatch.events`; a run refuses
    a token cache that lacks one before the first step."""

    event_arrays: tuple[str, ...]
