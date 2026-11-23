# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout-specific tensors for key-value-time tokenization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME


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
    hour: tuple[int, ...]
    dow: tuple[int, ...]
    dom: tuple[int, ...]
    prof_key_ids: tuple[int, ...]
    prof_value_ids: tuple[int, ...]
    prof_positions: tuple[int, ...]
    prof_offsets: tuple[int, ...]
    prof_time_log: tuple[float, ...]
    layout: str = LAYOUT_KEY_VALUE_TIME


@dataclass
class KeyValueTimeBatch:
    """Padded KVT tensors. Not a TransactionBatch; do not pass to TrxFM."""

    key_ids: Tensor
    value_ids: Tensor
    positions: Tensor
    token_mask: Tensor
    event_mask: Tensor
    time_log: Tensor
    hour: Tensor
    dow: Tensor
    dom: Tensor
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
            hour=self.hour.to(device),
            dow=self.dow.to(device),
            dom=self.dom.to(device),
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
