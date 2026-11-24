# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated model input batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TransactionBatch:
    """Tensors and metadata consumed by model code.

    `input_ids` and `attention_mask` are required. Padding positions must be 0
    in `attention_mask`. Labels use -100 for positions ignored by the loss.

    `segment_ids` is set only by the packing collate. When present, row `b` holds
    several concatenated sequences and `segment_ids[b, t]` names the one token
    `t` belongs to; attention is then block-diagonal over those segments.
    """

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor | None = None
    fraud_labels: Tensor | None = None
    numeric_features: Tensor | None = None
    segment_ids: Tensor | None = None
    entity_ids: tuple[str, ...] = ()
    sequence_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def to(self, device: torch.device) -> TransactionBatch:
        return TransactionBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=None if self.labels is None else self.labels.to(device),
            fraud_labels=None if self.fraud_labels is None else self.fraud_labels.to(device),
            numeric_features=(
                None if self.numeric_features is None else self.numeric_features.to(device)
            ),
            segment_ids=None if self.segment_ids is None else self.segment_ids.to(device),
            entity_ids=self.entity_ids,
            sequence_ids=self.sequence_ids,
            event_ids=self.event_ids,
        )
