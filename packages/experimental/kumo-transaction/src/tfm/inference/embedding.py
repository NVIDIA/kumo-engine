# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding extraction with alignment metadata, for every stack.

Any model whose `forward(batch, compute_loss=False)` returns `embeddings` and
that exposes `hidden_size` works here: `TrxFM` and both hierarchical models.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from tfm.adaptation.batching import minibatches
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError


@dataclass
class EmbeddingTable:
    vectors: NDArray[np.float32]
    entity_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    fraud_labels: NDArray[np.float32]
    numeric_features: NDArray[np.float32]

    def select(self, rows: list[int]) -> EmbeddingTable:
        """The given rows, every field kept aligned."""
        return EmbeddingTable(
            vectors=self.vectors[rows],
            entity_ids=tuple(self.entity_ids[i] for i in rows),
            sequence_ids=tuple(self.sequence_ids[i] for i in rows),
            event_ids=tuple(self.event_ids[i] for i in rows),
            fraud_labels=self.fraud_labels[rows] if len(self.fraud_labels) else self.fraud_labels,
            numeric_features=(
                self.numeric_features[rows] if len(self.numeric_features) else self.numeric_features
            ),
        )


def _numpy(tensor: torch.Tensor | None, empty_shape: tuple[int, ...]) -> NDArray[np.float32]:
    if tensor is None:
        return np.zeros(empty_shape, np.float32)
    return tensor.detach().cpu().numpy().astype(np.float32)


def trim_padding[BatchT](chunk: BatchT) -> BatchT:
    """Drop trailing token columns that no row attends to.

    Applies to right-padded, unpacked `TransactionBatch` rows only; any other
    batch is returned as is. Padding is masked, so the kept positions compute
    the same states either way.
    """
    if not isinstance(chunk, TransactionBatch) or chunk.segment_ids is not None:
        return chunk
    mask = chunk.attention_mask
    if mask.dim() != 2 or mask.shape[1] < 2 or not bool((mask[:, :-1] >= mask[:, 1:]).all()):
        return chunk
    width = max(1, int(mask.sum(dim=1).max()))
    if width == mask.shape[1]:
        return chunk
    labels = None if chunk.labels is None else chunk.labels[:, :width]
    return cast(
        BatchT,
        dataclasses.replace(
            chunk,
            input_ids=chunk.input_ids[:, :width],
            attention_mask=mask[:, :width],
            labels=labels,
        ),
    )


def extract_embeddings(
    model: nn.Module,
    batch: Any,
    *,
    device: torch.device,
    minibatch_size: int = 8,
    precision: Literal["fp32", "bf16"] = "fp32",
) -> EmbeddingTable:
    """One embedding per batch row, aligned with its ids, label, and raw features.

    `precision="bf16"` autocasts the forward pass; vectors are returned as fp32.
    """
    hidden = int(cast(Any, model).hidden_size)
    model.eval()
    model.to(device)
    chunks: list[np.ndarray] = []
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
    ):
        for chunk in minibatches(batch, minibatch_size):
            output = model(trim_padding(chunk).to(device), compute_loss=False)
            if output.embeddings is None:
                raise IncompatibleCompositionError(
                    f"{type(model).__name__} returned no embeddings to extract."
                )
            chunks.append(output.embeddings.detach().float().cpu().numpy())
    vectors = (
        np.concatenate(chunks, axis=0).astype(np.float32)
        if chunks
        else np.zeros((0, hidden), np.float32)
    )
    event_ids = tuple(getattr(batch, "event_ids", ()) or ("" for _ in batch.entity_ids))
    return EmbeddingTable(
        vectors=vectors,
        entity_ids=tuple(batch.entity_ids),
        sequence_ids=tuple(batch.sequence_ids),
        event_ids=event_ids,
        fraud_labels=_numpy(batch.fraud_labels, (0,)),
        numeric_features=_numpy(batch.numeric_features, (0, 0)),
    )
