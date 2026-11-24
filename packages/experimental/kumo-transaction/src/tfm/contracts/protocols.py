# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for composition units."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.outputs import ModelOutput


class AttentionKernel(Protocol):
    """Computes attention over [B, H, T, D] projections.

    Implementations must honor ``attention_mask`` padding and, when ``causal``
    is true, must not let a position attend to a later one.
    """

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        *,
        causal: bool,
        segment_ids: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class AttentionSpec(Protocol):
    """Declarative selector for an attention kernel.

    Composition tokens stay backend-neutral: the spec is a plain value object
    and ``build()`` is the only place a concrete kernel is constructed. This
    keeps model semantics separate from the attention implementation.
    """

    @property
    def name(self) -> str: ...

    def capabilities(self) -> dict[str, bool]: ...

    def build(self) -> nn.Module: ...


class HistoryModel(Protocol):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        causal: bool = True,
    ) -> Tensor: ...


class EventEncoder(Protocol):
    """Encodes each event's embedded key-value tokens into one summary vector.

    `tokens` is `[B, N, T, D]`. Attention stays within an event, so nothing
    crosses events or time. Returns `(summaries [B, N, D], token_states
    [B, N, T, D])`, with padded events' summaries zeroed.
    """

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        event_mask: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class ProfileEncoder(Protocol):
    """Encodes an entity's embedded profile tokens, as of the cutoff, into `[B, D]`.

    `tokens` is `[B, P, D]`. An entity without profile tokens still gets a
    defined vector.
    """

    def forward(self, tokens: Tensor, mask: Tensor, time_log: Tensor) -> Tensor: ...


class SummaryHistoryModel(Protocol):
    """Causal model over per-event summary vectors rather than token ids.

    `summaries` is `[B, S, D]`, `attention_mask` is `[B, S]` with padding
    after valid positions, and `time_positions` is `[B, S]` real-valued time
    that depends only on positions at or before each slot.
    """

    def forward(
        self,
        summaries: Tensor,
        attention_mask: Tensor,
        time_positions: Tensor,
    ) -> Tensor: ...


class PoolingStrategy(Protocol):
    def forward(self, states: Tensor, attention_mask: Tensor) -> Tensor: ...


class Objective(Protocol):
    name: str

    def loss(self, output: ModelOutput, batch: TransactionBatch) -> Tensor: ...


class TaskHead(Protocol):
    name: str

    def forward(self, embeddings: Tensor) -> Tensor: ...
