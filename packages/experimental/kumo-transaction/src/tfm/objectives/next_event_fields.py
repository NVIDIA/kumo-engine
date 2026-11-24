# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Next-event field prediction over key-value-time batches (ADR 0012)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tfm.contracts.kvt import KeyValueTimeBatch


@dataclass(frozen=True)
class NextEventFields:
    """Predict every value of event `j` from the decoder state before it.

    Decoder slot 0 holds the profile and slot `j >= 1` holds event `j - 1`, so
    slot `j` has seen the profile and events `0..j-1` and predicts event `j`.
    The last slot has no target. Keys of the target event are given to the
    head as queries; only values are predicted.
    """

    name: str = "next_event_fields"
    ignore_index: int = -100
    label_smoothing: float = 0.0
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            from tfm.contracts.errors import IncompatibleCompositionError

            raise IncompatibleCompositionError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}."
            )

    def required_outputs(self) -> tuple[str, ...]:
        return ("hidden_states", "embeddings", "loss", "losses")

    def target_level(self) -> str:
        return "event"

    def horizon(self) -> str:
        return "next event of the same entity, at or before the sequence cutoff"

    def cutoff_semantics(self) -> str:
        return "inputs and targets are events <= cutoff; the last event has no target"

    def gradient(self) -> str:
        return "through the profile encoder, event encoder, and history decoder"


@dataclass(frozen=True)
class NextEventTargets:
    """Positions and targets selected from an unmasked batch.

    `event`/`token` index the target event's tokens. Event `j` is predicted
    from decoder slot `j`, so `event` also indexes the predicting slot. All
    tensors are 1-D over the selected positions.
    """

    batch: Tensor
    event: Tensor
    token: Tensor
    key_ids: Tensor
    values: Tensor


def next_event_targets(batch: KeyValueTimeBatch) -> NextEventTargets:
    """Every valid token of event `j`, paired with decoder slot `j`.

    Must be given the batch *before* MLM masking: targets are true values.
    """
    valid = batch.token_mask.bool() & batch.event_mask.bool().unsqueeze(-1)
    b, e, t = valid.nonzero(as_tuple=True)
    return NextEventTargets(
        batch=b,
        event=e,
        token=t,
        key_ids=batch.key_ids[b, e, t],
        values=batch.value_ids[b, e, t],
    )
