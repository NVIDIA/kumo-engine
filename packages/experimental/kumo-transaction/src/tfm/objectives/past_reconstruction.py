# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Past-event reconstruction, after Featurespace's NPPR (experimental auxiliary)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import KeyValueTimeBatch


@dataclass(frozen=True)
class PastReconstruction:
    """Rebuild the values of earlier events from the state that has seen them.

    Decoder slot `j + 1` holds event `j` and has seen events `0..j`. For each
    lag `k` in `1..horizon`, the head reads that slot, the key of each target
    token, and the time gap `elapsed_log[j] - elapsed_log[j - k]`, and predicts
    the values of event `j - k`. Next-event prediction teaches the state to
    anticipate; this teaches it to retain, which NPPR reports helps downstream
    probes. Auxiliary to `NextEventFields` on `ProfileEventDecoder`.
    """

    name: str = "past_reconstruction"
    horizon: int = 3
    weight: float = 1.0
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise IncompatibleCompositionError(
                f"PastReconstruction.horizon must be >= 1, got {self.horizon}."
            )

    def required_outputs(self) -> tuple[str, ...]:
        return ("hidden_states", "loss", "losses")

    def target_level(self) -> str:
        return "event"

    def horizon_semantics(self) -> str:
        return "the `horizon` events before the current one, all at or before the cutoff"

    def cutoff_semantics(self) -> str:
        return "targets are events the predicting slot has already seen; nothing after it"

    def gradient(self) -> str:
        return "through the history decoder, both encoders, and the tied embedding"

    def leakage_checks(self) -> str:
        return (
            "reads only slots after their targets under the causal mask, so it adds "
            "no path from future events; tested by the causal-invariance contract"
        )


@dataclass(frozen=True)
class PastTargets:
    """Tokens of event `e`, predicted from the slot after event `e + lag`.

    All tensors are 1-D over the selected positions.
    """

    batch: Tensor
    event: Tensor
    slot: Tensor
    key_ids: Tensor
    values: Tensor
    lag: Tensor
    gap: Tensor


def past_targets(batch: KeyValueTimeBatch, lag: int) -> PastTargets:
    """Every valid token of event `e` whose event `e + lag` is also valid.

    Must be given the unmasked batch: targets are true values.
    """
    import torch

    if batch.elapsed_log is None:
        raise ValueError("PastReconstruction needs KeyValueTimeBatch.elapsed_log.")
    n_events = batch.event_mask.shape[1]
    later = torch.zeros_like(batch.event_mask, dtype=torch.bool)
    if lag < n_events:
        later[:, : n_events - lag] = batch.event_mask[:, lag:].bool()
    valid = batch.token_mask.bool() & (batch.event_mask.bool() & later).unsqueeze(-1)
    b, e, t = valid.nonzero(as_tuple=True)
    gap = batch.elapsed_log[b, e + lag] - batch.elapsed_log[b, e]
    return PastTargets(
        batch=b,
        event=e,
        slot=e + lag + 1,
        key_ids=batch.key_ids[b, e, t],
        values=batch.value_ids[b, e, t],
        lag=torch.full_like(b, lag),
        gap=gap,
    )
