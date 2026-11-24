# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-sequence contrastive learning, after CoLES (experimental auxiliary)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import KeyValueTimeBatch


@dataclass(frozen=True)
class SubsequenceContrastive:
    """Two random contiguous crops of one history should embed close together.

    Each crop keeps `crop_fraction` of an entity's events, re-timed from the
    crop's first event, and is embedded with the model's readout. The loss is
    symmetric InfoNCE: a crop's positive is the other crop of the same entity,
    and every other entity in the batch is a negative. CoLES and Plaid's
    temporal contrastive objective use this idea. Auxiliary to
    `NextEventFields` on `ProfileEventDecoder`.
    """

    name: str = "subsequence_contrastive"
    crop_fraction: float = 0.7
    temperature: float = 0.1
    weight: float = 1.0
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if not 0.0 < self.crop_fraction <= 1.0:
            raise IncompatibleCompositionError(
                f"crop_fraction must be in (0, 1], got {self.crop_fraction}."
            )
        if self.temperature <= 0.0:
            raise IncompatibleCompositionError(
                f"temperature must be positive, got {self.temperature}."
            )

    def required_outputs(self) -> tuple[str, ...]:
        return ("embeddings", "loss", "losses")

    def target_level(self) -> str:
        return "sequence"

    def horizon_semantics(self) -> str:
        return "none: both views are sub-sequences of the same history"

    def cutoff_semantics(self) -> str:
        return "each crop holds only events at or before the sequence cutoff"

    def gradient(self) -> str:
        return "through the readout, history decoder, and both encoders, for both views"

    def leakage_checks(self) -> str:
        return "uses no labels; crops never add events, so no future event is introduced"


def _inverse_time_encode(encoded: Tensor) -> Tensor:
    """Seconds from `8·ln(1 + Δ/8)`."""
    return 8.0 * torch.expm1(encoded / 8.0)


def random_crop(
    batch: KeyValueTimeBatch, fraction: float, *, generator: torch.Generator | None = None
) -> KeyValueTimeBatch:
    """One contiguous crop per row, shifted to start at event 0 and re-timed.

    `elapsed_log` is recomputed from the crop's first event, so a crop looks
    like a history that began there. `time_log` is left as is: causal
    consumers do not read it.
    """
    if batch.elapsed_log is None:
        raise ValueError("SubsequenceContrastive needs KeyValueTimeBatch.elapsed_log.")
    lengths = batch.event_mask.long().sum(dim=1)
    n_rows, n_events = batch.event_mask.shape
    device = batch.event_mask.device
    keep = torch.clamp((lengths.float() * fraction).ceil().long(), min=1)
    keep = torch.minimum(keep, lengths.clamp(min=1))
    room = (lengths - keep).clamp(min=0)
    draw = torch.rand(n_rows, generator=generator, device=device)
    start = torch.floor(draw * (room + 1).float()).long().clamp(max=room)
    offsets = torch.arange(n_events, device=device).unsqueeze(0)
    source = (start.unsqueeze(1) + offsets).clamp(max=n_events - 1)
    kept = offsets < keep.unsqueeze(1)
    kept &= lengths.unsqueeze(1) > 0

    def gather_events(tensor: Tensor) -> Tensor:
        index = source.view(n_rows, n_events, *([1] * (tensor.dim() - 2)))
        index = index.expand_as(tensor)
        moved = torch.gather(tensor, 1, index)
        mask = kept.view(n_rows, n_events, *([1] * (tensor.dim() - 2))).to(moved.dtype)
        return moved * mask

    seconds = _inverse_time_encode(batch.elapsed_log)
    moved_seconds = torch.gather(seconds, 1, source)
    first = moved_seconds[:, :1]
    elapsed = 8.0 * torch.log1p((moved_seconds - first).clamp(min=0.0) / 8.0)
    elapsed = elapsed * kept.to(elapsed.dtype)
    return replace(
        batch,
        key_ids=gather_events(batch.key_ids),
        value_ids=gather_events(batch.value_ids),
        positions=gather_events(batch.positions),
        token_mask=gather_events(batch.token_mask),
        event_mask=kept.to(batch.event_mask.dtype),
        time_log=gather_events(batch.time_log),
        events={name: gather_events(array) for name, array in batch.events.items()},
        elapsed_log=elapsed,
        mlm_labels=None,
    )


def info_nce(first: Tensor, second: Tensor, temperature: float) -> Tensor:
    """Symmetric InfoNCE between two views; row `i` of each is a positive pair."""
    left = F.normalize(first.float(), dim=-1)
    right = F.normalize(second.float(), dim=-1)
    logits = left @ right.t() / temperature
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
