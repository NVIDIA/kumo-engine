# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed tensor inputs for PRAGMA; construction and encoding belong to the caller.

The packed token contract and tensor-transfer helpers are retained from the
original model; unused frozen-text embedding fields are omitted. Attention processes each segment with native PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PackedBatch:
    """A padding-free batch of tokenized user histories.

    Token-level arrays are concatenated across every event of every user in the
    batch; event-level arrays across every user. The ``cu_seqlens_*`` tensors
    recover the boundaries. ``event_of_token``/``user_of_event`` are convenience
    scatter indices used by the history encoder and MLM head.
    """

    # token level (sum over all events of all users)
    key_ids: torch.Tensor  # int64[T]
    value_ids: torch.Tensor  # int64[T]
    positions: torch.Tensor  # int64[T] within-field position
    cu_seqlens_event: torch.Tensor  # int32[n_events + 1]
    event_of_token: torch.Tensor  # int64[T] -> global event index
    # event level (sum over all users)
    event_ts: torch.Tensor  # int64[E] µs since epoch (raw event timestamp)
    event_time_log: torch.Tensor  # float32[E] log-seconds to last event
    event_hour: torch.Tensor  # int64[E]
    event_dow: torch.Tensor  # int64[E]
    event_dom: torch.Tensor  # int64[E]
    event_source: torch.Tensor  # int64[E]
    cu_seqlens_history: torch.Tensor  # int32[n_users + 1] event boundaries per user
    user_of_event: torch.Tensor  # int64[E] -> batch user index
    # profile level
    prof_key_ids: torch.Tensor  # int64[P]
    prof_value_ids: torch.Tensor  # int64[P]
    prof_positions: torch.Tensor  # int64[P]
    prof_time_log: torch.Tensor  # float32[n_prof_items]
    cu_seqlens_profile_item: torch.Tensor  # int32[n_prof_items + 1] tokens per profile item
    cu_seqlens_profile: torch.Tensor  # int32[n_users + 1] profile items per user
    item_of_prof_token: torch.Tensor  # int64[P] -> global profile-item index
    user_of_prof_item: torch.Tensor  # int64[n_prof_items]
    # bookkeeping
    user_ids: list[str]
    n_events_per_user: torch.Tensor  # int64[n_users]
    # Optional legacy length metadata, INCLUDING the [EVT]/[USR] prefix.
    # No padding plan depends on these values. If supplied, attention validates
    # that they cover the actual lengths; leaving them None is recommended.
    max_len_event: int | None = None
    max_len_history: int | None = None
    max_len_profile: int | None = None
    # Optional caller bookkeeping (tokens, events, users, profile items).
    # Retained for batch compatibility; it does not control model execution.
    real_sizes: tuple[int, int, int, int] | None = None

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_events(self) -> int:
        return int(self.cu_seqlens_event.numel() - 1)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.numel())

    def to(self, device: torch.device | str, non_blocking: bool = False) -> PackedBatch:
        """Move all tensors to ``device`` (lists and ints are left as-is).

        ``non_blocking=True`` overlaps the host→device copy with compute when the
        source tensors are pinned (see :meth:`pin_memory`); it is a no-op otherwise.
        """
        moved: dict[str, Any] = {
            f: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v)
            for f, v in self.__dict__.items()
        }
        return PackedBatch(**moved)

    def pin_memory(self) -> PackedBatch:
        """Copy every tensor into page-locked host memory (for async H2D copies)."""
        pinned: dict[str, Any] = {
            f: (v.pin_memory() if isinstance(v, torch.Tensor) else v)
            for f, v in self.__dict__.items()
        }
        return PackedBatch(**pinned)

    def token_budget(self) -> int:
        """Total tokens (events + profile) — the dynamic-batch budget metric."""
        return self.n_tokens + int(self.prof_key_ids.numel())
