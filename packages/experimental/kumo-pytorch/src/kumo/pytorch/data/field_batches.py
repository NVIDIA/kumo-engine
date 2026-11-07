# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic masked-field batches with padded and packed event layouts."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class PackedEventLayout:
    """Indices and boundaries needed to encode events without padding."""

    source_indices: torch.Tensor
    offsets: torch.Tensor
    prediction_indices: torch.Tensor
    min_sequence_length: int
    max_sequence_length: int

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PackedEventLayout:
        return PackedEventLayout(
            source_indices=self.source_indices.to(device, non_blocking=non_blocking),
            offsets=self.offsets.to(device, non_blocking=non_blocking),
            prediction_indices=self.prediction_indices.to(device, non_blocking=non_blocking),
            min_sequence_length=self.min_sequence_length,
            max_sequence_length=self.max_sequence_length,
        )

    def pin_memory(self) -> PackedEventLayout:
        return PackedEventLayout(
            source_indices=self.source_indices.pin_memory(),
            offsets=self.offsets.pin_memory(),
            prediction_indices=self.prediction_indices.pin_memory(),
            min_sequence_length=self.min_sequence_length,
            max_sequence_length=self.max_sequence_length,
        )


def build_packed_event_layout(
    event_token_mask: torch.Tensor,
    event_loss_mask: torch.Tensor,
) -> PackedEventLayout:
    """Derive compact event boundaries and lookups from the two logical masks."""

    if event_token_mask.shape != event_loss_mask.shape:
        raise ValueError("event token and loss masks must have the same shape")
    event_count = event_token_mask.shape[0]
    event_lengths = event_token_mask.sum(dim=1)
    if bool((event_lengths == 0).any()):
        raise ValueError("cannot pack events without value tokens")

    packed_lengths = event_lengths + 1
    packed_offsets = torch.zeros(
        event_count + 1,
        dtype=torch.long,
        device=event_token_mask.device,
    )
    packed_offsets[1:] = packed_lengths.cumsum(dim=0)
    packed_size = int(packed_offsets[-1])
    special_positions = packed_offsets[:-1]
    token_positions = torch.ones(
        packed_size,
        dtype=torch.bool,
        device=event_token_mask.device,
    )
    token_positions[special_positions] = False
    source_indices = torch.empty(
        packed_size,
        dtype=torch.long,
        device=event_token_mask.device,
    )
    source_indices[special_positions] = torch.arange(
        event_count,
        device=event_token_mask.device,
    )
    source_indices[token_positions] = torch.arange(
        event_count,
        event_count + int(event_token_mask.sum()),
        device=event_token_mask.device,
    )
    prediction_positions = event_loss_mask.nonzero(as_tuple=False)
    packed_prediction_indices = (
        packed_offsets[prediction_positions[:, 0]] + prediction_positions[:, 1] + 1
    )
    return PackedEventLayout(
        source_indices=source_indices,
        offsets=packed_offsets,
        prediction_indices=packed_prediction_indices,
        min_sequence_length=int(packed_lengths.min()),
        max_sequence_length=int(packed_lengths.max()),
    )


@dataclass
class FieldEventBatch:
    """All tensors needed for one hierarchical field-model forward pass.

    Events are flattened across entities. ``event_owner`` and
    ``event_history_position`` reconnect each event to its entity and chronological
    slot; ``history_event_indices`` provides the inverse map used by the history
    encoder. The ordinary event tensors stay rectangular as a readable reference,
    while ``packed_events`` identifies only the positions used by packed attention.
    """

    entity_ids: torch.Tensor
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_position_ids: torch.Tensor
    profile_elapsed_seconds: torch.Tensor
    profile_mask: torch.Tensor
    event_key_ids: torch.Tensor
    event_value_ids: torch.Tensor
    event_target_ids: torch.Tensor
    event_position_ids: torch.Tensor
    event_token_mask: torch.Tensor
    event_loss_mask: torch.Tensor
    event_calendar: torch.Tensor
    event_owner: torch.Tensor
    event_history_position: torch.Tensor
    history_event_indices: torch.Tensor
    history_elapsed_seconds: torch.Tensor
    history_mask: torch.Tensor
    packed_events: PackedEventLayout

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> FieldEventBatch:
        values = {
            field.name: getattr(self, field.name).to(
                device,
                non_blocking=non_blocking,
            )
            for field in fields(self)
        }
        return FieldEventBatch(**values)

    def pin_memory(self) -> FieldEventBatch:
        values = {field.name: getattr(self, field.name).pin_memory() for field in fields(self)}
        return FieldEventBatch(**values)

    @property
    def prediction_count(self) -> int:
        return int(self.event_loss_mask.sum().item())

    @property
    def batch_size(self) -> int:
        return int(self.entity_ids.numel())

    @property
    def token_count(self) -> int:
        return int(self.profile_mask.sum().item() + self.event_token_mask.sum().item())
