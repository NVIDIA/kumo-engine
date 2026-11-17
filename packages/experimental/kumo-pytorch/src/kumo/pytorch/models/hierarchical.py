# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical encoder for field tokens, events, and entity histories."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from kumo.pytorch.config import ModelConfig
from kumo.pytorch.data.field_batches import FieldEventBatch
from kumo.pytorch.models.layers import TransformerStack, sinusoidal_positions
from kumo.pytorch.models.protocols import ModelState, model_inputs


def _soft_log_time(seconds: torch.Tensor) -> torch.Tensor:
    return 8.0 * torch.log1p(seconds.clamp_min(0).to(torch.float32) / 8.0)


class HierarchicalFieldEncoder(nn.Module):
    """Summarize fields into events and events into an entity representation.

    The encoder also retains the context at every masked field. A separate
    objective decides how that context should be supervised. Keeping those two
    jobs apart lets the same encoder serve pretraining and labeled tasks.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        vocabulary_size: int,
        pad_id: int,
        entity_token_id: int,
        event_token_id: int,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.output_width = config.d_model
        if len({pad_id, entity_token_id, event_token_id}) != 3:
            raise ValueError("padding, entity, and event summary IDs must be distinct")
        if any(
            value < 0 or value >= vocabulary_size
            for value in (pad_id, entity_token_id, event_token_id)
        ):
            raise ValueError("summary and padding IDs must be inside the vocabulary")
        self.pad_id = pad_id
        self.entity_token_id = entity_token_id
        self.event_token_id = event_token_id
        self.embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.profile_encoder = TransformerStack(
            config.profile_encoder_layers,
            config.d_model,
            config.d_ffn,
            config.attention_heads,
            config.dropout,
        )
        self.event_encoder = TransformerStack(
            config.event_encoder_layers,
            config.d_model,
            config.d_ffn,
            config.attention_heads,
            config.dropout,
        )
        self.history_encoder = TransformerStack(
            config.history_encoder_layers,
            config.d_model,
            config.d_ffn,
            config.attention_heads,
            config.dropout,
        )
        self.calendar_mlp = nn.Sequential(
            nn.Linear(6, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.mlm_projection = nn.Linear(3 * config.d_model, config.d_model)

        self.capture_module_times = False
        self._last_module_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    @property
    def output_embedding(self) -> nn.Embedding:
        """Embedding table reused by the masked-value objective."""

        return self.embedding

    @contextmanager
    def _profile_module(self, name: str, device: torch.device) -> Iterator[None]:
        """Name a trace range and, when enabled, retain its CUDA timing events."""

        events = None
        if self.capture_module_times and device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        with torch.profiler.record_function(f"kumo_pytorch.field_hierarchy.{name}"):
            yield
        if events is not None:
            events[1].record()
            self._last_module_events[name] = events

    def module_times_seconds(self) -> dict[str, float]:
        """Resolve the most recent forward pass after its CUDA stream is synchronized."""

        return {
            f"{name}_seconds": start.elapsed_time(end) / 1_000
            for name, (start, end) in self._last_module_events.items()
        }

    def _embed_pairs(
        self, key_ids: torch.Tensor, value_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.embedding(key_ids) + self.embedding(value_ids)
        positions = sinusoidal_positions(position_ids, self.config.d_model).to(embeddings.dtype)
        return self.embedding_dropout(embeddings + positions)

    def _special(self, token_id: int, batch_size: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((batch_size,), token_id, dtype=torch.long, device=device)
        return self.embedding(ids).unsqueeze(1)

    def _encode_events_padded(
        self,
        batch: FieldEventBatch,
        event_indices: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the rectangular reference representation, including padding."""

        event_count = batch.event_key_ids.shape[0]
        event_pairs = self._embed_pairs(
            batch.event_key_ids, batch.event_value_ids, batch.event_position_ids
        )
        event_inputs = torch.cat(
            (
                self._special(self.event_token_id, event_count, batch.event_key_ids.device),
                event_pairs,
            ),
            dim=1,
        )
        event_mask = torch.cat(
            (
                torch.ones(
                    (event_count, 1),
                    dtype=torch.bool,
                    device=batch.event_key_ids.device,
                ),
                batch.event_token_mask,
            ),
            dim=1,
        )
        event_output = self.event_encoder(event_inputs, event_mask)
        return event_output[:, 0], event_output[event_indices, token_indices + 1]

    def _encode_events_packed(self, batch: FieldEventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only real event tokens using jagged, event-isolated attention."""

        event_count = batch.event_key_ids.shape[0]
        valid = batch.event_token_mask
        event_pairs = self._embed_pairs(
            batch.event_key_ids[valid],
            batch.event_value_ids[valid],
            batch.event_position_ids[valid],
        )
        special_tokens = self._special(
            self.event_token_id, event_count, batch.event_key_ids.device
        ).squeeze(1)
        source = torch.cat((special_tokens, event_pairs), dim=0)
        packed_inputs = source[batch.packed_events.source_indices]
        packed_output = self.event_encoder.forward_packed(
            packed_inputs,
            batch.packed_events.offsets,
            min_length=batch.packed_events.min_sequence_length,
            max_length=batch.packed_events.max_sequence_length,
        )
        event_summary = packed_output[batch.packed_events.offsets[:-1]]
        local_context = packed_output[batch.packed_events.prediction_indices]
        return event_summary, local_context

    def _encode_backbone(
        self,
        batch: FieldEventBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch.profile_key_ids.device
        batch_size = batch.profile_key_ids.shape[0]
        prediction_positions = batch.event_loss_mask.nonzero(as_tuple=False)
        event_indices = prediction_positions[:, 0]
        token_indices = prediction_positions[:, 1]
        self._last_module_events = {}
        with self._profile_module("profile_encoder", device):
            profile_pairs = self._embed_pairs(
                batch.profile_key_ids, batch.profile_value_ids, batch.profile_position_ids
            )
            profile_inputs = torch.cat(
                (self._special(self.entity_token_id, batch_size, device), profile_pairs), dim=1
            )
            profile_mask = torch.cat(
                (
                    torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                    batch.profile_mask,
                ),
                dim=1,
            )
            profile_coordinates = torch.cat(
                (
                    torch.zeros((batch_size, 1), device=device),
                    _soft_log_time(batch.profile_elapsed_seconds),
                ),
                dim=1,
            )
            profile_output = self.profile_encoder(profile_inputs, profile_mask, profile_coordinates)
            anchor_summary = profile_output[:, 0]

        with self._profile_module("event_encoder", device):
            if self.config.event_encoder_layout == "packed":
                event_summary, local_context = self._encode_events_packed(batch)
            else:
                event_summary, local_context = self._encode_events_padded(
                    batch, event_indices, token_indices
                )
            event_summary = event_summary + self.calendar_mlp(
                self._calendar_features(batch.event_calendar)
            )

        with self._profile_module("history_encoder", device):
            history_length = batch.history_event_indices.shape[1]
            history_inputs = torch.zeros(
                (batch_size, history_length + 1, self.config.d_model),
                device=device,
                dtype=event_summary.dtype,
            )
            history_inputs[:, 0] = anchor_summary
            valid_event_slots = batch.history_event_indices >= 0
            history_events = history_inputs[:, 1:]
            history_events[valid_event_slots] = event_summary[
                batch.history_event_indices[valid_event_slots]
            ]
            history_output = self.history_encoder(
                history_inputs,
                batch.history_mask,
                _soft_log_time(batch.history_elapsed_seconds),
            )

        return history_output, local_context, event_indices, token_indices

    def forward(self, wrapped_batch: object) -> ModelState:
        batch = model_inputs(wrapped_batch)
        if not isinstance(batch, FieldEventBatch):
            raise TypeError("hierarchical field encoder requires FieldEventBatch")
        history_output, local_context, event_indices, token_indices = self._encode_backbone(batch)
        device = batch.profile_key_ids.device

        with self._profile_module("mlm_context", device):
            owners = batch.event_owner[event_indices]
            history_positions = batch.event_history_position[event_indices] + 1
            history_context = history_output[owners, history_positions]
            anchor_context = history_output[owners, 0]
            prediction_context = torch.cat((local_context, history_context, anchor_context), dim=-1)
        with self._profile_module("mlm_projection", device):
            prediction_embeddings = self.mlm_projection(prediction_context)
        targets = batch.event_target_ids[event_indices, token_indices]
        return ModelState(
            contextual=history_output,
            mask=batch.history_mask,
            representations={"entity": history_output[:, 0]},
            auxiliary={
                "masked_value_embeddings": prediction_embeddings,
                "masked_value_targets": targets,
            },
        )

    @staticmethod
    def _calendar_features(calendar: torch.Tensor) -> torch.Tensor:
        periods = torch.tensor((24.0, 7.0, 31.0), device=calendar.device)
        offsets = torch.tensor((0.0, 0.0, 1.0), device=calendar.device)
        angles = 2.0 * math.pi * (calendar - offsets) / periods
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
