# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin test adapters from public batches to the reference models.

Real users own this boundary: their model reads ``Batch.inputs`` and
``Batch.targets`` directly. These adapters route explicitly selected public
arrays to model arguments. They never fit encoders or select rows.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch

from kumo.pytorch.api import Batch
from kumo.pytorch.data.field_batches import FieldEventBatch, build_packed_event_layout
from kumo.pytorch.data.sequences import TokenSequenceBatch
from kumo.pytorch.data.typed_vectors import TypedVectorBatch


def _unpack(values: torch.Tensor, offsets: torch.Tensor, fill: int = 0) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    width = int(lengths.max()) if len(lengths) else 0
    result = torch.full((len(lengths), width), fill, dtype=values.dtype)
    for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        result[row, : int(stop - start)] = values[int(start) : int(stop)]
    return result


def pragma_batch(batch: Batch) -> FieldEventBatch:
    entity = batch.inputs["entity"]
    events = batch.inputs["history"]
    history = events["sequence"]
    target = batch.targets.get(
        "masked_values",
        {
            "values": torch.full_like(events["value_ids"], -100),
            "valid": torch.zeros_like(events["valid"]),
        },
    )
    profile_timing = batch.inputs["profile_timing"]
    timing = batch.inputs["history_timing"]
    calendar = batch.inputs["history_calendar"]
    if "offsets" in entity:
        profile_keys = _unpack(entity["field_ids"], entity["offsets"])
        profile_values = _unpack(entity["value_ids"], entity["offsets"])
        profile_positions = _unpack(entity["position_ids"], entity["offsets"])
        profile_elapsed = _unpack(
            profile_timing["age_seconds"]["values"], profile_timing["offsets"]
        )
        profile_valid = _unpack(entity["valid"], entity["offsets"]).bool()
    else:
        profile_keys = entity["field_ids"]
        profile_values = entity["value_ids"]
        profile_positions = entity["position_ids"]
        profile_elapsed = profile_timing["age_seconds"]["values"]
        profile_valid = entity["valid"]
    if "field_offsets" in events:
        offsets = events["field_offsets"]
        event_keys = _unpack(events["field_ids"], offsets)
        event_values = _unpack(events["value_ids"], offsets)
        event_positions = _unpack(events["position_ids"], offsets)
        event_mask = _unpack(events["valid"], offsets).bool()
        target_ids = _unpack(target["values"], offsets, fill=-100)
        target_valid = _unpack(target["valid"], offsets).bool()
    else:
        event_keys = events["field_ids"]
        event_values = events["value_ids"]
        event_positions = events["position_ids"]
        event_mask = events["valid"]
        target_ids = target["values"]
        target_valid = target["valid"]
    if "offsets" in history:
        history_indices = _unpack(history["event_indices"], history["offsets"], fill=-1)
        history_valid = _unpack(history["valid"], history["offsets"]).bool()
    else:
        history_indices = history["event_indices"]
        history_valid = history["valid"]
    anchor_ages = timing["age_seconds"]["values"]
    age_valid = timing["age_seconds"]["valid"]
    if "offsets" in timing:
        anchor_ages = _unpack(anchor_ages, timing["offsets"])
        age_valid = _unpack(age_valid, timing["offsets"]).bool()
    # PRAGMA's time origin is the latest retained event, not the anchor.
    # Subtract in float64 before the model transforms/casts time coordinates.
    anchor_ages = anchor_ages.to(torch.float64)
    if anchor_ages.shape[1]:
        latest_age = anchor_ages.masked_fill(~age_valid, float("inf")).amin(dim=1)
        latest_age = torch.where(age_valid.any(dim=1), latest_age, 0)
        history_elapsed = torch.where(age_valid, anchor_ages - latest_age[:, None], 0)
    else:
        history_elapsed = anchor_ages
    entity_ids = torch.as_tensor(batch.metadata["entity_ids"], dtype=torch.long)
    batch_size = entity_ids.numel()
    history_elapsed = torch.cat(
        (history_elapsed.new_zeros((batch_size, 1)), history_elapsed), dim=1
    )
    history_valid = torch.cat((torch.ones((batch_size, 1), dtype=torch.bool), history_valid), dim=1)
    event_calendar = torch.stack(
        [
            calendar[name]["values"]
            if "offsets" in calendar
            else calendar[name]["values"][events["owner"], events["history_position"]]
            for name in ("hour", "weekday", "day")
        ],
        dim=-1,
    )
    return FieldEventBatch(
        entity_ids=entity_ids,
        profile_key_ids=profile_keys,
        profile_value_ids=profile_values,
        profile_position_ids=profile_positions,
        profile_elapsed_seconds=profile_elapsed,
        profile_mask=profile_valid,
        event_key_ids=event_keys,
        event_value_ids=event_values,
        event_target_ids=target_ids,
        event_position_ids=event_positions,
        event_token_mask=event_mask,
        event_loss_mask=target_valid,
        event_calendar=event_calendar,
        event_owner=events["owner"],
        event_history_position=events["history_position"],
        history_event_indices=history_indices,
        history_elapsed_seconds=history_elapsed,
        history_mask=history_valid,
        packed_events=build_packed_event_layout(event_mask, target_valid),
    )


def _token_batch(
    batch: Batch,
    *,
    target_name: str,
    group_name: str | None = None,
    entity_name: str | None = None,
) -> TokenSequenceBatch:
    source = batch.inputs["history"]
    target = batch.targets.get(target_name, {"values": torch.full_like(source["value_ids"], -100)})
    if "offsets" in source:
        inputs = _unpack(source["value_ids"], source["offsets"])
        values = _unpack(target["values"], target.get("offsets", source["offsets"]), fill=-100)
        valid = _unpack(source["valid"], source["offsets"]).bool()
    else:
        inputs = source["value_ids"]
        values = target["values"]
        valid = source["valid"]
    if entity_name is None:
        profile_ids = torch.zeros((batch.batch_size, 1), dtype=torch.long)
        profile_valid = torch.zeros_like(profile_ids, dtype=torch.bool)
    else:
        entity = batch.inputs[entity_name]
        profile_ids = torch.stack(
            [field["values"] for field in entity.values()],
            dim=1,
        )
        profile_valid = torch.stack(
            [field["valid"] for field in entity.values()],
            dim=1,
        )
        inputs = torch.cat((torch.zeros((batch.batch_size, 1), dtype=torch.long), inputs), dim=1)
        valid = torch.cat(
            (torch.ones((batch.batch_size, 1), dtype=torch.bool), valid),
            dim=1,
        )
    groups = (
        batch.targets[group_name]["values"]
        if group_name is not None and group_name in batch.targets
        else torch.full((batch.batch_size,), -1, dtype=torch.long)
    )
    return TokenSequenceBatch(
        entity_ids=torch.as_tensor(batch.metadata["entity_ids"], dtype=torch.long),
        input_ids=inputs,
        target_ids=values,
        token_mask=valid,
        profile_ids=profile_ids,
        profile_mask=profile_valid,
        contrastive_groups=groups,
    )


def nuformer_batch(batch: Batch) -> TokenSequenceBatch:
    return _token_batch(batch, target_name="next_token")


def panther_batch(batch: Batch) -> TokenSequenceBatch:
    return _token_batch(
        batch,
        target_name="next_behavior",
        group_name="entity_match",
        entity_name="entity",
    )


def treasure_batch(batch: Batch) -> TypedVectorBatch:
    entity = batch.inputs["entity"]
    events = batch.inputs["events"]
    current = batch.targets["current_event"]
    static_numerical = entity["active_account_count"]
    static_categorical = entity["country"]
    dynamic_numerical = events["amount"]
    dynamic_categories = (
        "channel",
        "country",
        "currency",
        "entry_mode",
        "mcc",
        "recurring_candidate_flag",
    )
    signal_categories = ("status", "synthetic_abnormal")
    batch_size, event_count = events["valid"].shape
    return TypedVectorBatch(
        entity_ids=batch.metadata["entity_ids"],
        static_numerical=static_numerical["values"].float().unsqueeze(-1),
        static_numerical_present=static_numerical["valid"].unsqueeze(-1),
        static_categorical=static_categorical["values"].unsqueeze(-1),
        dynamic_numerical=dynamic_numerical["values"].float().unsqueeze(-1),
        dynamic_numerical_present=dynamic_numerical["valid"].unsqueeze(-1),
        dynamic_categorical=torch.stack(
            [events[name]["values"] for name in dynamic_categories],
            dim=-1,
        ),
        dynamic_categorical_present=torch.stack(
            [events[name]["valid"] for name in dynamic_categories],
            dim=-1,
        ),
        signal_numerical=torch.empty((batch_size, event_count, 0)),
        signal_numerical_present=torch.empty((batch_size, event_count, 0), dtype=torch.bool),
        signal_categorical=torch.stack(
            [current[name]["values"] for name in signal_categories],
            dim=-1,
        ),
        signal_categorical_present=torch.stack(
            [current[name]["valid"] for name in signal_categories],
            dim=-1,
        ),
        event_mask=events["valid"],
    )


MODEL_BATCH_ADAPTERS: Mapping[str, Callable[[Batch], Any]] = {
    "pragma": pragma_batch,
    "nuformer": nuformer_batch,
    "panther": panther_batch,
    "treasure": treasure_batch,
}
