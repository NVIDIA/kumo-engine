# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the declared Kumo outputs into Blueprint's KeyValueTimeBatch."""

from collections.abc import Mapping
from typing import Any

import torch
from kumo.pytorch import api
from torch import Tensor

from contract import (
    FIRST_TRANSACTION,
    FIRST_TRANSACTION_AGE,
    PROFILE_FIELDS,
    TIME_FIELDS,
)
from tfm.contracts.kvt import KeyValueTimeBatch


def pad_events(value: Tensor) -> Tensor:
    """Keep an all-empty batch representable with a masked storage slot."""
    return value if value.shape[1] else value.new_zeros((value.shape[0], 1, *value.shape[2:]))


class KumoToKVT:
    def __init__(self, task: api.Task) -> None:
        self.vocab_size = task.spaces["model"].size

    @staticmethod
    def event_grid(
        history: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        event_mask = history["sequence"]["valid"].bool()
        values = history["value_ids"]
        if not values.numel():
            zero = values.new_zeros((event_mask.shape[0], max(event_mask.shape[1], 1), 1))
            return zero, zero.clone(), zero.clone(), zero.bool(), pad_events(event_mask)
        indices = history["sequence"]["event_indices"].masked_fill(~event_mask, 0)
        token_mask = history["valid"][indices].bool() & event_mask.unsqueeze(-1)
        arrays = tuple(
            history[name][indices].masked_fill(~token_mask, 0)
            for name in ("field_ids", "value_ids", "position_ids")
        )
        return *arrays, token_mask, event_mask

    def __call__(self, batch: api.Batch) -> KeyValueTimeBatch:
        keys, values, positions, token_mask, event_mask = self.event_grid(batch.inputs["history"])
        auxiliary = batch.inputs["history_aux"]
        times = {name: pad_events(auxiliary[name]["values"]) for name in TIME_FIELDS}
        ages = times["age_seconds"].to(torch.float64)
        last = (event_mask.sum(dim=1) - 1).clamp_min(0).unsqueeze(1)
        # age_j = anchor - ts_j, so differences recover time since the first
        # retained event and time before the last retained event, respectively.
        elapsed = (ages[:, :1] - ages).masked_fill(~event_mask, 0)
        recency = (ages - ages.gather(1, last)).masked_fill(~event_mask, 0)

        profile = batch.inputs["profile"]
        profile_mask = profile["valid"].bool().clone()
        profile_time = torch.zeros_like(profile["value_ids"], dtype=torch.float32)
        age = batch.inputs["profile_ages"][FIRST_TRANSACTION_AGE]
        column = PROFILE_FIELDS.index(FIRST_TRANSACTION)
        profile_mask[:, column] &= age["valid"].bool()
        seconds = age["values"].to(torch.float64).masked_fill(~age["valid"].bool(), 0)
        profile_time[:, column] = (8 * torch.log1p(seconds / 8)).float()

        return KeyValueTimeBatch(
            key_ids=keys,
            value_ids=values,
            positions=positions,
            token_mask=token_mask,
            event_mask=event_mask,
            time_log=(8 * torch.log1p(recency / 8)).float(),
            elapsed_log=(8 * torch.log1p(elapsed / 8)).float(),
            events={
                name: times[name].long().masked_fill(~event_mask, 0)
                for name in ("hour", "dow", "dom")
            },
            prof_key_ids=profile["field_ids"].masked_fill(~profile_mask, 0),
            prof_value_ids=profile["value_ids"].masked_fill(~profile_mask, 0),
            prof_positions=profile["position_ids"].masked_fill(~profile_mask, 0),
            prof_mask=profile_mask,
            prof_time_log=profile_time.masked_fill(~profile_mask, 0),
            entity_ids=tuple(str(x) for x in batch.metadata["entity_ids"]),
            sequence_ids=tuple(str(x) for x in batch.metadata["sample_ids"]),
        )
