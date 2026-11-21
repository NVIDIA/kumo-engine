# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare flattened model histories with timestamped scalar fields from source rows."""

from bisect import bisect_right
from datetime import UTC, datetime

import torch

from kumo.pytorch import api
from tests.test_inference_lifecycle import assert_same


def check_scalar_history(task, batch, fields, histories):
    fields = {"age_range": api.Buckets(bins=64), **fields}
    for name, values in (("hour", range(24)), ("weekday", range(7)), ("day", range(1, 32))):
        fields[name] = api.Vocabulary(values=tuple((i, i + 2) for i in values))
    bindings = task.output("history").bindings
    assert {name: binding.encoder.config for name, binding in bindings.items()} == fields
    space = task.spaces["history"]

    def token(name, value):
        encoder = bindings[name].encoder
        assert value is not None
        if isinstance(fields[name], api.Buckets):
            local = bisect_right(encoder.boundaries, float(value)) + 1
        else:
            local = dict(encoder.entries).get(value, encoder.config.unknown_id)
        return space.members[encoder.identity][local]

    stream = batch.inputs["history"]
    valid = stream["valid"]
    assert valid.dtype == torch.bool and valid.ndim == 2 and valid.shape[0] == len(histories)
    assert set(stream) == {"value_ids", "source_event_indices", "position_ids", "valid"}
    for name in ("value_ids", "source_event_indices", "position_ids"):
        assert stream[name].dtype == torch.int64 and stream[name].shape == valid.shape
    target = batch.targets.get("next_token")
    if target is not None:
        assert_same(target["valid"], valid)
        assert target["values"].dtype == torch.int64 and target["values"].shape == valid.shape
        assert (target["values"][~valid] == -100).all()

    offset = 0
    for owner, (cutoff, history) in enumerate(histories):
        expected = []
        for seconds, fields_from_source in history:
            stamp = datetime.fromtimestamp(seconds, UTC)
            values = dict(
                fields_from_source,
                age_range=cutoff - seconds,
                hour=stamp.hour,
                weekday=stamp.weekday(),
                day=stamp.day,
            )
            expected.append((seconds, tuple(token(n, values[n]) for n in fields if n in values)))
        mask = valid[owner]
        size = max(0, sum(len(record) + 1 for _, record in expected) - 1 - int(target is not None))
        assert_same(mask, torch.arange(mask.numel()) < size)
        assert not stream["position_ids"][owner][mask].any()
        tokens = stream["value_ids"][owner]
        assert (tokens[~mask] == space.special(role="pad")).all()
        ids = tokens[mask].tolist()
        if target is not None:
            labels = target["values"][owner][mask].tolist()
            assert labels[:-1] == ids[1:]
            ids.extend(labels[-1:])

        records = [[]] if ids else []
        for value in ids:
            if value == space.special(role="separator"):
                records.append([])
            else:
                records[-1].append(value)
        # Compare whole events within each timestamp, preserving duplicates and chronology.
        times = sorted(seconds for seconds, _ in expected)
        assert len(times) == len(records)
        assert sorted(zip(times, map(tuple, records), strict=True)) == sorted(expected)
        sources = []
        for index, record in enumerate(records, offset):
            if sources:
                sources.append(-1)
            sources.extend([index] * len(record))
        assert stream["source_event_indices"][owner][mask].tolist() == sources[:size]
        offset += len(history)
