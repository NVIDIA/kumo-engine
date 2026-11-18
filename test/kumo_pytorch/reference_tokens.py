# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-to-token oracle shared by AML task checks, without engine encoders.

Fitting and ID-space allocation are covered by package tests. Given their saved
state, independently apply the prototype's vocabulary and numeric-bin rules.
"""

import json
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby

FIELDS = (
    "direction",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
)


def history_tokens(anchor, events, state):
    """Reconstruct a selected AML flat history from source values, before shifting."""
    cutoff = anchor["as_of_time"] // 10**9
    sequence = []
    for index, event in enumerate(events):
        if index:
            sequence.append(state["spaces"]["history"]["roles"]["separator"])
        direction = (
            "self"
            if event["sender"] == event["receiver"]
            else "outgoing"
            if event["sender"] == anchor["entity_id"]
            else "incoming"
        )
        stamp = datetime.fromtimestamp(event["time"], timezone.utc)
        values = {
            **event["values"],
            "direction": direction,
            "age_range": cutoff - event["time"],
            "hour": stamp.hour,
            "weekday": stamp.weekday(),
            "day": stamp.day,
        }
        for field in ("age_range", *FIELDS, "hour", "weekday", "day"):
            assert values[field] is not None, (
                "Prototype flat-token fields reject missing values"
            )
            sequence.append(encoded_value(state, field, values[field]))
    return sequence


def encoded_value(state, field, value):
    binding = json.dumps(["inputs", "history", "fields", field], separators=(",", ":"))
    encoder = state["encoders"][binding]
    if encoder["kind"] == "vocabulary":
        local = (
            encoder["roles"]["missing"]
            if value is None
            else dict(encoder["entries"]).get(value, encoder["roles"]["unknown"])
        )
    else:
        assert encoder["kind"] == "buckets" and not encoder["separate_zero"]
        local = (
            0
            if value is None
            else bisect_right(encoder["boundaries"], float(value)) + 1
        )
    return dict(state["spaces"]["history"]["members"][binding])[local]


def verify_combined_values(batch, anchors, reference, state, identities):
    """Source values and chronological event groups; equal-time order may vary."""
    import torch

    from .involvement import history

    stream = batch.inputs["history"]
    assert set(stream) == {
        "value_ids",
        "position_ids",
        "source_event_indices",
        "valid",
    }, "Unexpected history tensor layout"
    assert stream["valid"].dtype == torch.bool and stream["valid"].ndim == 2
    size = len(batch.metadata["sample_ids"])
    assert stream["valid"].shape[0] == size, "History sample dimension differs"
    rows = [anchors[sample] for sample in batch.metadata["sample_ids"]]
    histories = [history(row, reference) for row in rows]
    source_count = sum(map(len, histories))
    assert (stream["source_event_indices"][stream["valid"]] < source_count).all(), (
        "History source indices exceed the batch's source rows"
    )
    for field in ("value_ids", "position_ids", "source_event_indices"):
        assert stream[field].dtype == torch.int64
        assert stream[field].shape == stream["valid"].shape, "Unaligned history tensors"
    source_offset = 0
    for i, (anchor, events) in enumerate(zip(rows, histories, strict=True)):
        sample = anchor["sample_id"]
        valid = stream["valid"][i]
        source_indices = stream["source_event_indices"][i][valid]
        if events:
            assert source_indices[0].item() == source_offset, (
                "Native history source indices do not start at this sample's rows"
            )
        source_offset += len(events)
        actual = stream["value_ids"][i][valid].tolist()
        # Ten scalar fields per event, one separator between events. Compare
        # whole records within each timestamp, preserving duplicates and values.
        assert len(actual) == max(0, 11 * len(events) - 1), (
            "History token count differs"
        )
        assert actual[10::11] == [
            state["spaces"]["history"]["roles"]["separator"]
        ] * max(0, len(events) - 1), "History separators differ"
        offset = 0
        for stamp, grouped in groupby(events, key=lambda event: event["time"]):
            group = list(grouped)
            expected = Counter(
                tuple(history_tokens(anchor, [event], state)) for event in group
            )
            emitted = Counter(
                tuple(actual[slice(11 * j, 11 * j + 10)])
                for j in range(offset, offset + len(group))
            )
            assert emitted == expected, (
                f"{sample}: history values, multiplicity or chronological order differ at {stamp}"
            )
            offset += len(group)
        # Positions index pieces within a field; each scalar contributes one.
        assert not stream["position_ids"][i][valid].any(), (
            "Scalar piece positions differ"
        )
    for field in batch.inputs["aggregates"].values():
        assert field["values"].dtype == torch.float32
        assert field["valid"].dtype == torch.bool
        assert field["values"].shape == field["valid"].shape == (size,)
    for node in batch.inputs["graph"]["nodes"].values():
        assert node["ids"].dtype == node["batch"].dtype == torch.int64
        assert node["ids"].shape == node["batch"].shape and node["ids"].ndim == 1
        assert ((node["batch"] >= 0) & (node["batch"] < size)).all()
    accounts = batch.inputs["graph"]["nodes"]["accounts"]
    actual_accounts = list(zip(accounts["batch"].tolist(), accounts["ids"].tolist()))
    expected_accounts = {
        (owner, identities[name])
        for owner, (row, events) in enumerate(zip(rows, histories, strict=True))
        for name in {row["entity_id"]}
        | {name for event in events for name in event["accounts"]}
    }
    assert (
        len(actual_accounts) == len(expected_accounts)
        and set(actual_accounts) == expected_accounts
    ), "Graph account membership differs from roots and historical endpoints"
    events = {event["id"]: event for event in reference["events"]}
    nodes = batch.inputs["graph"]["nodes"].get("transactions")
    if nodes is None:
        return
    for field, encoded in nodes["features"].items():
        key = json.dumps(
            ["inputs", "graph", "graph", "transactions", field], separators=(",", ":")
        )
        encoder = state["encoders"][key]
        assert encoded["values"].shape == encoded["valid"].shape == nodes["ids"].shape
        assert encoded["valid"].dtype == torch.bool
        assert encoded["values"].dtype == (
            torch.float32 if encoder["kind"] == "zscore" else torch.int64
        )
        for i, identity in enumerate(nodes["ids"].tolist()):
            event = events[identity]
            value = (
                event["time"]
                if field == "event_time_seconds"
                else event["values"][field]
            )
            assert bool(encoded["valid"][i]) == (value is not None), (
                "Graph scalar validity differs"
            )
            if value is None:
                continue
            if encoder["kind"] == "zscore":
                expected = (float(value) - encoder["mean"]) / encoder["scale"]
                # G-ENCODING-001 uses binary64 arithmetic; this contract then
                # stores float32. Compare the rounded result without a tolerance.
                rounded = torch.tensor(expected, dtype=torch.float32).item()
                assert encoded["values"][i].item() == rounded, (
                    "Graph numerical encoding differs from source"
                )
            else:
                expected = (
                    dict(encoder["entries"]).get(value, encoder["roles"]["unknown"])
                    if encoder["kind"] == "vocabulary"
                    else value
                )
                assert encoded["values"][i].item() == expected, (
                    "Graph categorical/raw encoding differs from source"
                )
