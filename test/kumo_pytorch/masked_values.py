# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML PRAGMA reference checks, independent of masking and model loss helpers.

Use source rows and the saved fitted vocabulary, not a second masked batch, to
recover original IDs. Generic encoder fitting and random draws have package tests.
"""

import math
from datetime import datetime, timezone

from .involvement import history
from .reference_tokens import FIELDS, encoded_value


def original_tokens(
    batch, anchors, reference, state, identities, *, event_families=None
):
    """Check source history/field alignment and return original packed value IDs."""
    assert not batch.metadata["is_padding"].any(), (
        "Source comparison needs unpadded requests"
    )
    assert set(batch.inputs) == {"history", "history_calendar", "history_timing"}
    assert set(batch.inputs["history_calendar"]) == {"hour", "weekday", "day", "valid"}
    assert set(batch.inputs["history_timing"]) == {"age_seconds", "valid"}
    space = state["spaces"]["history"]
    assert space["roles"] == {
        "pad": 0,
        "unknown": 1,
        "mask": 2,
        "entity": 3,
        "event": 4,
    }
    members = {key: dict(value) for key, value in space["members"].items()}
    field_ids = {
        tuple(path): members['["fields","fields"]'][index]
        for path, index in state["field_names"]["fields"]
    }
    assert all(path[0] == "events" and len(path) == 3 for path in field_ids)
    families = {path[1] for path in field_ids}
    for family in families:
        assert {path[-1] for path in field_ids if path[1] == family} == set(FIELDS), (
            "Unexpected maskable field"
        )
    if event_families is None:
        assert len(families) == 1, (
            "Multiple event sources require explicit source provenance"
        )
    values, keys, owners, positions, offsets = [], [], [], [], [0]
    for owner, sid in enumerate(batch.metadata["sample_ids"]):
        anchor = anchors[sid]
        cutoff = anchor["as_of_time"] // 10**9
        assert int(batch.metadata["as_of_time"][owner]) == cutoff
        assert (
            int(batch.metadata["entity_ids"][owner]) == identities[anchor["entity_id"]]
        )
        events = history(anchor, reference)
        for position, event in enumerate(events):
            root = anchor["entity_id"]
            direction = (
                "self"
                if event["sender"] == event["receiver"]
                else "outgoing"
                if event["sender"] == root
                else "incoming"
            )
            raw = {**event["values"], "direction": direction}
            family = (
                next(iter(families))
                if event_families is None
                else event_families[event["id"]]
            )
            owners.append(owner)
            positions.append(position)
            for field in FIELDS:
                value = raw[field]
                if value is None:  # The declared PRAGMA layout omits missing fields.
                    continue
                values.append(encoded_value(state, field, value))
                keys.append(field_ids[("events", family, field)])
            offsets.append(len(values))
            stamp = datetime.fromtimestamp(event["time"], timezone.utc)
            for name, expected in (
                ("hour", stamp.hour),
                ("weekday", stamp.weekday()),
                ("day", stamp.day),
            ):
                actual = batch.inputs["history_calendar"][name]
                assert (
                    actual["valid"][owner, position]
                    and actual["values"][owner, position].item() == expected
                )
            age = batch.inputs["history_timing"]["age_seconds"]
            assert (
                age["valid"][owner, position]
                and age["values"][owner, position].item() == cutoff - event["time"]
            )
        valid = batch.inputs["history_calendar"]["valid"][owner].tolist()
        assert valid == [True] * len(events) + [False] * (len(valid) - len(events))
    packed = batch.inputs["history"]
    assert packed["field_ids"].tolist() == keys, "Source fields/order differ"
    assert packed["field_offsets"].tolist() == offsets, "Source event lengths differ"
    assert (
        packed["owner"].tolist() == owners
        and packed["history_position"].tolist() == positions
    )
    assert packed["valid"].all() and packed["field_valid"].all()
    assert packed["position_ids"].tolist() == [0] * len(values)
    return values


def verify_targets(batch, original):
    """Check corruption and supervision separately under the prototype policy."""
    assert set(batch.targets) == {"masked_values"}
    target = batch.targets["masked_values"]
    packed = batch.inputs["history"]
    actual = packed["value_ids"].tolist()
    assert len(actual) == len(original)
    assert all(
        value == before or value in (1, 2)
        for value, before in zip(actual, original, strict=True)
    )
    # Original UNK is an eligible value. Unknown *replacement* is unsupervised;
    # it need not change the bytes when the original value was already UNK.
    supervised = [value == 2 for value in actual]
    assert all(value not in (0, 2, 3, 4) for value in original)
    assert target["valid"].tolist() == supervised, (
        "Supervision mask differs from masked inputs"
    )
    assert target["values"].tolist() == [
        before if keep else -100
        for before, keep in zip(original, supervised, strict=True)
    ], "Original target values differ"
    assert target["corruption"].tolist() == [
        value != before for value, before in zip(actual, original, strict=True)
    ], "Corruption mask differs"
    if "offsets" in target:
        assert target["offsets"].tolist() == packed["field_offsets"].tolist()
    owners = token_owners(batch)
    for owner in set(owners):
        assert any(
            keep and member == owner
            for keep, member in zip(supervised, owners, strict=True)
        ), "No supervised value in nonempty sample"


def token_owners(batch):
    history = batch.inputs["history"]
    offsets, owners = history["field_offsets"].tolist(), history["owner"].tolist()
    assert len(offsets) == len(owners) + 1 and offsets[0] == 0
    assert offsets[-1] == history["value_ids"].numel()
    return [
        owner
        for owner, start, stop in zip(owners, offsets[:-1], offsets[1:], strict=True)
        for _ in range(start, stop)
    ]


def selected_labels(batch):
    target = batch.targets["masked_values"]
    padding = batch.metadata["is_padding"].tolist()
    indices, labels, invalid, padded = [], [], 0, 0
    for index, (owner, valid, label) in enumerate(
        zip(
            token_owners(batch),
            target["valid"].tolist(),
            target["values"].tolist(),
            strict=True,
        )
    ):
        if padding[owner]:
            padded += 1
        elif not valid or label == -100:
            invalid += 1
        else:
            indices.append(index)
            labels.append({1: 2, 2: 1}.get(label, label))  # PRAGMA swaps UNK/MASK.
    return indices, labels, invalid, padded


def verify_loss(batch, result):
    """Independent scalar log-sum-exp CE with uniform 0.1 label smoothing."""
    indices, labels, invalid, padding = selected_labels(batch)
    logits = result.logits.detach().double().tolist()
    assert len(logits) == len(indices) == result.weight, "Wrong eligible-position count"
    terms = []
    for row, label in zip(logits, labels, strict=True):
        assert 0 <= label < len(row) and all(math.isfinite(value) for value in row)
        peak = max(row)
        normalizer = peak + math.log(math.fsum(math.exp(value - peak) for value in row))
        terms.append(normalizer - 0.9 * row[label] - 0.1 * math.fsum(row) / len(row))
    total = math.fsum(terms)
    mean = total / len(terms) if terms else None
    assert math.isclose(result.loss.item(), mean or 0.0, rel_tol=2e-6, abs_tol=2e-7), (
        "Wrong smoothed masked-value loss"
    )
    return {
        "loss_sum": total,
        "eligible": len(terms),
        "invalid": invalid,
        "padding": padding,
        "mean_loss": mean,
    }
