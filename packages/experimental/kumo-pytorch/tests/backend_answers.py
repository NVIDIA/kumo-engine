# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture calculations for delivered native batches; no engine supplies answers."""

import bisect
import json
from collections import Counter

import numpy as np
import torch
from tokenizers import Tokenizer

from kumo.pytorch import api
from kumo.pytorch.data.records import cutoff_to_epoch


def columns(actual, expected, fields, *, padded=False, packed=False):
    """Expected scalar columns, row validity and layout boundaries."""
    rows = expected
    if padded:
        width = max(map(len, rows), default=0)
        valid = [[True] * len(row) + [False] * (width - len(row)) for row in rows]
        assert actual["valid"].tolist() == valid
        rows = [row + [{}] * (width - len(row)) for row in rows]
    elif packed:
        lengths = list(map(len, rows))
        boundaries = np.cumsum([0, *lengths]).tolist()
        assert actual["offsets"].tolist() == actual["cu_seqlens"].tolist() == boundaries
        assert actual["max_seqlen"] == max(lengths, default=0)
        assert actual["valid"].tolist() == [True] * sum(lengths)
        rows = [value for row in rows for value in row]
    for name, dtype, encoding in fields:

        def encoded(row):
            value = row.get(name)
            if value is None:
                return 0
            return encoding.get(value, 1) if encoding is not None else value

        values = (
            [[encoded(row) for row in group] for group in rows]
            if padded
            else [encoded(row) for row in rows]
        )
        valid = (
            [[row.get(name) is not None for row in group] for group in rows]
            if padded
            else [row.get(name) is not None for row in rows]
        )
        assert actual[name]["values"].dtype == dtype
        assert actual[name]["valid"].dtype == torch.bool
        assert actual[name]["valid"].tolist() == valid
        torch.testing.assert_close(actual[name]["values"], torch.tensor(values, dtype=dtype))


def temporal_batch(batch, expected, contract):
    chosen = [expected[sid] for sid in batch.metadata["sample_ids"]]
    for name in ("entity", "transactions", "timeline"):
        fields = []
        for use in contract.batch.inputs[name].representation.fields:
            vocabulary = (
                dict(use.encoding.values) if isinstance(use.encoding, api.Vocabulary) else None
            )
            dtype = (
                torch.int64
                if vocabulary is not None or use.field == "transaction_count"
                else torch.float64
            )
            fields.append((use.field, dtype, vocabulary))
        rows = [item["entity"] if name == "entity" else item[name] for item in chosen]
        columns(
            batch.inputs[name],
            rows,
            fields,
            padded=name == "transactions",
            packed=name == "timeline",
        )
    columns(
        {"future_spend": batch.targets["future_spend"]},
        [{"future_spend": item["future_spend"]} for item in chosen],
        [("future_spend", torch.float64, None)],
    )
    columns(
        batch.targets["next_transaction"],
        [item["next_transaction"] for item in chosen],
        [("amount", torch.float64, None), ("channel", torch.int64, {"online": 2, "store": 3})],
    )


def pragma_batches(batches, rows, requests, state):
    """Apply the saved scalar transforms independently, then check packing and masks."""
    encoders = state["encoders"]

    def identity(*parts):
        return json.dumps(parts, separators=(",", ":"))

    qualified = [("entity", name) for name in ("country", "risk_segment", "transaction_count")]
    qualified += [("events", "transactions", name) for name in ("amount", "channel", "description")]
    expected_fields = [[list(name), index + 1] for index, name in enumerate(qualified)]
    assert state["field_names"] == {"fields": expected_fields}
    roles = dict(zip(("pad", "unknown", "missing", "mask", "entity", "event"), range(6)))
    members = {identity("fields", "fields"): {"vocabulary_size": 7, "roles": {"pad": 0}}}
    members.update({key: value for key, value in encoders.items() if "vocabulary_size" in value})
    allocations, next_id = {}, 6
    for key, encoder in sorted(members.items(), key=lambda pair: json.loads(pair[0])):
        local_roles = {number: role for role, number in encoder["roles"].items()}
        mapping = {}
        for local in range(encoder["vocabulary_size"]):
            if local in local_roles:
                mapping[local] = roles[local_roles[local]]
            else:
                mapping[local] = next_id
                next_id += 1
        allocations[key] = mapping
    assert state["spaces"] == {
        "pragma": {
            "roles": roles,
            "size": next_id,
            "members": {
                key: [[local, final] for local, final in mapping.items()]
                for key, mapping in allocations.items()
            },
        }
    }
    field_ids = {
        tuple(key): allocations[identity("fields", "fields")][local]
        for key, local in expected_fields
    }
    by_id = {r.sample_id: r for r in requests}
    users = {row["user_id"]: row for row in rows["users"]}
    cutoff = max(cutoff_to_epoch(r.as_of_time) for r in requests)
    categories = [
        row[name]
        for row in rows["users"]
        if cutoff_to_epoch(row["signup_at"]) <= cutoff
        for name in ("country", "risk_segment")
    ]
    categories += [
        row["channel"]
        for row in rows["card_transactions"]
        if cutoff_to_epoch(row["occurred_at"]) <= cutoff
    ]
    counts = Counter(value for value in categories if value is not None)
    assert encoders[identity("shared", "categories")]["entries"] == [
        [value, i + 2]
        for i, value in enumerate(sorted(counts, key=lambda value: (-counts[value], value)))
    ]

    def events(request):
        return sorted(
            [
                row
                for row in rows["card_transactions"]
                if row["user_id"] == request.entity_id
                and cutoff_to_epoch(row["occurred_at"]) <= cutoff_to_epoch(request.as_of_time)
            ],
            key=lambda row: (row["occurred_at"], row["card_transaction_id"]),
        )[-16:]

    # Computed count buckets fit only anchor observations, with exact linear quantiles.
    population = [len(events(request)) for request in requests]
    cuts = np.unique(np.quantile(population, [1 / 3, 2 / 3])).tolist()
    assert (
        encoders[identity("inputs", "entity", "fields", "transaction_count")]["boundaries"] == cuts
    )

    tokenizers = {
        key: Tokenizer.from_str(value["tokenizer_json"])
        for key, value in encoders.items()
        if value["kind"] == "text"
    }

    def tokens(input_name, raw):
        names = (
            ("country", "risk_segment", "transaction_count")
            if input_name == "entity"
            else ("amount", "channel", "description")
        )
        output, fields = [], []
        for name in names:
            key = (
                identity("shared", "categories")
                if name in ("country", "risk_segment", "channel")
                else identity("inputs", input_name, "fields", name)
            )
            encoder, value = encoders[key], raw[name]
            if value is None:
                local = [encoder["roles"]["missing"]]
            elif encoder["kind"] == "vocabulary":
                local = [dict(encoder["entries"]).get(value, encoder["roles"]["unknown"])]
            elif encoder["kind"] == "buckets":
                local = [
                    1
                    if value == 0 and encoder["separate_zero"]
                    else 1
                    + int(encoder["separate_zero"])
                    + bisect.bisect_right(encoder["boundaries"], value)
                ]
            else:
                local = tokenizers[key].encode(value).ids or [encoder["roles"]["unknown"]]
            output.extend(allocations[key][code] for code in local)
            qualified = (
                ("entity", name) if input_name == "entity" else ("events", "transactions", name)
            )
            fields.extend([field_ids[qualified]] * len(local))
        return output[:8], fields[:8]

    for batch in batches:
        chosen = [by_id[sid] for sid in batch.metadata["sample_ids"]]
        histories = [events(request) for request in chosen]
        entity_values, entity_fields = [], []
        for request, history in zip(chosen, histories, strict=True):
            values, fields = tokens(
                "entity", {**users[request.entity_id], "transaction_count": len(history)}
            )
            entity_values.append(values)
            entity_fields.append(fields)
        entity = batch.inputs["entity"]
        assert entity["value_ids"].dtype == entity["field_ids"].dtype == torch.int64
        assert entity["value_ids"].tolist() == entity_values
        assert entity["field_ids"].tolist() == entity_fields
        assert entity["valid"].all() and entity["field_valid"].all()
        calendar = [
            [
                {
                    "hour": row["occurred_at"].hour,
                    "weekday": row["occurred_at"].weekday(),
                    "day": row["occurred_at"].day,
                }
                for row in history
            ]
            for history in histories
        ]
        timing = [
            [
                {
                    "age_seconds": float(
                        cutoff_to_epoch(request.as_of_time) - cutoff_to_epoch(row["occurred_at"])
                    )
                }
                for row in history
            ]
            for request, history in zip(chosen, histories, strict=True)
        ]
        columns(
            batch.inputs["history_calendar"],
            calendar,
            [(name, torch.int64, None) for name in ("hour", "weekday", "day")],
            padded=True,
        )
        columns(
            batch.inputs["history_timing"],
            timing,
            [("age_seconds", torch.float64, None)],
            padded=True,
        )
        values, fields, offsets, owners, positions = [], [], [0], [], []
        for owner, history in enumerate(histories):
            for position, row in enumerate(history):
                value, field = tokens("history", row)
                values.extend(value)
                fields.extend(field)
                offsets.append(len(values))
                owners.append(owner)
                positions.append(position)
        actual, target = batch.inputs["history"], batch.targets["masked_values"]
        assert (
            actual["value_ids"].dtype
            == actual["field_ids"].dtype
            == target["values"].dtype
            == torch.int64
        )
        assert actual["field_offsets"].tolist() == offsets
        assert actual["owner"].tolist() == owners
        assert actual["history_position"].tolist() == positions
        assert actual["valid"].tolist() == [True] * len(values)
        expected = torch.tensor(values, dtype=torch.int64)
        selected, corrupt = target["valid"], target["corruption"]
        for owner, history in enumerate(histories):
            if history:
                slots = [i for i, event_owner in enumerate(owners) if event_owner == owner]
                assert any(bool(selected[offsets[i] : offsets[i + 1]].any()) for i in slots), (
                    "nonempty sample has no supervised mask"
                )
        assert selected.dtype == corrupt.dtype == torch.bool
        assert torch.equal(selected, corrupt & (actual["value_ids"] == roles["mask"]))
        assert target["values"][selected].tolist() == expected[selected].tolist()
        assert target["values"][~selected].tolist() == [-100] * int((~selected).sum())
        assert actual["value_ids"][~corrupt].tolist() == expected[~corrupt].tolist()
        assert set(actual["value_ids"][corrupt].tolist()) <= {roles["mask"], roles["unknown"]}
        assert target["offsets"].tolist() == offsets
        field_valid = actual["field_valid"]
        assert (
            actual["field_ids"][field_valid].tolist() == torch.tensor(fields)[field_valid].tolist()
        )
        assert set(actual["field_ids"][~field_valid].tolist()) <= {roles["pad"]}


def banking_graph(rows, requests):
    """The fixture's two recent-neighbor hops: six accounts, four transactions each."""
    nodes = {"users": set(), "accounts": set(), "card_transactions": set()}
    edges = {"user_accounts": set(), "account_card_transactions": set()}
    for owner, request in enumerate(requests):
        cutoff = cutoff_to_epoch(request.as_of_time)
        nodes["users"].add((request.entity_id, owner))
        accounts = [
            row
            for row in rows["accounts"]
            if row["user_id"] == request.entity_id and cutoff_to_epoch(row["opened_at"]) <= cutoff
        ]
        accounts.sort(key=lambda row: row["opened_at"], reverse=True)
        assert len(accounts) <= 6 or accounts[5]["opened_at"] > accounts[6]["opened_at"], (
            "fixture needs a tie-aware account boundary"
        )
        for account in accounts[:6]:
            account_id = account["account_id"]
            nodes["accounts"].add((account_id, owner))
            edges["user_accounts"].add((request.entity_id, account_id))
            events = [
                row
                for row in rows["card_transactions"]
                if row["account_id"] == account_id and cutoff_to_epoch(row["occurred_at"]) <= cutoff
            ]
            events.sort(key=lambda row: row["occurred_at"], reverse=True)
            assert len(events) <= 4 or events[3]["occurred_at"] > events[4]["occurred_at"], (
                "fixture needs a tie-aware transaction boundary"
            )
            for event in events[:4]:
                event_id = event["card_transaction_id"]
                nodes["card_transactions"].add((event_id, owner))
                edges["account_card_transactions"].add((account_id, event_id))
    return {
        "nodes": {name: sorted(values) for name, values in nodes.items()},
        "edges": {name: sorted(values) for name, values in edges.items()},
    }
