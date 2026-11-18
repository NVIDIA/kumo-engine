# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent original-row answers for declared multi-table event histories."""

import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import groupby

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .bindings import dataset_definition, profile
from .feature_checks import FeatureChecks
from .population_anchors import projected_rows
from .reference_tokens import encoded_pieces, encoded_value


def source_events(binding, mapping, root_ids):
    """Join original rows independently; never read the prepared event tables."""
    options = dataset_definition(binding)["source_options"]
    owned_roots = options.get("root_level", "owner") == "owned"
    ownership = options["ownership"]
    accounts = {}
    for source in mapping["sources"][ownership["table"]]:
        for batch in pq.ParquetFile(source["path"]).iter_batches(
            columns=[ownership["key"], ownership["root"]]
        ):
            for row in batch.to_pylist():
                assert row[ownership["key"]] not in accounts, "Duplicate owned identity"
                accounts[row[ownership["key"]]] = row[ownership["root"]]
    selected_accounts = pa.array(
        [
            key
            for key, root in accounts.items()
            if (key if owned_roots else root) in root_ids
        ],
        pa.int64(),
    )
    parent_roots = {accounts[key] for key in root_ids} if owned_roots else root_ids
    selected_roots = pa.array(sorted(parent_roots), pa.int64())
    children = defaultdict(set)
    all_children = defaultdict(list)
    for key, parent in accounts.items():
        all_children[parent].append(key)
        if key in root_ids:
            children[parent].add(key)
    slots = {
        key: slot
        for keys in all_children.values()
        for slot, key in enumerate(sorted(keys))
    }
    fanout = max(map(len, all_children.values()), default=1)
    result = defaultdict(list)
    selected_families = {f[0] for f in profile(binding, "input")["families"]}
    for family, spec in options["events"].items():
        if family not in selected_families:
            continue
        owner = spec["ownership"]
        columns = [
            spec["key"],
            spec["time"],
            *spec["fields"].values(),
            *owner.get("columns", [owner.get("column")]),
        ]
        if owner.get("verify_root"):
            columns.append(owner["verify_root"])
        columns = list(dict.fromkeys(c for c in columns if c))
        seen = set()
        for source in mapping["sources"][family]:
            for batch in pq.ParquetFile(source["path"]).iter_batches(columns=columns):
                if owner["kind"] == "endpoints":
                    masks = [
                        pc.is_in(batch.column(c), value_set=selected_accounts)
                        for c in owner["columns"]
                    ]
                    selected = pc.or_(masks[0], masks[1])
                else:
                    selected = pc.is_in(
                        batch.column(owner["column"]),
                        value_set=selected_accounts
                        if owner["kind"] == "owned"
                        else selected_roots,
                    )
                for row in batch.filter(selected).to_pylist():
                    identity = row[spec["key"]]
                    assert identity not in seen, (
                        f"Duplicate physical identity: {family}/{identity}"
                    )
                    seen.add(identity)
                    directions = {}
                    if owner["kind"] == "endpoints":
                        sender, receiver = [
                            (row[c] if owned_roots else accounts[row[c]])
                            if row[c] is not None
                            else None
                            for c in owner["columns"]
                        ]
                        for root in {sender, receiver} - {None}:
                            directions[root] = (
                                "self"
                                if sender == receiver
                                else "outgoing"
                                if root == sender
                                else "incoming"
                            )
                        roots = set(directions)
                    elif owner["kind"] == "owned":
                        root = accounts[row[owner["column"]]]
                        if owner.get("verify_root"):
                            assert row[owner["verify_root"]] == root, (
                                "Original ownership mismatch"
                            )
                        roots = {row[owner["column"]] if owned_roots else root}
                    else:
                        roots = (
                            children[row[owner["column"]]]
                            if owned_roots
                            else {row[owner["column"]]}
                        )
                    for root in roots & root_ids:
                        values = {
                            alias: row[column]
                            for alias, column in spec["fields"].items()
                        }
                        values["source_kind"] = family
                        if directions:
                            values["direction"] = directions[root]
                        result[root].append(
                            {
                                "family": family,
                                "source_id": identity,
                                "mapped_id": identity * fanout + slots[root]
                                if owned_roots and owner["kind"] == "direct"
                                else identity * 2
                                + int(directions.get(root) == "incoming"),
                                "time": row[spec["time"]],
                                "values": values,
                            }
                        )
    return {root: sorted(result[root], key=lambda e: e["time"]) for root in root_ids}


def event_tokens(event, cutoff, fields, state):
    stamp = datetime.fromtimestamp(event["time"], timezone.utc)
    values = {
        **event["values"],
        "age_range": cutoff - event["time"],
        "hour": stamp.hour,
        "weekday": stamp.weekday(),
        "day": stamp.day,
    }
    # Tokens.missing applies to NULL declared values. Other event families
    # do not declare these fields and therefore contribute no placeholder.
    selected = [name for name in fields if name in values]
    assert all(values[name] is not None for name in selected), (
        "NULL declared value under Tokens(missing='error')"
    )
    return tuple(encoded_value(state, name, values[name]) for name in selected)


def verify_sequence(batch, anchors, events, fields, state, *, targets, native=True):
    import torch

    stream = batch.inputs["history"]
    target = batch.targets["next_token"] if native else None
    if native:
        assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}
    assert set(stream) == {"value_ids", "valid", "position_ids", "source_event_indices"}
    assert stream["valid"].dtype == torch.bool
    assert stream["valid"].ndim == 2
    for name in ("value_ids", "position_ids", "source_event_indices"):
        assert (
            stream[name].dtype == torch.int64
            and stream[name].shape == stream["valid"].shape
        )
    if native:
        assert target["valid"].dtype == torch.bool
        assert target["values"].dtype == torch.int64
        assert target["values"].shape == target["valid"].shape == stream["valid"].shape
    assert stream["valid"].shape[0] == len(batch.metadata["sample_ids"])
    roles = state["spaces"]["history"]["roles"]
    source_offset = 0
    for i, sample in enumerate(batch.metadata["sample_ids"]):
        anchor = anchors[sample]
        cutoff = anchor["as_of_time"]
        history = [e for e in events[anchor["entity_id"]] if e["time"] < cutoff]
        valid = stream["valid"][i]
        expected_records = [event_tokens(e, cutoff, fields, state) for e in history]
        size = max(sum(map(len, expected_records)) + len(history) - 1 - int(native), 0)
        assert valid.tolist() == [True] * size + [False] * (len(valid) - size), (
            "Token coverage/padding differs"
        )
        if native:
            assert torch.equal(valid, target["valid"][i]), (
                "Next-token eligibility differs"
            )
        assert (stream["value_ids"][i][~valid] == roles["pad"]).all()
        if native:
            assert (target["values"][i][~valid] == -100).all()
        assert not stream["position_ids"][i][valid].any(), (
            "Scalar field positions differ"
        )
        actual = stream["value_ids"][i][valid].tolist()
        labels = target["values"][i][valid].tolist() if native else []
        if targets:
            assert actual[1:] == labels[:-1], "Next-token labels shifted incorrectly"
        # Recover the original full stream including the terminal target. Whole
        # event records may permute only within one timestamp group.
        complete = actual + labels[-1:]
        assert len(complete) == (size + int(native) if history else 0)
        records = []
        if complete:
            records = [[]]
            for value in complete:
                if value == roles["separator"]:
                    records.append([])
                else:
                    records[-1].append(value)
        assert len(records) == len(history), "Event/separator multiplicity differs"
        offset = 0
        for stamp, group in groupby(history, key=lambda e: e["time"]):
            group = list(group)
            expected = Counter(event_tokens(e, cutoff, fields, state) for e in group)
            stop = offset + len(group)
            emitted = Counter(tuple(record) for record in records[offset:stop])
            assert emitted == expected, (
                f"{sample}: original history values/membership/chronology differ at {stamp}"
            )
            offset += len(group)
        indices = stream["source_event_indices"][i][valid].tolist()
        expected_indices = []
        for j, record in enumerate(records):
            if j:
                expected_indices.append(-1)
            expected_indices.extend([source_offset + j] * len(record))
        assert indices == (expected_indices[:-1] if native else expected_indices), (
            "History source ownership/layout differs"
        )
        source_offset += len(history)


def sequence_features(binding, projection, mapping, state, *, mode):
    from .population import check_delivery

    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    assert all(str(row["entity_id"]) == row["registered_entity_id"] for row in rows)
    events = source_events(binding, mapping, {row["entity_id"] for row in rows})
    names = list(
        dict.fromkeys(
            alias
            for _, _, _, fs in profile(binding, "input")["families"]
            for alias, _, _ in fs
        )
    )
    fields = ("age_range", *names, "hour", "weekday", "day")
    checks = {
        "identity_cutoff": lambda batch: check_delivery(
            batch, iter(anchors[s] for s in batch.metadata["sample_ids"])
        ),
        "history_values": lambda batch: verify_sequence(
            batch, anchors, events, fields, state, targets=False
        ),
        "next_token_targets": lambda batch: verify_sequence(
            batch, anchors, events, fields, state, targets=True
        ),
    }
    return FeatureChecks(
        binding["feature_checks"],
        mode,
        checks,
        {r["sample_id"]: r["split"] for r in rows},
    )


def combined_features(binding, projection, mapping, state, *, mode):
    from .aggregate_features import aggregate_check
    from .bindings import graph_layout
    from .graph_features import graph_check
    from .population import check_delivery
    from .targets import (
        source_target_rows,
        verify_stored_batch,
        verify_target_projection,
    )

    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    assert all(str(row["entity_id"]) == row["registered_entity_id"] for row in rows)
    originals = source_target_rows(binding, mapping["anchor_source"]["path"], rows)
    for row in rows:
        verify_target_projection(binding, row, originals[row["anchor_row_position"]])
    inputs = profile(binding, "input")
    events = source_events(binding, mapping, {row["entity_id"] for row in rows})
    prediction_anchors = {
        sample: {**row, "as_of_time": row["prediction_time_ns"] // 10**9}
        for sample, row in anchors.items()
    }
    names = list(
        dict.fromkeys(
            alias for _, _, _, fields in inputs["families"] for alias, _, _ in fields
        )
    )
    checks = {
        "identity_cutoff": lambda batch: check_delivery(
            batch,
            iter(anchors[s] for s in batch.metadata["sample_ids"]),
            graph_clocks={
                node: fields["timestamp"]
                for node, fields in graph_layout(inputs)[0].items()
            },
        ),
        "graph_values": graph_check(binding, mapping, anchors, events, state),
        "history_values": lambda batch: verify_sequence(
            batch,
            prediction_anchors,
            events,
            ("age_range", *names, "hour", "weekday", "day"),
            state,
            targets=False,
            native=False,
        ),
        "aggregate_values": aggregate_check(inputs, events, anchors, state),
        "stored_targets": lambda batch: verify_stored_batch(
            binding, batch, [anchors[s] for s in batch.metadata["sample_ids"]], state
        ),
    }
    return FeatureChecks(
        binding["feature_checks"],
        mode,
        checks,
        {row["sample_id"]: row["split"] for row in rows},
    )


def _matching(candidates):
    """Match tied records bijectively; unknown replacements may be ambiguous."""
    assigned, matched = {}, {}
    for start in sorted(range(len(candidates)), key=lambda i: len(candidates[i])):
        pending, previous, free = deque([start]), {}, None
        while pending and free is None:
            actual = pending.popleft()
            for expected in candidates[actual]:
                if expected in previous:
                    continue
                previous[expected] = actual
                if expected not in assigned:
                    free = expected
                    break
                pending.append(assigned[expected])
        assert free is not None, (
            "Masked source records cannot match eligible tied events"
        )
        while free is not None:
            actual = previous[free]
            prior = matched.get(actual)
            assigned[free] = actual
            matched[actual] = free
            free = prior
    return matched


def masked_records(binding, events, state):
    members = state["spaces"]["history"]["members"]
    names = dict(members['["fields","fields"]'])
    field_ids = {
        tuple(path): names[index] for path, index in state["field_names"]["fields"]
    }
    fields = {family: fs for family, _, _, fs in profile(binding, "input")["families"]}
    result = {}
    for root, rows in events.items():
        result[root] = []
        for event in rows:
            keys, values, positions = [], [], []
            for name, _, _ in fields[event["family"]]:
                value = event["values"][name]
                if value is None:
                    continue
                pieces = encoded_pieces(state, name, value)
                values.extend(pieces)
                keys.extend(
                    [field_ids[("events", event["family"], name)]] * len(pieces)
                )
                positions.extend(range(len(pieces)))
            result[root].append(
                {**event, "keys": keys, "tokens": values, "positions": positions}
            )
    return result


def verify_masked_history(batch, anchors, records):
    """Return original packed IDs from source records, allowing tied permutations."""
    import torch

    assert set(batch.inputs) == {"history", "history_calendar", "history_timing"}
    assert set(batch.targets) == {"masked_values"}
    packed, target = batch.inputs["history"], batch.targets["masked_values"]
    count = packed["value_ids"].numel()
    for name in ("value_ids", "field_ids", "position_ids"):
        assert packed[name].dtype == torch.int64 and packed[name].shape == (count,)
    for name in ("valid", "field_valid"):
        assert (
            packed[name].dtype == torch.bool
            and packed[name].shape == (count,)
            and packed[name].all()
        )
    assert target["values"].dtype == torch.int64 and target["values"].shape == (count,)
    for name in ("valid", "corruption"):
        assert target[name].dtype == torch.bool and target[name].shape == (count,)
    offsets = packed["field_offsets"].tolist()
    assert packed["field_offsets"].dtype == torch.int64
    assert offsets[0] == 0 and offsets[-1] == count and offsets == sorted(offsets)
    if "offsets" in target:
        assert (
            target["offsets"].dtype == torch.int64
            and target["offsets"].tolist() == offsets
        )
    tokens = {
        name: packed[name].tolist()
        for name in ("value_ids", "field_ids", "position_ids")
    }
    labels = {name: target[name].tolist() for name in ("values", "valid", "corruption")}

    def compatible(index, row):
        start, stop = offsets[index], offsets[index + 1]
        if (
            tokens["field_ids"][start:stop] != row["keys"]
            or tokens["position_ids"][start:stop] != row["positions"]
        ):
            return False
        for j, original in zip(range(start, stop), row["tokens"], strict=True):
            value = tokens["value_ids"][j]
            if not (
                value in (original, 1, 2)
                and labels["valid"][j] == (value == 2)
                and labels["values"][j] == (original if value == 2 else -100)
                and labels["corruption"][j] == (value != original)
            ):
                return False
        return True

    originals, owners, positions, event_offset = [], [], [], 0
    sequence = packed["sequence"]
    assert (
        sequence["event_indices"].dtype == torch.int64
        and sequence["valid"].dtype == torch.bool
        and sequence["valid"].ndim == 2
    )
    assert sequence["event_indices"].shape == sequence["valid"].shape
    assert sequence["valid"].shape[0] == len(batch.metadata["sample_ids"])
    derived = {
        "history_calendar": {
            "hour": torch.int64,
            "weekday": torch.int64,
            "day": torch.int64,
        },
        "history_timing": {"age_seconds": torch.float64},
    }
    for name, fields in derived.items():
        values = batch.inputs[name]
        assert set(values) == {"valid", *fields}
        for item in (values, *(values[field] for field in fields)):
            assert item["valid"].dtype == torch.bool
            assert torch.equal(item["valid"], sequence["valid"]), (
                "Derived field validity differs"
            )
        for field, dtype in fields.items():
            assert values[field]["values"].dtype == dtype, "Derived field dtype differs"
            assert values[field]["values"].shape == sequence["valid"].shape, (
                "Derived field shape differs"
            )
    for owner, sample in enumerate(batch.metadata["sample_ids"]):
        anchor = anchors[sample]
        history = [
            r for r in records[anchor["entity_id"]] if r["time"] < anchor["as_of_time"]
        ]
        mask = [True] * len(history) + [False] * (
            sequence["valid"].shape[1] - len(history)
        )
        assert sequence["valid"][owner].tolist() == mask, (
            "Event sequence coverage differs"
        )
        assert sequence["event_indices"][owner][
            sequence["valid"][owner]
        ].tolist() == list(range(event_offset, event_offset + len(history)))
        for field in ("history_calendar", "history_timing"):
            assert batch.inputs[field]["valid"].dtype == torch.bool
            assert batch.inputs[field]["valid"][owner].tolist() == mask
        position = 0
        for stamp, group in groupby(history, key=lambda row: row["time"]):
            group = list(group)
            candidates = [
                [
                    i
                    for i, row in enumerate(group)
                    if compatible(event_offset + position + j, row)
                ]
                for j in range(len(group))
            ]
            matching = _matching(candidates)
            for j in range(len(group)):
                row = group[matching[j]]
                originals.extend(row["tokens"])
                owners.append(owner)
                positions.append(position + j)
                clock = datetime.fromtimestamp(stamp, timezone.utc)
                for field, name, expected in (
                    ("history_calendar", "hour", clock.hour),
                    ("history_calendar", "weekday", clock.weekday()),
                    ("history_calendar", "day", clock.day),
                    ("history_timing", "age_seconds", anchor["as_of_time"] - stamp),
                ):
                    item = batch.inputs[field][name]
                    assert (
                        item["valid"].dtype == torch.bool
                        and item["valid"][owner, position + j]
                    )
                    assert item["values"][owner, position + j].item() == expected, (
                        f"{sample}: {name} differs"
                    )
            position += len(group)
        event_offset += len(history)
    assert len(offsets) == event_offset + 1 and len(originals) == count
    for name, expected in (("owner", owners), ("history_position", positions)):
        assert packed[name].dtype == torch.int64 and packed[name].tolist() == expected
    return originals


def masked_features(binding, projection, mapping, state, *, mode):
    from .masked_values import verify_targets
    from .population import check_delivery

    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    assert all(str(row["entity_id"]) == row["registered_entity_id"] for row in rows)
    events = source_events(binding, mapping, {row["entity_id"] for row in rows})
    records = masked_records(binding, events, state)
    original = lambda batch: verify_masked_history(batch, anchors, records)
    checks = {
        "identity_cutoff": lambda batch: check_delivery(
            batch, iter(anchors[s] for s in batch.metadata["sample_ids"])
        ),
        "history_values": original,
        "masked_targets": lambda batch: verify_targets(batch, original(batch)),
    }
    if "bpe_pieces" in binding["feature_checks"]["assertions"]:
        fields = profile(binding, "input")["encodings"]
        for name in fields:
            key = json.dumps(
                ["inputs", "history", "fields", name], separators=(",", ":")
            )
            assert state["encoders"][key]["kind"] == "text", (
                "Required BPE encoder missing"
            )
        checks["bpe_pieces"] = original
    return FeatureChecks(
        binding["feature_checks"],
        mode,
        checks,
        {r["sample_id"]: r["split"] for r in rows},
    )
