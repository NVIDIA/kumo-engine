# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent original-row answers for declared multi-table event histories."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import groupby

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .bindings import dataset_definition, profile
from .feature_checks import FeatureChecks
from .population_anchors import projected_rows
from .reference_tokens import encoded_value


def source_events(binding, mapping, root_ids):
    """Join original rows independently; never read the prepared event tables."""
    options = dataset_definition(binding)["source_options"]
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
        [key for key, root in accounts.items() if root in root_ids], pa.int64()
    )
    selected_roots = pa.array(sorted(root_ids), pa.int64())
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
                            accounts[row[c]] if row[c] is not None else None
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
                        roots = {root}
                    else:
                        roots = {row[owner["column"]]}
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


def verify_sequence(batch, anchors, events, fields, state, *, targets):
    import torch

    stream = batch.inputs["history"]
    target = batch.targets["next_token"]
    assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}
    assert set(stream) == {"value_ids", "valid", "position_ids", "source_event_indices"}
    assert stream["valid"].dtype == target["valid"].dtype == torch.bool
    assert stream["valid"].ndim == 2
    for name in ("value_ids", "position_ids", "source_event_indices"):
        assert (
            stream[name].dtype == torch.int64
            and stream[name].shape == stream["valid"].shape
        )
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
        size = max(sum(map(len, expected_records)) + len(history) - 2, 0)
        assert valid.tolist() == [True] * size + [False] * (len(valid) - size), (
            "Token coverage/padding differs"
        )
        assert torch.equal(valid, target["valid"][i]), "Next-token eligibility differs"
        assert (stream["value_ids"][i][~valid] == roles["pad"]).all()
        assert (target["values"][i][~valid] == -100).all()
        assert not stream["position_ids"][i][valid].any(), (
            "Scalar field positions differ"
        )
        actual = stream["value_ids"][i][valid].tolist()
        labels = target["values"][i][valid].tolist()
        if targets:
            assert actual[1:] == labels[:-1], "Next-token labels shifted incorrectly"
        # Recover the original full stream including the terminal target. Whole
        # event records may permute only within one timestamp group.
        complete = actual + labels[-1:]
        assert len(complete) == (size + 1 if history else 0)
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
        assert indices == expected_indices[:-1], (
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
