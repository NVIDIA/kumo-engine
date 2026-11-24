# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow as pa
import pytest

from kumo.fm.batch import PayloadBatch

BIG_ID = 9007199254740993


def workload():
    tables = {
        "users": {"primary_key": "user_id", "columns": {
            "user_id": {"dtype": "int64", "nullable": False},
            "segment": {"dtype": "string", "nullable": True}}},
        "events": {"primary_key": "event_id", "columns": {
            "event_id": {"dtype": "int64", "nullable": False},
            "user_id": {"dtype": "int64", "nullable": True}}},
    }
    bundle = SimpleNamespace(tables=tables, relations={
        "event_user": {"child": "events", "foreign_key": "user_id", "parent": "users"}})
    return SimpleNamespace(
        bundle=bundle, entity="users", table_names=set(tables), relation_names={"event_user"},
        projections={"users": ["segment"], "events": ["event_id", "user_id"]},
        target={"dtype": "int64", "classes": [0, 1]}, kind="classification",
        budgets={"max_columns": 10, "max_rows_per_table": 10,
                 "max_decoded_arrow_bytes": 100000, "max_payload_bytes": 100000})


def request(work):
    anchor = datetime(2026, 2, 1, tzinfo=timezone.utc)
    context = [{"instance_id": 7, "entity_key": BIG_ID, "anchor_time": anchor, "target": 1}]
    predict = [{"instance_id": 8, "entity_key": BIG_ID, "anchor_time": anchor}]

    def related(instance_id):
        return {
            "users": pa.table({"instance_id": pa.array([instance_id], pa.int64()),
                               "user_id": pa.array([BIG_ID], pa.int64()),
                               "segment": pa.array(["a"])}),
            "events": pa.table({"instance_id": pa.array([instance_id], pa.int64()),
                                "event_id": pa.array([17], pa.int64()),
                                "user_id": pa.array([BIG_ID], pa.int64()),
                                "__fm_edge_event_user": pa.array([BIG_ID], pa.int64())}),
        }
    return PayloadBatch(work, {"context": (context, related(7)),
                               "predict": (predict, related(8))})


def test_retained_tables_encode_both_formats_without_mutation():
    batch = request(workload())
    before = batch.tables("context")["related_tables"]["events"]
    metadata = batch.metadata
    metadata["task"]["kind"] = "changed"
    assert batch.metadata["task"]["kind"] == "classification"
    direct = batch.as_arrow_request()
    arrays = json.loads(batch.encode("arrays"))
    ipc = json.loads(batch.encode("arrow_ipc"))
    assert arrays["context"]["related_tables"]["events"]["rows"] == [[7, 17, str(BIG_ID), str(BIG_ID)]]
    for split in ("context", "predict"):
        for name in ("users", "events"):
            encoded = ipc[split]["related_tables"][name]
            decoded = pa.ipc.open_stream(base64.b64decode(encoded["payload_base64"])).read_all()
            assert decoded.equals(direct[split]["related_tables"][name])
    assert batch.tables("context")["related_tables"]["events"].equals(before)
    assert "__fm_target" not in batch.tables("predict")["instance_table"].column_names
    assert batch.logical_arrow_bytes == sum(
        frame.nbytes for split in batch.splits for frame in (
            batch.tables(split)["instance_table"],
            *batch.tables(split)["related_tables"].values()))
    with pytest.raises(ValueError, match="Unknown encoding"):
        batch.encode("invalid")
    assert batch.encode("arrays")


def test_logical_and_wire_budgets_are_independent():
    work = workload()
    work.budgets["max_decoded_arrow_bytes"] = 1
    with pytest.raises(ValueError, match="Decoded Arrow budget"):
        request(work)
    work = workload()
    work.budgets["max_payload_bytes"] = 1
    batch = request(work)
    assert batch.tables("context")["instance_table"].num_rows == 1
    with pytest.raises(ValueError, match="byte budget"):
        batch.encode("arrays")
    with pytest.raises(ValueError, match="byte budget"):
        batch.encode("arrow_ipc")


def test_native_encoder_scalar_types_and_unicode():
    from kumo.fm.native_json import encode_rows

    table = pa.table({
        "int": pa.array([BIG_ID, -BIG_ID, None], pa.int64()),
        "float": [1.0, -0.0, None],
        "text": ["snow: \u2603", "music: \U0001d11e\n", None],
        "bool": [True, False, None],
        "date": pa.array([0, -1, None], pa.date32()),
        "timestamp": pa.array([0, -1, None], pa.timestamp("us", tz="UTC")),
    })
    expected = [[str(BIG_ID), 1.0, "snow: \u2603", True,
                 "1970-01-01", "1970-01-01T00:00:00.000000Z"],
                [str(-BIG_ID), -0.0, "music: \U0001d11e\n", False,
                 "1969-12-31", "1969-12-31T23:59:59.999999Z"],
                [None] * 6]
    wire = encode_rows(table, 10000).data
    assert wire == json.dumps(expected, separators=(",", ":"), ensure_ascii=True).encode()
    assert encode_rows(table.slice(1), 10000).data == json.dumps(
        expected[1:], separators=(",", ":"), ensure_ascii=True).encode()
