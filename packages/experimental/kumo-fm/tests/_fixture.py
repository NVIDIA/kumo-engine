# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrated multi-layer FM sampling over a reusable native index."""
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch import api

UTC = timezone.utc
LARGE = 9007199254740993


def when(day):
    return datetime(2025, 1, day, tzinfo=UTC)


def build_index(tmp_path):
    rows = {
        "users": [{"user_id": LARGE}, {"user_id": -5}],
        "orders": [
            {"order_id": 11, "user_id": LARGE, "at": when(1)},
            {"order_id": 12, "user_id": LARGE, "at": when(2)},
            {"order_id": 13, "user_id": -5, "at": when(2)},
            {"order_id": 14, "user_id": LARGE, "at": when(15)},
        ],
        "payments": [
            {"payment_id": 21, "order_id": 11, "at": when(3)},
            {"payment_id": 22, "order_id": 12, "at": when(3)},
            {"payment_id": 23, "order_id": 13, "at": when(3)},
        ],
        "audits": [
            {"audit_id": 31, "payment_id": 21, "at": when(4)},
            {"audit_id": 32, "payment_id": 22, "at": when(4)},
            {"audit_id": 33, "payment_id": 23, "at": when(4)},
        ],
        "notes": [
            {"note_id": 41, "audit_id": 31, "at": when(5)},
            {"note_id": 42, "audit_id": 32, "at": when(5)},
            {"note_id": 43, "audit_id": 33, "at": when(5)},
        ],
    }
    keys = {"users": "user_id", "orders": "order_id", "payments": "payment_id",
            "audits": "audit_id", "notes": "note_id"}
    sources, tables = {}, []
    for name, values in rows.items():
        path = tmp_path / f"{name}.parquet"
        table = pa.Table.from_pylist(values)
        pq.write_table(table, path)
        sources[name] = api.Parquet(str(path))
        tables.append(api.TableSpec(name, keys[name],
                                    {field.name: api.DType.from_arrow(field.type) for field in table.schema},
                                    time_column="at" if name != "users" else api.NON_TEMPORAL))
    relations = (
        api.RelationSpec("order_user", "orders", "user_id", "users", "user_id"),
        api.RelationSpec("payment_order", "payments", "order_id", "orders", "order_id"),
        api.RelationSpec("audit_payment", "audits", "payment_id", "payments", "payment_id"),
        api.RelationSpec("note_audit", "notes", "audit_id", "audits", "audit_id"),
    )
    schema = api.DataSchema(tuple(tables), relations)
    first_layer = api.GraphLayer((api.GraphSample(
        "order_user", direction="reverse", fanout=2, strategy="recent"),))
    graph = api.SampleSpec(entity=api.EntitySpec("users"),
                           neighborhoods={"history": api.NeighborhoodSpec((first_layer,))})
    contract = api.DataLoaderSpec(schema, graph, api.BatchSpec(inputs={
        "graph": api.Input(api.NeighborhoodSource("history"), api.Graph(node_fields={
            name: api.Fields(tuple(api.Encode(column, api.Raw()) for column in keys_and_types))
            for name, keys_and_types in ((table.name, table.columns) for table in tables
                                         if table.name in ("users", "orders"))
        }), None)}, batching=api.Batching(max_samples=2)))
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=contract, dataset=api.Dataset(sources),
                     engine=engine):
        pass
    return engine.index, schema
