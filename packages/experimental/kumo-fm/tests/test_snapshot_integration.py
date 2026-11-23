# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrated multi-layer FM sampling over a reusable native index."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo import fm
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


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_snapshot_multihop_and_occurrence_alignment(tmp_path, depth, monkeypatch):
    index, schema = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(
        SimpleNamespace(index_path=index, id="fm-test"), num_threads=2)
    assert engine.schema == schema
    # No original sources or new base ingest may be needed after adoption.
    for source in tmp_path.glob("*.parquet"):
        source.unlink()
    import diskgraph

    import kumo.settings

    kumo.settings.configure(scratch_dir=tmp_path / "other")
    ingest = diskgraph.ingest
    extensions = []

    def ingest_task(*args, **kwargs):
        assert kwargs.get("parent_index_path") == str(index), "unexpected base ingest"
        extensions.append(args[1])
        assert Path(args[1]).parent == index.parent.parent / "tasks" / engine.index_build_id
        return ingest(*args, **kwargs)

    monkeypatch.setattr(diskgraph, "ingest", ingest_task)
    names = ("order_user", "payment_order", "audit_payment", "note_audit")
    future = api.Aggregate(api.Rows(
        path=(names[0],), time=api.TimeWindow("at", start=0, stop=7*86400)), "count")
    layers = tuple(api.GraphLayer((api.GraphSample(
        name, direction="reverse", fanout=2, strategy="recent"),))
        for name in names[:depth])
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"future": future},
                            neighborhoods={"history": api.NeighborhoodSpec(layers)})
    tables = ("users", "orders", "payments", "audits", "notes")[:depth+1]
    features = {name: [] for name in tables}
    context = fm.Context(pairs=[(LARGE, when(10), 100),
                                (LARGE, when(10), 101), (-5, when(10), 102)])
    query = fm.Query(pairs=[(LARGE, when(25), 200), (-5, when(25), 201)])
    with fm.prepare(engine=engine, sample=sample, target="future", task="regression",
                    features=features, batch_size=2) as loader:
        batch = loader.load(context=context, query=query)
    instance = batch.tables("context")["instance_table"].to_pylist()
    assert [(row["instance_id"], row["__fm_target"]) for row in instance] == [
        (100, 1.0), (101, 1.0), (102, 0.0)]
    assert "__fm_target" not in batch.tables("predict")["instance_table"].column_names
    related = batch.tables("context")["related_tables"]
    assert [row["order_id"] for row in related["orders"].to_pylist()
            if row["instance_id"] == 100] == [11, 12]
    assert [row["order_id"] for row in related["orders"].to_pylist()
            if row["instance_id"] == 101] == [11, 12]
    assert [row["order_id"] for row in related["orders"].to_pylist()
            if row["instance_id"] == 102] == [13]
    expected_nodes = {
        "payments": ("payment_id", [(100, 21), (100, 22), (101, 21),
                                    (101, 22), (102, 23)]),
        "audits": ("audit_id", [(100, 31), (100, 32), (101, 31),
                                (101, 32), (102, 33)]),
        "notes": ("note_id", [(100, 41), (100, 42), (101, 41),
                              (101, 42), (102, 43)]),
    }
    for name, (key, expected) in expected_nodes.items():
        if name in related:
            assert [(row["instance_id"], row[key]) for row in
                    related[name].to_pylist()] == expected
    expected_edges = {
        "order_user": [LARGE, LARGE, LARGE, LARGE, -5],
        "payment_order": [11, 12, 11, 12, 13],
        "audit_payment": [21, 22, 21, 22, 23],
        "note_audit": [31, 32, 31, 32, 33],
    }
    for step in names[:depth]:
        child = {"order_user": "orders", "payment_order": "payments",
                 "audit_payment": "audits", "note_audit": "notes"}[step]
        assert "__fm_edge_" + step in related[child].column_names
        assert related[child]["__fm_edge_" + step].null_count == 0
        assert related[child]["__fm_edge_" + step].to_pylist() == expected_edges[step]
    assert len(extensions) == 3  # Context labels, context graph, query graph.
    assert not (tmp_path / "other").exists()
    assert batch.encode("arrays") == batch.encode("arrays")
    assert batch.encode("arrow_ipc") == batch.encode("arrow_ipc")
