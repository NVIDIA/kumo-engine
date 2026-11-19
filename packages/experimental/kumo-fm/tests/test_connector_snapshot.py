# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel connector → snapshot → FM, including exported reuse."""
import base64
import hashlib
import json
import shutil
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo import fm
from kumo.pytorch import api

kumo = pytest.importorskip("kumo.engine")
from kumoapi.typing import Stype  # noqa: E402

from kumo.engine._scratch import get_scratch_dir  # noqa: E402


def inventory(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def at(second):
    return datetime.fromtimestamp(second, tz=timezone.utc)


def test_connector_export_reuse_and_aliased_relation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(pa.table({"id": [-5, 9007199254740993]}), source / "users.parquet")
    pq.write_table(pa.table({"id": [1, 2, 3, 4],
                             "user": [-5, -5, -5, 9007199254740993],
                             "at": pa.array([10, 20, 30, 10], pa.timestamp("s"))}),
                   source / "events.parquet")
    connector = kumo.FileConnector(root_dir=str(source))
    graph = kumo.Graph(tables={
        "users": kumo.Table.from_source_table(connector["users.parquet"], primary_key="id"),
        "events": kumo.Table.from_source_table(connector["events.parquet"],
                                               primary_key="id", time_column="at").with_column("user", stype=Stype.ID),
    }, edges=[kumo.Edge(src_table="events", fkey="user", dst_table="users")])
    previous = get_scratch_dir()
    try:
        kumo.configure(scratch_dir=tmp_path / "producer")
        snapshot = graph.snapshot()
        original_hashes = inventory(snapshot.index_path)

        def run(snapshot, work):
            engine = api.DiskGraphEngine.from_snapshot(snapshot, work_dir=work,
                                                       num_threads=2)
            physical = engine.schema.relations[0]
            alias = api.RelationSpec("events_selected", physical.left_table,
                                     physical.left_key, physical.right_table, physical.right_key)
            schema = api.DataSchema(engine.schema.tables, (alias,))
            future = api.Aggregate(api.Rows(
                path=(alias.name,), time=api.TimeWindow("at", start=0, stop=10)), "count")
            layer = api.GraphLayer((api.GraphSample(
                alias.name, direction="reverse", fanout=2, strategy="recent"),))
            sample = api.SampleSpec(
                entity=api.EntitySpec("users"), targets={"next": future},
                neighborhoods={"history": api.NeighborhoodSpec((layer,))})
            with fm.prepare(engine=engine, schema=schema, sample=sample, target="next",
                            task="regression", features={"users": [], "events": ["at"]},
                            batch_size=1) as loader:
                batch = loader.load(context=fm.Context(pairs=[(-5, at(15), 88),
                                                              (-5, at(15), 89), (9007199254740993, at(15), 90)]),
                                    query=fm.Query(pairs=[(-5, at(40), 91),
                                                          (9007199254740993, at(40), 92)]))
            context = batch.tables("context")
            assert [(row["instance_id"], row["__fm_target"])
                    for row in context["instance_table"].to_pylist()] == [
                        (88, 1.0), (89, 1.0), (90, 0.0)]
            rows = context["related_tables"]["events"].to_pylist()
            assert [(row["instance_id"], row["id"], row["__fm_edge_events_selected"])
                    for row in rows] == [(88, 1, -5), (89, 1, -5),
                                         (90, 4, 9007199254740993)]
            assert "__fm_target" not in batch.tables("predict")["instance_table"].column_names
            arrays, ipc = batch.encode("arrays"), batch.encode("arrow_ipc")
            assert arrays == batch.encode("arrays") and ipc == batch.encode("arrow_ipc")
            wire_arrays, wire_ipc = json.loads(arrays), json.loads(ipc)
            assert wire_arrays["context"]["related_tables"]["events"]["rows"][2][0] == 90
            assert str(9007199254740993) in arrays.decode()
            for split in batch.splits:
                direct = batch.tables(split)
                entries = {"instance_table": direct["instance_table"],
                           **direct["related_tables"]}
                for name, table in entries.items():
                    encoded = (wire_ipc[split]["instance_table"] if name == "instance_table"
                               else wire_ipc[split]["related_tables"][name])
                    decoded = pa.ipc.open_stream(base64.b64decode(
                        encoded["payload_base64"])).read_all()
                    assert decoded.equals(table)
            return batch.tables("predict")["related_tables"]["events"]
        first = run(snapshot, tmp_path / "work1")
        assert inventory(snapshot.index_path) == original_hashes
        exports = kumo.FileConnector(root_dir=str(tmp_path / "exports"))
        location = snapshot.export(kumo.SnapshotOutputConfig(exports, "saved"))
        shutil.rmtree(source)
        shutil.rmtree(tmp_path / "producer")
        kumo.configure(scratch_dir=tmp_path / "consumer")
        reopened = kumo.GraphSnapshot.load_from_connector(exports, path=location)
        exported_hashes = inventory(reopened.index_path)
        second = run(reopened, tmp_path / "work2")
        assert first.equals(second)
        assert inventory(reopened.index_path) == exported_hashes
    finally:
        kumo.configure(scratch_dir=previous)
