# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.engine.connector._staging import stage_source_table
from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.source_table import SourceTable


def test_stage_source_table_retains_many_parts(tmp_path):
    calls = []

    class Backend:
        @staticmethod
        def prepare_ingest(connector, table_name, destination, dtypes, sql):
            assert dtypes is None
            assert sql is None
            calls.append((connector, table_name, destination))
            first = Path(destination) / "first.parquet"
            second = Path(destination) / "second.parquet"
            first.parent.mkdir()
            pq.write_table(pa.table({"ENTITY1": [1], "TIMESTAMP": [100]}),
                           first)
            pq.write_table(pa.table({"ENTITY1": [2], "TIMESTAMP": [200]}),
                           second)
            return SimpleNamespace(
                source={
                    "type": "parquet",
                    "paths": [str(first), str(second)],
                },
                credentials=None,
                version=None,
            )

    connector = SimpleNamespace(source_backend=Backend())
    source = SourceTable("curated", connector)
    output, _ = stage_source_table(source, tmp_path / "staged")

    assert output == tmp_path / "staged" / "parts"
    assert pq.read_table(output)["ENTITY1"].to_pylist() == [1, 2]
    assert pq.read_table(output)["TIMESTAMP"].to_pylist() == [100, 200]
    assert calls == [(connector, "curated", str(tmp_path / "staged" / "source"))
                     ]


def test_stage_source_table_accepts_partitioned_file_connector_directory(
        tmp_path):
    root = tmp_path / "connector"
    parts = root / "parts"
    parts.mkdir(parents=True)
    pq.write_table(pa.table({"ENTITY1": [1]}), parts / "one.parquet")
    pq.write_table(pa.table({"ENTITY1": [2]}), parts / "two.parquet")
    connector = FileConnector(root_dir=str(root))

    output, _ = stage_source_table(connector["parts"], tmp_path / "staged")

    assert output == tmp_path / "staged" / "parts"
    assert sorted(pq.read_table(output)["ENTITY1"].to_pylist()) == [1, 2]


@pytest.mark.parametrize("with_dtypes", [False, True])
def test_stage_source_table_keeps_hive_partition_keys(tmp_path, with_dtypes):
    root = tmp_path / "connector"
    for day in ("2024-01-01", "2024-01-02"):
        directory = root / "events" / f"day={day}"
        directory.mkdir(parents=True)
        pq.write_table(pa.table({"ENTITY1": [1]}), directory / "part.parquet")
    source = FileConnector(root_dir=str(root))["events"]
    dtypes = ({column.name: column.dtype
               for column in source.columns} if with_dtypes else None)

    output, _ = stage_source_table(source, tmp_path / "staged", dtypes=dtypes)

    assert sorted(pq.read_table(output)["day"].to_pylist()) == [
        "2024-01-01", "2024-01-02"
    ]


@pytest.mark.parametrize("layout", ["files", "directories", "nested"])
def test_stage_source_table_preserves_parquet_parts(tmp_path, layout):
    contents = []

    def prepare_ingest(connector, table_name, destination, **_):
        paths = []
        for index in range(2):
            directory = Path(destination) / str(index)
            if layout == "nested":
                directory = directory / "partition"
            directory.mkdir(parents=True)
            part = directory / "same-name.parquet"
            pq.write_table(pa.table({"ENTITY1": [index] * 8}), part,
                           row_group_size=1)
            contents.append(part.read_bytes())
            paths.append(part if layout == "files" else Path(destination) /
                         str(index))
        return SimpleNamespace(source={
            "type": "parquet",
            "paths": [str(p) for p in paths]
        }, credentials=None, version=None)

    backend = Mock()
    backend.prepare_ingest.side_effect = prepare_ingest
    source = SourceTable("curated", Mock(source_backend=backend))

    output, _ = stage_source_table(source, tmp_path / "staged")

    parts = sorted(output.glob("*.parquet"))
    assert [part.read_bytes() for part in parts] == contents
    assert not any(part.is_symlink() for part in parts)
    assert not list((tmp_path / "staged" / "source").rglob("*.parquet"))


@pytest.mark.parametrize("key_in_file", [True, False])
def test_stage_source_table_requires_hive_keys_in_files(tmp_path, key_in_file):

    def prepare_ingest(connector, table_name, destination, **_):
        directory = Path(destination) / "day=2024-01-01"
        directory.mkdir(parents=True)
        columns = {"ENTITY1": [1]}
        if key_in_file:
            columns["day"] = ["2024-01-01"]
        pq.write_table(pa.table(columns), directory / "part.parquet")
        return SimpleNamespace(source={
            "type": "parquet",
            "paths": [destination]
        }, credentials=None, version=None)

    backend = Mock()
    backend.prepare_ingest.side_effect = prepare_ingest
    source = SourceTable("curated", Mock(source_backend=backend))

    if key_in_file:
        output, _ = stage_source_table(source, tmp_path / "staged")
        assert pq.read_table(output).column_names == ["ENTITY1", "day"]
    else:
        with pytest.raises(RuntimeError, match=r"\['day'\]"):
            stage_source_table(source, tmp_path / "staged")


@pytest.mark.parametrize("linked_source", [False, True])
def test_stage_source_table_refuses_files_outside_its_directory(
        tmp_path, linked_source):
    outside = tmp_path / "user" / "data.parquet"
    outside.parent.mkdir()
    pq.write_table(pa.table({"ENTITY1": [1]}), outside)

    def prepare_ingest(connector, table_name, destination, **_):
        path = outside
        if linked_source:
            Path(destination).symlink_to(outside.parent)
            path = Path(destination) / outside.name
        return SimpleNamespace(source={
            "type": "parquet",
            "paths": [str(path)]
        }, credentials=None, version=None)

    backend = Mock()
    backend.prepare_ingest.side_effect = prepare_ingest
    source = SourceTable("curated", Mock(source_backend=backend))

    with pytest.raises(RuntimeError, match="outside"):
        stage_source_table(source, tmp_path / "staged")

    assert outside.exists()


def test_stage_source_table_casts_to_what_the_table_reader_reads(tmp_path):
    from decimal import Decimal

    root = tmp_path / "connector"
    root.mkdir()
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array([Decimal(1), Decimal(2)], pa.decimal128(38, 0)),
            "TARGET1": pa.array(["a", "b"]).dictionary_encode(),
        }), root / "curated.parquet")

    output, _ = stage_source_table(
        FileConnector(root_dir=str(root))["curated"], tmp_path / "staged")

    assert pq.read_table(output).schema == pa.schema([("ENTITY1", pa.int64()),
                                                      ("TARGET1", pa.string())])


def test_stage_source_table_keeps_stored_types_of_the_columns_in_dtypes(
        tmp_path):
    from decimal import Decimal

    root = tmp_path / "connector"
    root.mkdir()
    table = pa.table({
        "ENTITY1":
        pa.array([Decimal(1), Decimal(2)], pa.decimal128(38, 0)),
        "AMOUNT":
        pa.array([Decimal("123456.78"), Decimal("0.01")], pa.decimal128(12, 2)),
        "NOTE": ["a", "b"],
    })
    pq.write_table(table, root / "curated.parquet")
    source = FileConnector(root_dir=str(root))["curated"]
    columns = source.column_dict
    dtypes = {name: columns[name].dtype for name in ("ENTITY1", "AMOUNT")}

    output, _ = stage_source_table(source, tmp_path / "staged", dtypes=dtypes)

    assert pq.read_table(output).equals(table.select(["ENTITY1", "AMOUNT"]))


def test_stage_source_table_retains_single_file_in_parts_directory(tmp_path):
    root = tmp_path / "connector"
    root.mkdir()
    pq.write_table(pa.table({"ENTITY1": [1, 2]}), root / "table.parquet")
    connector = FileConnector(root_dir=str(root))

    output, _ = stage_source_table(connector["table.parquet"], tmp_path / "staged")

    assert output == tmp_path / "staged" / "parts"
    parts = list(output.glob("*.parquet"))
    assert len(parts) == 1
    assert parts[0].read_bytes() == (root / "table.parquet").read_bytes()
    assert pq.read_table(output)["ENTITY1"].to_pylist() == [1, 2]


def test_stage_source_table_retains_reencoded_float64_weight_parts(
        tmp_path, monkeypatch):
    from kumo_core.sources import object_store

    # Force the connector's normal re-encode path to produce small shards.
    monkeypatch.setattr(
        object_store, "materialize_arrow_streams",
        partial(object_store.materialize_arrow_streams, max_rows_per_shard=2))
    root = tmp_path / "connector"
    root.mkdir()
    table = pa.table({
        "ENTITY1":
        list(range(9)),
        "TARGET1":
        pa.array([[1, 2]] * 9, pa.list_(pa.int64())),
        "weight":
        pa.array([[0.25, 0.75]] * 9, pa.list_(pa.float64())),
    })
    pq.write_table(table, root / "weighted.parquet")
    connector = FileConnector(root_dir=str(root))

    output, _ = stage_source_table(connector["weighted.parquet"],
                                   tmp_path / "staged")

    assert output == tmp_path / "staged" / "parts"
    assert len(list(output.glob("*.parquet"))) == 5
    expected_schema = table.schema.set(
        2, pa.field("weight", pa.list_(pa.float32())))
    assert pq.read_table(output).equals(table.cast(expected_schema))
