# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.engine.connector import _staging
from kumo.engine.connector._staging import stage_source_table
from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.source_table import SourceTable


def test_stage_source_table_retains_many_parts(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"ENTITY1": [1], "TIMESTAMP": [100]}), first)
    pq.write_table(pa.table({"ENTITY1": [2], "TIMESTAMP": [200]}), second)
    calls = []

    class Backend:
        @staticmethod
        def prepare_ingest(connector, table_name, destination, dtypes, sql):
            assert dtypes is None
            assert sql is None
            calls.append((connector, table_name, destination))
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


@pytest.mark.parametrize("layout", ["files", "directories", "nested"])
@pytest.mark.parametrize("link_error", [None, errno.EPERM, errno.EOPNOTSUPP])
def test_stage_source_table_preserves_parquet_parts(tmp_path, monkeypatch,
                                                    layout, link_error):
    paths = []
    originals = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        if layout == "nested":
            directory = directory / "partition"
            directory.mkdir()
        part = directory / "same-name.parquet"
        table = pa.table({"ENTITY1": [index] * 8})
        pq.write_table(table, part, row_group_size=1)
        originals.append(part)
        paths.append(part if layout == "files" else tmp_path / str(index))
    if link_error is not None:

        def refuse_link(*args, **kwargs):
            raise OSError(link_error, "symlinks unavailable")

        monkeypatch.setattr(_staging.os, "symlink", refuse_link)
    backend = Mock()
    backend.prepare_ingest.return_value = SimpleNamespace(
        source={
            "type": "parquet",
            "paths": [str(p) for p in paths]
        }, credentials=None, version=None)
    source = SourceTable("curated", Mock(source_backend=backend))

    output, _ = stage_source_table(source, tmp_path / "staged")

    retained = sorted(output.glob("*.parquet"))
    assert len(retained) == 2
    for original, part in zip(originals, retained):
        assert part.is_symlink() == (link_error is None)
        if part.is_symlink():
            assert not part.readlink().is_absolute()
            assert part.resolve() == original.resolve()
        assert part.read_bytes() == original.read_bytes()
        assert pq.read_metadata(part).num_row_groups == 8
    assert pq.read_table(output).num_rows == 16


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
    assert parts[0].is_symlink()
    assert parts[0].read_bytes() == (root / "table.parquet").read_bytes()
    assert pq.read_table(output)["ENTITY1"].to_pylist() == [1, 2]
    (tmp_path / "staged").rename(tmp_path / "moved")
    assert pq.read_table(tmp_path / "moved" / "parts")[
        "ENTITY1"].to_pylist() == [1, 2]


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
    assert len(list((tmp_path / "staged" / "source").rglob("*.parquet"))) > 1
    assert len(list(output.glob("*.parquet"))) == 5
    expected_schema = table.schema.set(
        2, pa.field("weight", pa.list_(pa.float32())))
    assert pq.read_table(output).equals(table.cast(expected_schema))
