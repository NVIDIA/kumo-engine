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
from kumoapi.train import TrainingTableSpec

from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.pquery import _custom_source
from kumo.engine.pquery._custom_source import (
    stage_source_table,
    validate_prediction_source,
    validate_training_source,
)


def test_stage_source_table_retains_many_parts(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"ENTITY1": [1], "TIMESTAMP": [100]}), first)
    pq.write_table(pa.table({"ENTITY1": [2], "TIMESTAMP": [200]}), second)
    calls = []

    class Backend:
        @staticmethod
        def prepare_ingest(connector, table_name, destination, sql):
            assert sql is None
            calls.append((connector, table_name, destination))
            return SimpleNamespace(
                source={
                    "type": "parquet",
                    "paths": [str(first), str(second)],
                },
                credentials=None,
            )

    connector = SimpleNamespace(source_backend=Backend())
    source = SourceTable("curated", connector)
    output = stage_source_table(source, tmp_path / "staged")

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

    output = stage_source_table(connector["parts"], tmp_path / "staged")

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

        monkeypatch.setattr(_custom_source.os, "symlink", refuse_link)
    backend = Mock()
    backend.prepare_ingest.return_value = SimpleNamespace(
        source={
            "type": "parquet",
            "paths": [str(p) for p in paths]
        }, credentials=None)
    source = SourceTable("curated", Mock(source_backend=backend))

    output = stage_source_table(source, tmp_path / "staged")

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

    output = stage_source_table(
        FileConnector(root_dir=str(root))["curated"], tmp_path / "staged")

    assert pq.read_table(output).schema == pa.schema([("ENTITY1", pa.int64()),
                                                      ("TARGET1", pa.string())])


def test_stage_source_table_retains_single_file_in_parts_directory(tmp_path):
    root = tmp_path / "connector"
    root.mkdir()
    pq.write_table(pa.table({"ENTITY1": [1, 2]}), root / "table.parquet")
    connector = FileConnector(root_dir=str(root))

    output = stage_source_table(connector["table.parquet"], tmp_path / "staged")

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

    output = stage_source_table(connector["weighted.parquet"],
                                tmp_path / "staged")

    assert output == tmp_path / "staged" / "parts"
    assert len(list((tmp_path / "staged" / "source").rglob("*.parquet"))) > 1
    assert len(list(output.glob("*.parquet"))) == 5
    expected_schema = table.schema.set(
        2, pa.field("weight", pa.list_(pa.float32())))
    assert pq.read_table(output).equals(table.cast(expected_schema))


class _EntityTable:
    primary_key = "id"
    time_column = None

    @staticmethod
    def column(name):
        assert name == "id"
        return SimpleNamespace(dtype="int64")


class _Graph:
    @staticmethod
    def table(name):
        assert name == "users"
        return _EntityTable()


_QUERY = SimpleNamespace(
    task_type="binary_classification",
    query_type="static",
    entity_table="users",
    list_distinct_target_is_array=False,
)


def test_training_source_defers_negative_weights_to_ingestion(
        tmp_path, monkeypatch):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("source validation must not scan rows")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", unexpected_scan)
    source = tmp_path / "negative.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1": [1, 2],
            "TARGET1": [0, 1],
            "WEIGHT": [-1.0, 1.0],
        }),
        source,
    )

    validate_training_source(
        source,
        _QUERY,
        _Graph(),
        TrainingTableSpec(weight_col="WEIGHT"),
    )


def test_prediction_source_defers_null_entity_keys_to_ingestion(tmp_path):
    source = tmp_path / "null-key.parquet"
    pq.write_table(pa.table({"ENTITY1": pa.array([1, None])}), source)

    validate_prediction_source(source, _QUERY, _Graph())


@pytest.mark.parametrize("prediction", [False, True])
def test_validation_defers_later_part_contents(
        tmp_path, monkeypatch, prediction):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("source validation must not scan rows")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", unexpected_scan)
    first = {"ENTITY1": [1, 2]}
    second = {"ENTITY1": pa.array([None], pa.int64())}
    if not prediction:
        first["TARGET1"] = [0, 1]
        second["TARGET1"] = [1]
    pq.write_table(pa.table(first), tmp_path / "part-0.parquet")
    pq.write_table(pa.table(second), tmp_path / "part-1.parquet")
    if prediction:
        validate_prediction_source(tmp_path, _QUERY, _Graph())
    else:
        validate_training_source(tmp_path, _QUERY, _Graph(), TrainingTableSpec())


def test_training_validation_allows_empty_part(tmp_path):
    table = pa.table({"ENTITY1": [1, 2], "TARGET1": [0, 1]})
    pq.write_table(table.slice(0, 0), tmp_path / "part-0.parquet")
    pq.write_table(table, tmp_path / "part-1.parquet")
    validate_training_source(tmp_path, _QUERY, _Graph(), TrainingTableSpec())


def test_training_validation_rejects_schema_mismatch(tmp_path):
    pq.write_table(pa.table({
        "ENTITY1": [1],
        "TARGET1": [0]
    }), tmp_path / "part-0.parquet")
    pq.write_table(pa.table({"ENTITY1": [2]}), tmp_path / "part-1.parquet")
    with pytest.raises(ValueError, match="matching schemas"):
        validate_training_source(tmp_path, _QUERY, _Graph(),
                                 TrainingTableSpec())
