# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.train import TrainingTableSpec

from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.pquery._custom_source import (
    stage_source_table,
    validate_prediction_source,
    validate_training_source,
)


def test_stage_source_table_streams_many_parts_to_one_parquet(tmp_path):
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

    assert output == tmp_path / "staged" / "custom.parquet"
    assert pq.read_table(output)["ENTITY1"].to_pylist() == [1, 2]
    assert pq.read_table(output)["TIMESTAMP"].to_pylist() == [100, 200]
    assert calls == [(connector, "curated",
                      str(tmp_path / "staged" / "source"))]


def test_stage_source_table_accepts_partitioned_file_connector_directory(
        tmp_path):
    root = tmp_path / "connector"
    parts = root / "parts"
    parts.mkdir(parents=True)
    pq.write_table(pa.table({"ENTITY1": [1]}), parts / "one.parquet")
    pq.write_table(pa.table({"ENTITY1": [2]}), parts / "two.parquet")
    connector = FileConnector(root_dir=str(root))

    output = stage_source_table(connector["parts"], tmp_path / "staged")

    assert output == tmp_path / "staged" / "custom.parquet"
    assert sorted(pq.read_table(output)["ENTITY1"].to_pylist()) == [1, 2]


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


def test_training_source_rejects_negative_weights(tmp_path):
    source = tmp_path / "negative.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1": [1, 2],
            "TARGET1": [0, 1],
            "WEIGHT": [-1.0, 1.0],
        }),
        source,
    )

    with pytest.raises(ValueError, match="negative"):
        validate_training_source(
            source,
            _QUERY,
            _Graph(),
            TrainingTableSpec(weight_col="WEIGHT"),
        )


def test_prediction_source_rejects_null_entity_keys(tmp_path):
    source = tmp_path / "null-key.parquet"
    pq.write_table(pa.table({"ENTITY1": pa.array([1, None])}), source)

    with pytest.raises(ValueError, match="ENTITY1 contains N/A"):
        validate_prediction_source(source, _QUERY, _Graph())
