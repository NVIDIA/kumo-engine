# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.train import TrainingTableSpec

from kumo.engine.pquery._custom_source import (
    validate_prediction_source,
    validate_training_source,
)


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
