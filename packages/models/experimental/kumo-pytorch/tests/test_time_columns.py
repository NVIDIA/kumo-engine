# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columns compared with the request time must be timestamp or date32."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    ActiveInterval,
    Aggregate,
    BatchSpec,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    Field,
    Fields,
    InMemoryEngine,
    InMemoryRows,
    Input,
    Parquet,
    Raw,
    RelationSpec,
    Rows,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
    ValidationError,
    check,
    prepare,
)
from kumo.pytorch.contract.sample import GraphLayer, GraphSample, NeighborhoodSpec

SCHEMA = DataSchema(
    (
        TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
        TableSpec(
            "events",
            "event_id",
            {
                "event_id": DType("int64"),
                "user_id": DType("int64"),
                "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                "booked_on": DType("date32"),
                "amount": DType("float64"),
                "sequence": DType("int64"),
            },
            time_column="occurred_at",
        ),
    ),
    (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)


def _window(column: str) -> SampleSpec:
    rows = Rows(("user_events",), TimeWindow(column, None, 0))
    return SampleSpec(
        EntitySpec("users"),
        entity_values={"value": Aggregate(rows, "count", dtype=DType("int64"))},
    )


def _field(column: str) -> SampleSpec:
    return SampleSpec(
        EntitySpec("users"),
        entity_values={
            "value": Field(
                column="amount", path=("user_events",), selection="latest", time_column=column
            )
        },
    )


def _traversal(column: str) -> SampleSpec:
    layer = GraphLayer(
        samples=(
            GraphSample(relation="user_events", fanout=2, strategy="recent", time_column=column),
        )
    )
    return SampleSpec(
        EntitySpec("users"),
        entity_values={"value": Field(column="user_id")},
        neighborhoods={"events": NeighborhoodSpec(layers=(layer,))},
    )


def _available_from(column: str) -> SampleSpec:
    return SampleSpec(
        EntitySpec("events", available_from=column),
        entity_values={"value": Field(column="amount")},
    )


def _active(column: str, end: bool) -> SampleSpec:
    columns = ("occurred_at", column) if end else (column, "occurred_at")
    rows = Rows(("user_events",), active=ActiveInterval(*columns))
    return SampleSpec(
        EntitySpec("users"),
        entity_values={"value": Aggregate(rows, "count", dtype=DType("int64"))},
    )


SHAPES = {
    "window": (_window, "sample.entity_values.value.source.time.column"),
    "available_from": (_available_from, "sample.entity.available_from"),
    "active_start": (
        lambda column: _active(column, end=False),
        "sample.entity_values.value.source.active.start_column",
    ),
    "active_end": (
        lambda column: _active(column, end=True),
        "sample.entity_values.value.source.active.end_column",
    ),
    "field": (_field, "sample.entity_values.value.time_column"),
    "traversal": (_traversal, "sample.neighborhoods.events.layers[0].samples[0].time_column"),
}


def _contract(sample: SampleSpec) -> DataLoaderSpec:
    return DataLoaderSpec(
        SCHEMA,
        sample,
        BatchSpec(
            inputs={
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field="value", encoding=Raw()),)),
                    Separate(),
                )
            }
        ),
    )


def _time_issues(contract: DataLoaderSpec, path: str):
    issues = check(
        contract, Dataset({"users": InMemoryRows([]), "events": InMemoryRows([])}), InMemoryEngine()
    )
    return [issue for issue in issues if issue.path == path]


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("column", ["amount", "sequence"])
def test_a_value_column_is_refused_as_a_time(shape: str, column: str) -> None:
    build, path = SHAPES[shape]
    contract = _contract(build(column))
    (issue,) = _time_issues(contract, path)
    assert f"events.{column}" in issue.message
    assert "timestamp or date32" in issue.message
    assert issue.suggestion
    dataset = Dataset({"users": InMemoryRows([]), "events": InMemoryRows([])})
    engine = InMemoryEngine()
    with patch.object(engine, "open", side_effect=AssertionError("opened before refusal")):
        with pytest.raises(ValidationError) as error:
            prepare(contract, dataset, engine)
    assert issue in error.value.issues


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("column", ["occurred_at", "booked_on"])
def test_timestamp_and_date32_columns_are_times(shape: str, column: str) -> None:
    build, path = SHAPES[shape]
    assert _time_issues(_contract(build(column)), path) == []


def test_diskgraph_refuses_a_window_on_a_value_column_before_reading_data(tmp_path) -> None:
    pytest.importorskip("diskgraph")
    from kumo.pytorch.api import DiskGraphEngine

    contract = _contract(_window("amount"))
    dataset = Dataset(
        {name: Parquet(path=str(tmp_path / f"{name}.parquet")) for name in ("users", "events")}
    )
    engine = DiskGraphEngine(index=tmp_path / "index")
    with patch.object(engine, "open", side_effect=AssertionError("opened before refusal")):
        issues = check(contract, dataset, engine)
    path = SHAPES["window"][1]
    assert [issue.path for issue in issues if "amount" in issue.message] == [path]
    assert not (tmp_path / "index").exists()
