# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational reads inherit the final destination table's declared clock."""

from __future__ import annotations

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    BatchSpec,
    Columns,
    DataLoaderSpec,
    DataSchema,
    DType,
    EntitySource,
    EntitySpec,
    EventSpec,
    Field,
    Input,
    NonTemporal,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
)
from kumo.pytorch.contract.batch import Encode, Fields
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder

USER = EntitySpec("users", entity_type="user")
REQUEST = (SampleRequest("s1", 1, 100),)
USER_ROWS = [{"user_id": 1}]
EVENT_ROWS = [
    {"event_id": 1, "user_id": 1, "occurred_at": 110, "processed_at": 90, "amount": 1},
    {"event_id": 2, "user_id": 1, "occurred_at": 90, "processed_at": 110, "amount": 2},
]


def _event_schema(*, clock: str | NonTemporal = "occurred_at") -> DataSchema:
    return DataSchema(
        (
            TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
            TableSpec(
                "events",
                "event_id",
                {
                    "event_id": DType("int64"),
                    "user_id": DType("int64"),
                    "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                    "processed_at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                },
                time_column=clock,
            ),
        ),
        (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
    )


def _build(schema: DataSchema, sample: SampleSpec, tables=None):
    data = tables if tables is not None else {"users": USER_ROWS, "events": EVENT_ROWS}
    return InMemorySampleBuilder(schema, data).build(sample, REQUEST)[0]


def _value(sample, name: str):
    return next(field.value for field in sample.history.entity.values if field.key == name)


def _contract(schema: DataSchema, sample: SampleSpec, field: str) -> DataLoaderSpec:
    return DataLoaderSpec(
        schema,
        sample,
        BatchSpec(
            inputs={
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field=field, encoding=Raw()),), row=Columns()),
                    Separate(),
                )
            }
        ),
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        pytest.param(Rows(("user_events",)), 2, id="destination-default"),
        pytest.param(Rows(("user_events",), TimeWindow("processed_at")), 1, id="explicit-override"),
    ),
)
def test_rows_resolve_the_expected_clock(rows: Rows, expected: int) -> None:
    schema = _event_schema()
    sample = SampleSpec(
        USER,
        entity_values={"event_id_sum": Aggregate(rows, "sum", "event_id", dtype=DType("int64"))},
    )
    contract = _contract(schema, sample, "event_id_sum")
    assert DataLoaderSpec.from_json(contract.to_json()) == contract
    assert _value(_build(schema, sample), "event_id_sum") == expected


def test_only_the_final_destination_table_is_automatically_filtered() -> None:
    schema = DataSchema(
        (
            TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
            TableSpec(
                "accounts",
                "account_id",
                {
                    "account_id": DType("int64"),
                    "user_id": DType("int64"),
                    "opened_at": DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="opened_at",
            ),
            TableSpec(
                "events",
                "event_id",
                {
                    "event_id": DType("int64"),
                    "account_id": DType("int64"),
                    "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="occurred_at",
            ),
        ),
        (
            RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
            RelationSpec("account_events", "accounts", "account_id", "events", "account_id"),
        ),
    )
    sample = SampleSpec(
        USER,
        entity_values={
            "event_count": Aggregate(
                Rows(("user_accounts", "account_events")), "count", dtype=DType("int64")
            )
        },
    )
    tables = {
        "users": USER_ROWS,
        "accounts": [{"account_id": 10, "user_id": 1, "opened_at": 110}],
        "events": [{"event_id": 20, "account_id": 10, "occurred_at": 90}],
    }
    _contract(schema, sample, "event_count").validate()
    assert _value(_build(schema, sample, tables), "event_count") == 1


def test_a_destination_declared_non_temporal_is_unfiltered_by_declaration() -> None:
    schema = _event_schema(clock=NON_TEMPORAL)
    sample = SampleSpec(
        USER,
        entity_values={
            "event_count": Aggregate(Rows(("user_events",)), "count", dtype=DType("int64"))
        },
    )
    _contract(schema, sample, "event_count").validate()
    assert _value(_build(schema, sample), "event_count") == 2


def test_leaving_the_clock_unstated_is_not_expressible() -> None:
    with pytest.raises(TypeError, match="NON_TEMPORAL"):
        TableSpec("events", "event_id", {"event_id": DType("int64")}, time_column=None)
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        TableSpec("events", "event_id", {"event_id": DType("int64")})


def test_structural_columns_must_be_declared() -> None:
    with pytest.raises(ValueError, match="must declare its columns"):
        TableSpec("events", "event_id", {}, time_column=NON_TEMPORAL)
    with pytest.raises(ValueError, match="primary key"):
        TableSpec("events", "event_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL)
    with pytest.raises(ValueError, match="time column"):
        TableSpec("events", "event_id", {"event_id": DType("int64")}, time_column="occurred_at")
    with pytest.raises(ValueError, match="is not declared"):
        DataSchema(
            (
                TableSpec(
                    "users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL
                ),
                TableSpec(
                    "events", "event_id", {"event_id": DType("int64")}, time_column=NON_TEMPORAL
                ),
            ),
            (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
        )


@pytest.mark.parametrize(
    ("clock", "expected"),
    (
        pytest.param(None, "default", id="destination-default"),
        pytest.param("effective_at", "override", id="explicit-override"),
    ),
)
def test_joined_fields_resolve_the_expected_clock(clock: str | None, expected: str) -> None:
    schema = DataSchema(
        (
            TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
            TableSpec(
                "profiles",
                "profile_id",
                {
                    "profile_id": DType("int64"),
                    "user_id": DType("int64"),
                    "recorded_at": DType("timestamp", unit="s", timezone="UTC"),
                    "effective_at": DType("timestamp", unit="s", timezone="UTC"),
                    "status": DType("string"),
                },
                time_column="recorded_at",
            ),
        ),
        (RelationSpec("user_profiles", "users", "user_id", "profiles", "user_id"),),
    )
    sample = SampleSpec(
        USER,
        entity_values={
            "status": Field(
                column="status", path=("user_profiles",), selection="latest", time_column=clock
            )
        },
    )
    tables = {
        "users": USER_ROWS,
        "profiles": [
            {
                "profile_id": 1,
                "user_id": 1,
                "recorded_at": 80,
                "effective_at": 120,
                "status": "default",
            },
            {
                "profile_id": 2,
                "user_id": 1,
                "recorded_at": 120,
                "effective_at": 80,
                "status": "override",
            },
        ],
    }
    assert _value(_build(schema, sample, tables), "status") == expected


def test_event_rows_can_inherit_the_destination_clock() -> None:
    sample = SampleSpec(
        USER,
        events={"history": EventSpec(Rows(("user_events",)), {"amount": Field(column="amount")})},
    )
    assert [event.timestamp for event in _build(_event_schema(), sample).history.events] == [90]


def test_generated_targets_still_require_an_explicit_future_window() -> None:
    sample = SampleSpec(
        USER, targets={"future": Aggregate(Rows(("user_events",)), "count", dtype=DType("int64"))}
    )
    with pytest.raises(ValueError, match="future time window"):
        _build(_event_schema(), sample)
