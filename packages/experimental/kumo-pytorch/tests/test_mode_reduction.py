# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The categorical mode reduction has deterministic public semantics."""

from __future__ import annotations

import pytest

from kumo.pytorch import build
from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    DType,
    EntitySpec,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    TableSpec,
    TimeWindow,
)
from kumo.pytorch.contract.sample import ALL_REDUCTIONS
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder

SCHEMA = DataSchema(
    (
        TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
        TableSpec(
            "events",
            "event_id",
            {
                "event_id": DType("int64"),
                "user_id": DType("int64"),
                "at": DType("timestamp", unit="s", timezone="UTC"),
                "category": DType("string"),
            },
            time_column="at",
        ),
    ),
    (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)


def _mode(rows: list[dict[str, object]]) -> str | None:
    sample = SampleSpec(
        EntitySpec("users", entity_type="user"),
        entity_values={
            "favorite": Aggregate(
                Rows(("user_events",), TimeWindow("at")),
                "mode",
                "category",
                dtype=DType("string"),
            )
        },
    )
    built = InMemorySampleBuilder(SCHEMA, {"users": [{"user_id": 1}], "events": rows}).build(
        sample, (SampleRequest("s1", 1, 100),)
    )[0]
    return next(
        (field.value for field in built.history.entity.values if field.key == "favorite"), None
    )


def test_mode_ignores_nulls_and_uses_a_stable_tie_break() -> None:
    rows = [
        {"event_id": 4, "user_id": 1, "at": 40, "category": "b"},
        {"event_id": 2, "user_id": 1, "at": 20, "category": "a"},
        {"event_id": 3, "user_id": 1, "at": 30, "category": "b"},
        {"event_id": 1, "user_id": 1, "at": 10, "category": "a"},
        {"event_id": 5, "user_id": 1, "at": 50, "category": None},
    ]
    assert _mode(rows) == "a"
    assert _mode(list(reversed(rows))) == "a"


def test_mode_of_an_empty_or_all_null_window_is_missing() -> None:
    assert _mode([]) is None
    assert _mode([{"event_id": 1, "user_id": 1, "at": 10, "category": None}]) is None


def test_mode_requires_the_column_result_dtype_and_is_backend_visible() -> None:
    from kumo.pytorch.contract.sample import value_dtype

    with pytest.raises(ValueError, match="cannot produce"):
        value_dtype(
            SCHEMA,
            "users",
            Aggregate(
                Rows(("user_events",), TimeWindow("at")), "mode", "category", dtype=DType("float64")
            ),
        )
    assert "mode" in ALL_REDUCTIONS


def test_authoring_helper_lowers_to_the_same_operation() -> None:
    users = build.Table(
        "users", key="user_id", columns={"user_id": DType("int64")}, time=NON_TEMPORAL
    )
    events = build.Table(
        "events",
        key="event_id",
        columns={
            "event_id": DType("int64"),
            "user_id": DType("int64"),
            "at": DType("timestamp", unit="s", timezone="UTC"),
            "category": DType("string"),
        },
        time="at",
    )
    path = build.Link("user_events", users.user_id, events.user_id)
    schema = build.Schema(tables=(users, events), links=(path,))
    lowered = build.Mode(events.category, via=path, dtype=DType("string")).lower(schema, "users")
    assert lowered.operation == "mode"
    assert lowered.dtype == DType("string")
