# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time and exact arithmetic through joins, grouping, filters and physical sharding."""

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import INT, TEXT, TIME, batches, fields, materialized, record


@pytest.mark.parametrize("timed_hop", ["middle", "leaf"])
@pytest.mark.parametrize("seed", [0, 17])
def test_future_poison_at_each_hop(tmp_path, timed_hop, seed):
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT, "at": TIME}, time_column="at"),
            r.TableSpec(
                "accounts",
                "id",
                {"id": INT, "owner": INT, "at": TIME},
                time_column="at" if timed_hop == "middle" else r.NON_TEMPORAL,
            ),
            r.TableSpec(
                "events",
                "id",
                {"id": INT, "account": INT, "at": TIME, "value": INT},
                time_column="at",
            ),
        ),
        (
            r.RelationSpec("owns", "users", "id", "accounts", "owner"),
            r.RelationSpec("activity", "accounts", "id", "events", "account"),
        ),
    )
    query = r.Rows(("owns", "activity"), r.TimeWindow("at", stop=0))
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("users"),
            entity_values={
                "n": r.Aggregate(query, "count", dtype=INT),
                "total": r.Aggregate(query, "sum", "value", dtype=INT),
            },
        ),
        r.BatchSpec(
            {"values": r.Input(r.EntitySource(), fields("n", "total"), r.Separate())}
        ),
    )
    rows = {
        "users": [{"id": 1, "at": 0}, {"id": 2, "at": 0}],
        "accounts": [
            {"id": 10, "owner": 1, "at": 10},
            {"id": 20, "owner": 2, "at": 101 if timed_hop == "middle" else 10},
        ],
        "events": [
            {"id": 1, "account": 10, "at": 100, "value": 7},
            {
                "id": 2,
                "account": 20,
                "at": 101 if timed_hop == "leaf" else 99,
                "value": 9007199254740993,
            },
        ],
    }
    with materialized(
        contract, rows, [record("safe", 1), record("poison", 2)], tmp_path, seed=seed
    ) as (task, _, _):
        batch = next(batches(task, batch_size=2))
        values = batch.inputs["values"]
        assert values["n"]["values"].tolist() == [1, 0]
        assert values["total"]["valid"].tolist() == [True, True]
        # Empty selections have additive-identity sums (G-CALC-001).
        assert values["total"]["values"].tolist() == [7, 0]


@pytest.mark.parametrize("grouped", [False, True], ids=["direct", "grouped"])
@pytest.mark.parametrize("filtered", [False, True], ids=["window", "filtered-window"])
def test_exact_cancellation_after_selection(tmp_path, grouped, filtered):
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT}, time_column=r.NON_TEMPORAL),
            r.TableSpec(
                "events",
                "id",
                {"id": INT, "owner": INT, "at": TIME, "value": INT, "group": TEXT},
                time_column="at",
            ),
        ),
        (r.RelationSpec("events", "users", "id", "events", "owner"),),
    )
    query = r.Rows(
        ("events",),
        r.TimeWindow("at", -10, 0),
        filters=(r.RowFilter("value", "not_null"),) if filtered else (),
    )
    source = (
        r.GroupBy(query, ("group",), {"subtotal": r.Reduce("sum", "value", dtype=INT)})
        if grouped
        else query
    )
    total = r.Aggregate(source, "sum", "subtotal" if grouped else "value", dtype=INT)
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("users"),
            entity_values={
                "difference": r.Derived("subtract", ("total", "one")),
                "one": r.Constant(1, INT),
                "total": total,
            },
        ),
        r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(), fields("total", "difference"), r.Separate()
                )
            }
        ),
    )
    values = [
        (90, 777, "a"),
        (91, 2**53 + 1, "a"),
        (92, 1, "a"),
        (99, -(2**53), "b"),
        (100, None, "b"),
        (101, 999, "a"),
    ]
    rows = {
        "users": [{"id": 1}],
        "events": [
            {"id": i, "owner": 1, "at": at, "value": value, "group": group}
            for i, (at, value, group) in enumerate(values)
        ],
    }
    with materialized(contract, rows, [record("sum", 1)], tmp_path) as (task, _, _):
        out = next(batches(task)).inputs["values"]
        for name, expected in [("total", 2), ("difference", 1)]:
            assert out[name]["values"].dtype == torch.int64
            assert out[name]["valid"].tolist() == [True]
            assert out[name]["values"].tolist() == [expected]
