# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitting reads its anchors in chunks of ``_FIT_CHUNK``; every anchor counts exactly once."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pytest

from kumo.pytorch import api
from kumo.pytorch.assembly import _FIT_CHUNK

SCHEMA = api.DataSchema(
    tables=(
        api.TableSpec(
            "users", "user_id", {"user_id": api.DType("int64")}, time_column=api.NON_TEMPORAL
        ),
        api.TableSpec(
            "events",
            "event_id",
            {
                "event_id": api.DType("int64"),
                "user_id": api.DType("int64"),
                "occurred_at": api.DType("timestamp", unit="s", timezone="UTC"),
                "amount": api.DType("float64"),
            },
            time_column="occurred_at",
        ),
    ),
    relations=(api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)


def _spent(start: int | None, stop: int) -> api.Aggregate:
    window = api.Rows(("user_events",), api.TimeWindow("occurred_at", start, stop))
    return api.Aggregate(window, "sum", "amount", dtype=api.DType("float64"))


# Inputs and a target fit from per-anchor values, which fitting builds chunk by chunk.
CONTRACT = api.DataLoaderSpec(
    SCHEMA,
    api.SampleSpec(
        api.EntitySpec("users"),
        entity_values={"spent": _spent(None, 0)},
        targets={"later": _spent(0, 100)},
    ),
    api.BatchSpec(
        inputs={
            "scaled": api.Input(
                api.EntitySource(), api.Fields((api.Encode("spent", api.ZScore()),)), api.Separate()
            ),
            "tokens": api.Input(
                api.EntitySource(),
                api.Fields((api.Encode("spent", api.Vocabulary()),)),
                api.Separate(),
            ),
        },
        targets={"later": api.LabelTarget("later", api.ZScore())},
    ),
)


def _data(engine_name: str, users: range, directory: Path) -> tuple[api.Dataset, api.Engine]:
    # User u spends u before its anchor at 100 and 10u after it.
    rows = {
        "users": [{"user_id": user} for user in users],
        "events": [
            {"event_id": 2 * user + later, "user_id": user, "occurred_at": at, "amount": amount}
            for user in users
            for later, at, amount in ((0, 50, float(user)), (1, 150, float(10 * user)))
        ],
    }
    if engine_name == "in_memory":
        tables = {name: api.InMemoryRows(values) for name, values in rows.items()}
        return api.Dataset(tables), api.InMemoryEngine()
    pytest.importorskip("diskgraph")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sources = {}
    for table in SCHEMA.tables:
        path = directory / f"{table.name}.parquet"
        schema = pa.schema([(name, dtype.to_arrow()) for name, dtype in table.columns.items()])
        pq.write_table(pa.Table.from_pylist(rows[table.name], schema=schema), path)
        sources[table.name] = api.Parquet(str(path))
    engine = api.DiskGraphEngine(index=directory / "index", num_threads=1, num_shards=1)
    return api.Dataset(sources), engine


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("anchors", (_FIT_CHUNK - 1, _FIT_CHUNK, _FIT_CHUNK + 1))
def test_anchors_on_both_sides_of_a_chunk_edge_are_each_fitted_once(
    tmp_path: Path, engine_name: str, anchors: int
) -> None:
    users = range(1, anchors + 1)
    dataset, engine = _data(engine_name, users, tmp_path)
    requests = [api.SampleRequest(f"u{user}", user, 100) for user in users]
    with api.prepare(CONTRACT, dataset, engine) as plan:
        with plan.materialize(requests) as task:
            scaled = task.output("scaled").field(name="spent").encoder
            tokens = task.output("tokens").field(name="spent").encoder
            later = task.target("later").field(name="value").encoder

    # The values 1..n have mean (n + 1) / 2 and population variance (n² - 1) / 12.
    mean, scale = (anchors + 1) / 2, sqrt((anchors**2 - 1) / 12)
    assert scaled.mean == pytest.approx(mean, rel=1e-12)
    assert scaled.scale == pytest.approx(scale, rel=1e-12)
    assert later.mean == pytest.approx(10 * mean, rel=1e-12)
    assert later.scale == pytest.approx(10 * scale, rel=1e-12)
    # Every value is an entry, and entries take the IDs after missing 0 and unknown 1.
    entries = dict(tokens.entries)
    assert sorted(entries) == [float(user) for user in users]
    assert sorted(entries.values()) == list(range(2, anchors + 2))
