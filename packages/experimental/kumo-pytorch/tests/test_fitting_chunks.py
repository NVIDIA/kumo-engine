# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime
from math import sqrt
from pathlib import Path

import pytest

from kumo.pytorch import api
from kumo.pytorch.assembly import _FIT_CHUNK
from tests.test_anchor_tables import SCHEMA
from tests.test_diskgraph_backend import _write_parquet


def _spent(start: int | None, stop: int) -> api.Aggregate:
    window = api.Rows(("user_events",), api.TimeWindow("occurred_at", start, stop))
    return api.Aggregate(window, "sum", "amount", dtype=api.DType("float64"))


CONTRACT = api.DataLoaderSpec(
    SCHEMA,
    api.SampleSpec(
        api.EntitySpec("users"),
        entity_values={"spent": _spent(None, 0)},
        targets={"later": _spent(0, 100)},
    ),
    api.BatchSpec(
        inputs={
            "spent": api.Input(
                api.EntitySource(), api.Fields((api.Encode("spent", api.ZScore()),)), api.Separate()
            )
        },
        targets={"later": api.LabelTarget("later", api.ZScore())},
    ),
)


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("anchors", (_FIT_CHUNK - 1, _FIT_CHUNK, _FIT_CHUNK + 1))
def test_anchors_on_both_sides_of_a_chunk_edge_are_each_fitted_once(
    tmp_path: Path, engine_name: str, anchors: int
) -> None:
    users = range(1, anchors + 1)
    before, after = (datetime.fromtimestamp(at, UTC) for at in (50, 150))
    rows = {
        "users": [{"user_id": user, "score": 0.0} for user in users],
        "events": [
            {"event_id": 2 * user + i, "user_id": user, "occurred_at": at, "amount": amount}
            for user in users
            for i, (at, amount) in enumerate(((before, float(user)), (after, 10.0 * user)))
        ],
    }
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
        dataset = _write_parquet(rows, tmp_path / "parquet")
        engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1, num_shards=1)
    else:
        dataset = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
        engine = api.InMemoryEngine()
    requests = [api.SampleRequest(f"u{user}", user, 100) for user in users]
    with api.prepare(CONTRACT, dataset, engine) as plan, plan.materialize(requests) as task:
        spent = task.output("spent").field(name="spent").encoder
        later = task.target("later").field(name="value").encoder

    mean, scale = (anchors + 1) / 2, sqrt((anchors**2 - 1) / 12)
    assert (spent.mean, spent.scale) == pytest.approx((mean, scale), rel=1e-12)
    assert (later.mean, later.scale) == pytest.approx((10 * mean, 10 * scale), rel=1e-12)
