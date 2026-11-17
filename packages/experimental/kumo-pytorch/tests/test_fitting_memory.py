# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the peak memory of fitting at 25k and 400k anchors. It has no pass or fail limit."""

from __future__ import annotations

import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kumo.pytorch import api
from tests.test_anchor_tables import SCHEMA
from tests.test_diskgraph_backend import _write_parquet

# The data stays the same; only the anchor count changes. 400k anchors put each user at 16 times.
USERS, TIMES = 25_000, 16


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


def _status(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) * 1024
    raise KeyError(field)


def _engine(index: Path) -> api.DiskGraphEngine:
    return api.DiskGraphEngine(index=index, num_threads=1, num_shards=1)


def _fit(dataset: api.Dataset, anchors: Path, index: Path) -> tuple[float, float]:
    """Return the peak RSS of fitting and the part it adds, in MiB.

    Run it in a new process that only reopens the index. Memory that earlier work in the
    same process freed would be reused by fitting and hide part of its peak.
    """
    with api.prepare(CONTRACT, dataset, _engine(index)) as plan:
        gc.collect()
        before = _status("VmRSS")
        Path("/proc/self/clear_refs").write_text("5")  # reset the peak RSS to the current RSS
        with plan.materialize(api.AnchorTable(api.Parquet(str(anchors)), "user_id", "at")):
            peak = _status("VmHWM")
    return peak / 2**20, (peak - before) / 2**20


@pytest.mark.t2b
@pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="reads Linux /proc")
def test_record_fitting_memory_at_25k_and_400k_anchors(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    users = range(1, USERS + 1)
    rows = {
        "users": [{"user_id": user, "score": 0.0} for user in users],
        "events": [
            {
                "event_id": user * (TIMES + 1) + step,
                "user_id": user,
                "occurred_at": datetime.fromtimestamp(50 + 100 * step, UTC),
                "amount": float(user + step),
            }
            for user in users
            for step in range(TIMES + 1)
        ],
    }
    dataset = _write_parquet(rows, tmp_path / "parquet")
    with api.prepare(CONTRACT, dataset, _engine(tmp_path / "index")):
        pass  # build the index here, outside the measured processes
    results = {}
    for count in (25_000, 400_000):
        times = range(100, 100 * (count // USERS) + 1, 100)
        anchors = tmp_path / f"anchors-{count}.parquet"
        user_ids = pa.array([user for _ in times for user in users], pa.int64())
        at = pa.array([time for time in times for _ in users], pa.timestamp("s", tz="UTC"))
        pq.write_table(pa.table({"user_id": user_ids, "at": at}), anchors)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(1, mp_context=context) as pool:
            results[count] = pool.submit(_fit, dataset, anchors, tmp_path / "index").result()
    print(
        "Fitting peak RSS on DiskGraph: "
        + "; ".join(
            f"{count:,} anchors {peak:.0f} MiB, fitting adds {added:.1f} MiB"
            for count, (peak, added) in results.items()
        )
    )
