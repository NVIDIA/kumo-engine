# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default encoder cutoff comes from native anchor columns, not per-anchor requests."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import cutoff_nanos
from kumo.pytorch.engines.native_anchors import NativeAnchors

pytest.importorskip("diskgraph", reason="requires the optional native DiskGraph engine")

DAY = 86_400
SPLITS = ["train", "valid", "train", None, "train", "valid", "train", None]
# The latest train anchor is not the latest anchor overall, and not the last train row.
SECONDS = [5 * DAY, 9 * DAY, 3 * DAY, 10 * DAY, 7 * DAY + 1, 11 * DAY, 2 * DAY, 4 * DAY]
TIMES = {
    "int64": pa.array(SECONDS, type=pa.int64()),
    "int32": pa.array(SECONDS, type=pa.int32()),
    "date32": pa.array([s // DAY for s in SECONDS], type=pa.date32()),
    "timestamp-s-utc": pa.array(SECONDS, type=pa.timestamp("s", tz="UTC")),
    "timestamp-us-naive": pa.array([s * 10**6 + 250 for s in SECONDS], type=pa.timestamp("us")),
    "timestamp-ns-offset": pa.array(
        [s * 10**9 + 123 for s in SECONDS], type=pa.timestamp("ns", tz="+05:00")
    ),
    "timestamp-s-zone": pa.array(SECONDS, type=pa.timestamp("s", tz="America/New_York")),
}
SELECTIONS = [
    api.ALL_ANCHORS,
    frozenset({"train"}),
    frozenset({"valid", None}),
    frozenset({None}),
]


def _latest(requests):
    # Sub-microsecond times arrive as Arrow scalars, which Python cannot order.
    return max((request.as_of_time for request in requests), key=cutoff_nanos)


def _plan(tmp_path, times, *, batch_rows=None):
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("string")}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": api.DType("int64"),
                    "user": api.DType("string"),
                    "time": api.DType("timestamp", unit="s", timezone="UTC"),
                    "amount": api.DType("float64"),
                },
                time_column="time",
            ),
        ),
        relations=(api.RelationSpec("activity", "users", "id", "events", "user"),),
    )
    users = tmp_path / "users.parquet"
    events = tmp_path / "events.parquet"
    pq.write_table(pa.table({"id": ["u1", "u2"]}), users)
    pq.write_table(
        pa.table(
            {
                "id": list(range(6)),
                "user": ["u1", "u2"] * 3,
                "time": pa.array([DAY * i for i in range(6)], type=pa.timestamp("s", tz="UTC")),
                "amount": [float(i) for i in range(6)],
            }
        ),
        events,
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        neighborhoods={
            "graph": api.NeighborhoodSpec(
                (api.GraphLayer((api.GraphSample("activity", 2, strategy="recent"),)),)
            )
        },
        targets={"label": api.ExternalTarget(dtype=api.DType("float64"))},
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("graph"),
                    # ZScore fits from DiskGraph column statistics, which need the cutoff.
                    api.Graph(
                        node_fields={"events": api.Fields((api.Encode("amount", api.ZScore()),))}
                    ),
                    None,
                )
            },
            targets={"label": api.LabelTarget("label", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )
    rows = pa.table(
        {
            "sid": [f"s{i}" for i in range(len(SPLITS))],
            "entity": ["u1", "u2"] * (len(SPLITS) // 2),
            "cutoff": times,
            "label": [float(i) for i in range(len(SPLITS))],
            "split": SPLITS,
        }
    )
    anchors_path = tmp_path / "anchors.parquet"
    # Small row groups put one split's anchors in several Parquet batches.
    pq.write_table(rows, anchors_path, row_group_size=batch_rows or len(SPLITS))
    table = api.AnchorTable(
        api.Parquet(str(anchors_path)),
        "entity",
        "cutoff",
        sample_id="sid",
        split="split",
        targets={"label": "label"},
    )
    dataset = api.Dataset(
        tables={"users": api.Parquet(str(users)), "events": api.Parquet(str(events))}
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    return contract, dataset, engine, table


@pytest.mark.parametrize("kind", sorted(TIMES))
def test_latest_time_matches_the_requests_fitting_reads(tmp_path, kind):
    contract, dataset, engine, table = _plan(tmp_path, TIMES[kind])
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            anchors = task._anchors
            assert isinstance(anchors, NativeAnchors)
            for selection in SELECTIONS:
                expected = _latest(anchors.fitting_requests(selection))
                actual = anchors.latest_time(selection)
                assert type(actual) is type(expected)
                assert actual == expected
                assert cutoff_nanos(actual) == cutoff_nanos(expected)
            fitted = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
            train = anchors.fitting_requests(frozenset({"train"}))
            assert fitted == cutoff_nanos(_latest(train))


def test_latest_time_spans_parquet_batches(tmp_path):
    contract, dataset, engine, table = _plan(tmp_path, TIMES["timestamp-ns-offset"], batch_rows=3)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            anchors = task._anchors
            for selection in SELECTIONS:
                expected = _latest(anchors.fitting_requests(selection))
                assert anchors.latest_time(selection) == expected


def test_the_default_cutoff_reads_no_anchor_requests(tmp_path, monkeypatch):
    contract, dataset, engine, table = _plan(tmp_path, TIMES["int64"])

    def refuse(*args, **kwargs):
        raise AssertionError("the default cutoff rebuilt per-anchor requests")

    monkeypatch.setattr(NativeAnchors, "fetch", refuse)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            cutoff = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
    assert cutoff == (7 * DAY + 1) * 10**9


def test_sub_microsecond_anchor_times_fit_at_full_precision(tmp_path):
    # Python cannot order the Arrow scalars these anchors carry, so ordering
    # requests by their times used to raise TypeError.
    contract, dataset, engine, table = _plan(tmp_path, TIMES["timestamp-ns-offset"])
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            cutoff = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
    assert cutoff == (7 * DAY + 1) * 10**9 + 123
