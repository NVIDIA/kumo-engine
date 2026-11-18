# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default encoder cutoff comes from anchor columns, not per-anchor requests."""

from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import cutoff_nanos
from kumo.pytorch.engines.native_anchors import NativeAnchors
from kumo.pytorch.engines.parquet_anchors import ParquetAnchors

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


def _plan(tmp_path, times, *, splits=SPLITS):
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
            "sid": [f"s{i}" for i in range(len(splits))],
            "entity": ["u1" if i % 2 == 0 else "u2" for i in range(len(splits))],
            "cutoff": times,
            "label": [float(i) for i in range(len(splits))],
            "split": splits,
        }
    )
    anchors_path = tmp_path / "anchors.parquet"
    pq.write_table(rows, anchors_path)
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
            assert isinstance(task._anchors, NativeAnchors)
            for anchors in (task._anchors, ParquetAnchors(table)):
                for selection in SELECTIONS:
                    fetched = anchors.fetch(selection, range(anchors.count(selection)))
                    expected = _latest(fetched)
                    actual = anchors.latest_time(selection)
                    assert type(actual) is type(expected)
                    assert actual == expected
                    assert cutoff_nanos(actual) == cutoff_nanos(expected)
            fitted = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
            train = task._anchors.fitting_requests(frozenset({"train"}))
            assert fitted == cutoff_nanos(_latest(train))


def test_latest_time_spans_parquet_batches(tmp_path):
    # NativeAnchors reads 32768 rows per batch, independently of row-group size.
    # Keep every split's maximum in the first batch, followed by smaller times.
    first = TIMES["timestamp-ns-offset"]
    times = pa.concat_arrays([first, pa.repeat(pa.scalar(DAY * 10**9 + 123, first.type), 32768)])
    contract, dataset, engine, table = _plan(tmp_path, times, splits=SPLITS * 4097)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            anchors = task._anchors
            assert all(len(stage["latest"]) > 1 for stage in anchors._stages)
            for selection in SELECTIONS:
                expected = _latest(anchors.fitting_requests(selection))
                assert anchors.latest_time(selection) == expected


@pytest.mark.parametrize("zones", [("UTC", "Etc/UTC"), ("Etc/UTC", "UTC")])
def test_latest_time_normalizes_timezone_aliases_across_files(tmp_path, zones):
    times = TIMES["timestamp-us-naive"].cast(pa.timestamp("us", tz=zones[0]))
    contract, dataset, engine, table = _plan(tmp_path, times)
    rows = pq.read_table(table.source.path)
    parts = tmp_path / "anchor-parts"
    parts.mkdir()
    for i, zone in enumerate(zones):
        part = rows.slice(i * 4, 4)
        cutoff = part["cutoff"].cast(pa.timestamp("us", tz=zone))
        part = part.set_column(part.schema.get_field_index("cutoff"), "cutoff", cutoff)
        pq.write_table(part, parts / f"part-{i}.parquet")
    table = replace(table, source=api.Parquet(str(parts)))
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            assert {
                value.type.tz for stage in task._anchors._stages for value in stage["latest"]
            } == set(zones)
            for anchors in (task._anchors, ParquetAnchors(table)):
                for selection in SELECTIONS:
                    fetched = anchors.fetch(selection, range(anchors.count(selection)))
                    expected = max(r.as_of_time for r in fetched)
                    actual = anchors.latest_time(selection)
                    assert type(actual) is type(expected)
                    assert actual == expected
                    assert str(actual.tzinfo) == zones[0]
            assert task.fitted_state.fit["inputs"]["statistics"]["cutoff"] == (
                (7 * DAY + 1) * 10**9 + 250_000
            )


@pytest.mark.parametrize("temporal", [False, True])
def test_parquet_latest_time_empty_and_unknown_selections(tmp_path, temporal):
    _, _, _, table = _plan(tmp_path, TIMES["int64"])
    if not temporal:
        table = replace(table, as_of_time=None)
        assert ParquetAnchors(table).latest_time() == 0
    with pytest.raises(ValueError, match="no anchors in split"):
        ParquetAnchors(table).latest_time(frozenset({"missing"}))
    # No unsplit rows in a nonempty file is an empty selection too.
    rows = pq.read_table(table.source.path).slice(0, 3)
    pq.write_table(rows, table.source.path)
    with pytest.raises(ValueError, match="no anchors selected"):
        ParquetAnchors(table).latest_time(frozenset({None}))
    pq.write_table(rows.slice(0, 0), table.source.path)
    with pytest.raises(ValueError, match="no anchors selected"):
        ParquetAnchors(table).latest_time()


@pytest.mark.parametrize("kind,extra", [("int64", 0), ("timestamp-ns-offset", 123)])
def test_the_default_cutoff_reads_no_anchor_requests(tmp_path, monkeypatch, kind, extra):
    contract, dataset, engine, table = _plan(tmp_path, TIMES[kind])

    def refuse(*args, **kwargs):
        raise AssertionError("the default cutoff rebuilt per-anchor requests")

    monkeypatch.setattr(NativeAnchors, "fetch", refuse)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            cutoff = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
    assert cutoff == (7 * DAY + 1) * 10**9 + extra


@pytest.mark.parametrize("kind", ["int64", "timestamp-ns-offset"])
@pytest.mark.parametrize("case", ["before", "equal", "after", "unbounded"])
def test_explicit_cutoffs_preserve_fitting_and_warnings(tmp_path, monkeypatch, caplog, kind, case):
    contract, dataset, engine, table = _plan(tmp_path, TIMES[kind])
    # Include a row after the latest train anchor, but before the explicit later
    # cutoff and the latest validation anchor. Clamping or using all splits is wrong.
    path = tmp_path / "events.parquet"
    events = pq.read_table(path)
    future = pa.Table.from_pylist(
        [{"id": 6, "user": "u1", "time": datetime(1970, 1, 10, tzinfo=UTC), "amount": 6.0}],
        schema=events.schema,
    )
    pq.write_table(pa.concat_tables([events, future]), path)
    extra = 123 if kind == "timestamp-ns-offset" else 0
    equal = "1970-01-08T00:00:01" + (".000000123" if extra else "") + "Z"
    cutoff, expected, mean, warns = {
        "before": (3 * DAY, 3 * DAY * 10**9, 1.5, False),
        "equal": (equal, (7 * DAY + 1) * 10**9 + extra, 2.5, False),
        "after": ("1970-01-11T00:00:00Z", 10 * DAY * 10**9, 3.0, True),
        "unbounded": (None, None, 3.0, True),
    }[case]

    def refuse(*args, **kwargs):
        raise AssertionError("an explicit cutoff rebuilt native anchor requests")

    monkeypatch.setattr(NativeAnchors, "fetch", refuse)
    caplog.set_level("WARNING", logger="kumo.pytorch.preparation")
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as task:
            native = task.fitted_state
            assert native.fit["inputs"]["statistics"]["cutoff"] == expected
            encoder = next(e for e in task.encoders.values() if isinstance(e.config, api.ZScore))
            assert encoder.mean == pytest.approx(mean)
        messages = [r.getMessage() for r in caplog.records if r.name == "kumo.pytorch.preparation"]
        assert len(messages) == int(warns)
        if warns:
            assert "after the latest fitting anchor 1970-01-08T00:00:01+00:00" in messages[0]
            if cutoff is None:
                assert "None (every row)" in messages[0]
        if kind == "int64":
            # RequestAnchors exercises the unchanged per-request fallback.
            requests = [
                api.SampleRequest(
                    f"s{i}",
                    "u1" if i % 2 == 0 else "u2",
                    seconds,
                    split=SPLITS[i],
                    targets=(api.TargetValue("label", float(i)),),
                )
                for i, seconds in enumerate(SECONDS)
            ]
            caplog.clear()
            with plan.materialize(requests, fit_split="train", encoder_cutoff=cutoff) as task:
                assert task.fitted_state.fingerprint == native.fingerprint
                assert task.fitted_state.fit["inputs"]["statistics"]["cutoff"] == expected
            assert [
                r.getMessage() for r in caplog.records if r.name == "kumo.pytorch.preparation"
            ] == messages
