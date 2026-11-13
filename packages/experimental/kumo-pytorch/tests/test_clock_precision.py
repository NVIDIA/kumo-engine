# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent fractional-time goldens through scalar and native history paths."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api

pytest.importorskip("diskgraph")


@pytest.mark.parametrize(
    "unit,name",
    [
        ("ms", "milliseconds"),
        ("us", "microseconds"),
        ("ns", "nanoseconds"),
    ],
)
@pytest.mark.parametrize("step", [1, 250])
@pytest.mark.parametrize("physical_finer", [False, True])
def test_fractional_cutoffs_values_merge_and_deltas_survive_native_iteration(
    tmp_path, unit, name, step, monkeypatch, physical_finer
):
    base = 1700000000 * {"ms": 1000, "us": 1000000, "ns": 1000000000}[unit]
    sources = {}
    users = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": [1]}), users)
    sources["users"] = api.Parquet(str(users))
    for table, times, amounts in [
        ("events", [0, 2, 2, 4], [1.0, 2.0, 3.0, 99.0]),
        ("web", [1, 3], [9.0, 10.0]),
    ]:
        path = tmp_path / f"{table}.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": list(range(len(times))),
                    "user": [1] * len(times),
                    "time": pa.array(
                        [
                            (base + t * step)
                            * ({"ms": 10**6, "us": 1000, "ns": 1}[unit] if physical_finer else 1)
                            for t in times
                        ],
                        type=pa.timestamp("ns" if physical_finer else unit, tz="UTC"),
                    ),
                    "amount": amounts,
                }
            ),
            path,
        )
        sources[table] = api.Parquet(str(path))
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
            *(
                api.TableSpec(
                    table,
                    "id",
                    {
                        "id": api.DType("int64"),
                        "user": api.DType("int64"),
                        "time": api.DType("timestamp", unit=unit, timezone="UTC"),
                        "amount": api.DType("float64"),
                    },
                    time_column="time",
                )
                for table in ("events", "web")
            ),
        ),
        relations=tuple(
            api.RelationSpec(table, "users", "id", table, "user") for table in ("events", "web")
        ),
    )
    rows = {
        table: api.Rows((table,), time=api.TimeWindow("time", None, 0))
        for table in ("events", "web")
    }
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        entity_values={
            "sum": api.Aggregate(rows["events"], "sum", "amount", dtype=api.DType("float64")),
            "age": api.Aggregate(rows["events"], "recency", "time", dtype=api.DType("float64")),
        },
        events={
            table: api.EventSpec(path, {"amount": api.Field("amount"), "age": api.EventTimeDelta()})
            for table, path in rows.items()
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(("events", "web"), max_events=3, keep="last"),
                    api.Fields((api.Encode("amount", api.Raw()), api.Encode("age", api.Raw()))),
                    api.Padded(),
                ),
            }
        ),
    )
    path = tmp_path / "anchors.parquet"
    times = pa.array([base + step, base + 3 * step], type=pa.timestamp(unit, tz="UTC"))
    pq.write_table(pa.table({"id": [1, 1], "at": times, "sid": ["first", "second"]}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "id", "at", sample_id="sid")
    with api.prepare(
        contract, api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        assert (
            json.loads((tmp_path / "index/index_metadata.json").read_text())["timestamp_unit"]
            == name
        )
        with monkeypatch.context() as guard:
            guard.setattr(
                plan._require_open(),
                "_cutoff",
                lambda value: pytest.fail("native materializer reconstructed a scalar cutoff"),
            )
            values = plan._materialize_values(anchors, tmp_path / "values.parquet", batch_size=1)
        saved = pq.read_table(values.anchors.source.path)
        assert saved["_anchor_time"].equals(pa.chunked_array([times]))
        assert saved[values.features["sum"]].to_pylist() == [1.0, 6.0]
        age = step * {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
        assert saved[values.features["age"]].to_pylist() == pytest.approx([age, age])
        task = plan.materialize(values.anchors)
        with monkeypatch.context() as guard:
            guard.setattr(
                type(plan._require_open()),
                "_cutoff",
                lambda *args: pytest.fail("native iterator reconstructed a scalar cutoff"),
            )
            batch = next(task.batches(batch_size=2, output_format="model"))
        assert batch.metadata["as_of_time"].tolist() == [base + step, base + 3 * step]
        assert batch.metadata["timestamp_unit"] == name
        assert batch.inputs["history"]["amount"]["values"].tolist() == [
            [1.0, 9.0, 0.0],
            [2.0, 3.0, 10.0],
        ]
        assert batch.inputs["history"]["age"]["values"][0].tolist() == pytest.approx(
            [age, 0.0, 0.0]
        )
        assert batch.inputs["history"]["age"]["values"][1].tolist() == pytest.approx(
            [age, age, 0.0]
        )


@pytest.mark.parametrize(
    "clock,factor", [("milliseconds", 1000), ("microseconds", 10**6), ("nanoseconds", 10**9)]
)
def test_integer_anchor_times_keep_epoch_seconds_contract(clock, factor):
    import datetime as dt

    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = DiskGraphSource.__new__(DiskGraphSource)
    source._clock, source._pa, source._pc = clock, pa, pc
    seconds = 1700000000
    instant = dt.datetime.fromtimestamp(seconds, tz=dt.UTC)
    assert source._cutoff(seconds) == source._cutoff(instant) == seconds * factor
    assert source._cutoff_array(pa.array([seconds], type=pa.int64())).to_pylist() == [
        seconds * factor
    ]
    # Typed Arrow timestamps, unlike plain anchor integers, carry their unit.
    assert source._cutoff_array(
        pa.array(
            [seconds * factor],
            type=pa.timestamp(
                {"milliseconds": "ms", "microseconds": "us", "nanoseconds": "ns"}[clock]
            ),
        )
    ).to_pylist() == [seconds * factor]


@pytest.mark.parametrize("clock", ["milliseconds", "microseconds"])
def test_finer_anchor_times_are_refused_by_scalar_and_batch_paths(clock):
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = DiskGraphSource.__new__(DiskGraphSource)
    source._clock, source._pa, source._pc = clock, pa, pc
    finer = pa.array([1700000000000000001], type=pa.timestamp("ns", tz="UTC"))
    with pytest.raises(ValueError, match="anchor time is finer than the graph index clock"):
        source._cutoff(finer[0].as_py())
    with pytest.raises(pa.ArrowInvalid, match="would lose data"):
        source._cutoff_array(finer)
