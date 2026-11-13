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


@pytest.mark.parametrize("unit,tick", [("s", 10**9), ("ms", 10**6), ("us", 1000)])
@pytest.mark.parametrize("integer", [False, True])
def test_stored_array_values_round_up_before_safe_cast(unit, tick, integer):
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = DiskGraphSource.__new__(DiskGraphSource)
    source._clock, source._pa, source._pc = "nanoseconds", pa, pc
    values = pa.array(
        [tick + 1, -tick - 1, None], type=pa.int64() if integer else pa.timestamp("ns", tz="UTC")
    )
    actual = source._typed_array(values, api.DType("timestamp", unit=unit, timezone="UTC"))
    assert actual.cast(pa.int64()).to_pylist() == [2, -1, None]


@pytest.mark.parametrize("value", [1700000000000, -1700000000000])
@pytest.mark.parametrize("typed", [False, True])
def test_overflowing_cutoffs_are_refused_in_both_paths(value, typed):
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = DiskGraphSource.__new__(DiskGraphSource)
    source._clock, source._pa, source._pc = "milliseconds", pa, pc
    values = pa.array([value], type=pa.timestamp("s", tz="UTC") if typed else pa.int64())
    # An epoch-ms-looking integer still means seconds; on ms it fits exactly.
    assert source._cutoff(values[0] if typed else value) == value * 1000
    assert source._cutoff_array(values).to_pylist() == [value * 1000]
    source._clock = "nanoseconds"
    with pytest.raises((ValueError, pa.ArrowInvalid), match="overflow|bounds|range"):
        source._cutoff(values[0] if typed else value)
    with pytest.raises((ValueError, pa.ArrowInvalid), match="overflow|bounds|range"):
        source._cutoff_array(values)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_unpinned_graph_fitting_uses_the_source_clock(tmp_path, monkeypatch, unit):
    import datetime as dt

    factor = {"ms": 1000, "us": 10**6, "ns": 10**9}[unit]
    base, step = 1700000000 * factor, factor // 1000
    users, events = tmp_path / "users.parquet", tmp_path / "events.parquet"
    pq.write_table(pa.table({"id": [1]}), users)
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "owner": [1, 1],
                "time": pa.array([base + step, base + 5 * step], type=pa.timestamp(unit, tz="UTC")),
                "kind": ["seen", "future"],
            }
        ),
        events,
    )
    schema = api.DataSchema(
        (
            api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": api.DType("int64"),
                    "owner": api.DType("int64"),
                    "time": api.DType("timestamp", unit=unit, timezone="UTC"),
                    "kind": api.DType("string"),
                },
                time_column="time",
            ),
        ),
        (api.RelationSpec("activity", "users", "id", "events", "owner"),),
    )
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        neighborhoods={
            "graph": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("activity", 10),)),))
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            {
                "graph": api.Input(
                    api.NeighborhoodSource("graph"),
                    api.Graph(
                        node_fields={"events": api.Fields((api.Encode("kind", api.Vocabulary()),))}
                    ),
                    None,
                )
            }
        ),
    )
    data = api.Dataset({"users": api.Parquet(str(users)), "events": api.Parquet(str(events))})
    cutoff = dt.datetime.fromtimestamp(1700000000, tz=dt.UTC) + dt.timedelta(milliseconds=3)
    with api.prepare(contract, data, api.DiskGraphEngine(index=tmp_path / "index")) as plan:
        # Exercise the supported fallback for an encoder that is not pinned by column stats.
        monkeypatch.setattr(plan._require_open(), "column_stats", None)
        with plan.materialize([api.SampleRequest("one", 1, cutoff)]) as task:
            (encoder,) = task.fitted_state["encoders"].values()
            assert {value for value, _ in encoder["entries"]} == {"seen"}


def test_native_request_reconstruction_preserves_ns_without_pandas():
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    code = (
        "import sys, types\nk=types.ModuleType('kumo');k.__path__=["
        + repr(str(Path(api.__file__).parents[1]))
        + "];sys.modules['kumo']=k\n"
    )
    code += textwrap.dedent("""
        class WithoutPandas:
            def find_spec(self, name, path=None, target=None):
                if name == 'pandas' or name.startswith('pandas.'):
                    raise ModuleNotFoundError('pandas deliberately unavailable')
        sys.meta_path.insert(0, WithoutPandas())
        import pyarrow as pa
        from kumo.pytorch import api
        from kumo.pytorch.engines.native_anchors import NativeAnchors
        anchors = NativeAnchors.__new__(NativeAnchors)
        anchors._pa = pa
        anchors._aliases = {'entity': 'entity', 'at': 'at'}
        anchors._types = {'entity': pa.int64(), 'at': pa.timestamp('ns', tz='UTC')}
        anchors._ipc = set()
        anchors._table = api.AnchorTable(api.Parquet('unused'), 'entity', 'at')
        batch = pa.record_batch({'entity': [1], 'at': [1], 'anchor_position': [0]})
        requests = anchors._requests({'name': 'one', 'split': None}, {
            'raw_features': {'anchors/one': {'record_batches': [batch]}}
        })
        assert requests[0].as_of_time.value == 1
        assert 'pandas' not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


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


@pytest.mark.parametrize(
    "clock,unit",
    [
        ("seconds", "s"),
        ("milliseconds", "ms"),
        ("microseconds", "us"),
        ("nanoseconds", "ns"),
        ("hours", "s"),
        ("days", "s"),
    ],
)
@pytest.mark.parametrize("ns", [-1500000001, -500001, -1, 0, 1, 500001, 1500000001])
def test_cutoffs_preserve_causal_floor_in_scalar_and_batch_paths(clock, unit, ns):
    import diskgraph
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = DiskGraphSource.__new__(DiskGraphSource)
    source._clock, source._pa, source._pc = clock, pa, pc
    finer = pa.array([ns], type=pa.timestamp("ns", tz="UTC"))
    # Compute the golden without copying either Arrow's or DG's implementation.
    tick = {
        "seconds": 10**9,
        "milliseconds": 10**6,
        "microseconds": 1000,
        "nanoseconds": 1,
        "hours": 3600 * 10**9,
        "days": 86400 * 10**9,
    }[clock]
    expected = ns // tick
    assert diskgraph.convert_ticks(ns, "nanoseconds", clock, rounding="floor") == expected
    assert diskgraph.convert_ticks(ns, "nanoseconds", clock) == (abs(ns) // tick) * (
        -1 if ns < 0 else 1
    )
    assert source._cutoff(finer[0]) == expected
    assert source._cutoff_array(finer).to_pylist() == [expected]
    seconds = ns // 10**9
    assert source._cutoff(seconds) == seconds * 10**9 // tick
    assert source._cutoff_array(pa.array([seconds], type=pa.int64())).to_pylist() == [
        seconds * 10**9 // tick
    ]
    # Python datetimes cover the same rule without requiring pandas.
    datetime_ns = (abs(ns) // 1000) * (-1000 if ns < 0 else 1000)
    instant = pa.scalar(datetime_ns // 1000, type=pa.timestamp("us", tz="UTC")).as_py()
    expected_datetime = datetime_ns // tick
    assert source._cutoff(instant) == expected_datetime


@pytest.mark.parametrize("unit,tick", [("s", 10**9), ("ms", 10**6), ("us", 1000), ("ns", 1)])
@pytest.mark.parametrize("ns", [-1500000001, -500001, -1, 0, 1, 500001, 1500000001])
def test_stored_timestamps_round_up_consistently_without_pandas(unit, tick, ns, monkeypatch):
    import builtins

    from kumo.pytorch.encoding import json_value, typed_key, unjson_value

    original_import = builtins.__import__

    def without_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("timestamp normalization must not import pandas")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pandas)
    dtype = api.DType("timestamp", unit=unit, timezone="UTC")
    instant = pa.scalar(ns, type=pa.timestamp("ns", tz="UTC"))
    normalized = dtype.normalize(instant)
    expected = -(-ns // tick)
    assert pa.scalar(normalized, type=dtype.to_arrow()).cast(pa.int64()).as_py() == expected
    assert typed_key(dtype, normalized) == ("timestamp", expected)
    assert typed_key(dtype, dtype.normalize(expected)) == ("timestamp", expected)
    restored = unjson_value(json_value(normalized))
    assert typed_key(dtype, restored) == ("timestamp", expected)


def test_nanosecond_saved_state_without_pandas_in_fresh_process():
    import subprocess
    import sys
    import textwrap

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            class WithoutPandas:
                def find_spec(self, name, path=None, target=None):
                    if name == 'pandas' or name.startswith('pandas.'):
                        raise ModuleNotFoundError('pandas deliberately unavailable')
            sys.meta_path.insert(0, WithoutPandas())
            from kumo.pytorch.contract.dtypes import DType
            from kumo.pytorch.encoding import json_value, unjson_value, typed_key
            dtype = DType('timestamp', unit='ns', timezone='UTC')
            value = dtype.normalize(1)
            assert value.value == 1
            saved = json_value(value)
            assert typed_key(dtype, unjson_value(saved)) == ('timestamp', 1)
            assert 'pandas' not in sys.modules
        """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
