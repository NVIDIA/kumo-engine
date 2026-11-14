# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspectable native graphs retain Raw values without reconstructing requests."""

import sys
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors, _column_values


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "UTC", "America/New_York", "+05:30"])
def test_timestamp_columns_preserve_values_and_resolve_timezone_once(monkeypatch, unit, tz):
    scale = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}[unit]
    dtype = pa.timestamp(unit, tz=tz)
    # Both occurrences of the repeated DST hour, fractional precision, pre-epoch, NULL.
    values = pa.chunked_array(
        [[1699162200 * scale + 1, None], [1699165800 * scale + 1, -1]], type=dtype
    )
    try:
        expected = values.to_pylist()
    except ValueError as error:
        # Preserve Arrow's refusal of sub-microsecond values without pandas.
        with pytest.raises(ValueError, match="Nanosecond resolution"):
            _column_values(pa, values, dtype)
        assert "Nanosecond resolution" in str(error)
        return
    attempts = []

    class MissingPytz:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "pytz":
                attempts.append(fullname)
                raise ModuleNotFoundError("No module named 'pytz'", name="pytz")
            return None

    monkeypatch.delitem(sys.modules, "pytz", raising=False)
    monkeypatch.setattr(sys, "meta_path", [MissingPytz(), *sys.meta_path])
    actual = _column_values(pa, values, dtype)
    assert len(attempts) <= 1  # independent of row count, including missing optional pytz
    assert [None if x is None else x.isoformat() for x in actual] == [
        None if x is None else x.isoformat() for x in expected
    ]
    assert _column_values(pa, pa.array([], type=dtype), dtype) == []


def _equal(actual, expected):
    if isinstance(expected, (pa.Array, pa.ChunkedArray)):
        assert actual.equals(expected)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _equal(actual[name], expected[name])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _equal(left, right)
    else:
        assert actual == expected


def test_empty_timestamp_column_does_not_resolve_timezone():
    dtype = pa.timestamp("us", tz="not-a-timezone")
    for values in (pa.array([], type=dtype), pa.array([None, None], type=dtype)):
        assert _column_values(pa, values, dtype) == values.to_pylist()


@pytest.fixture
def graph_task(tmp_path, request):
    pytest.importorskip("diskgraph")
    variant = getattr(request, "param", "single")
    ids = [2**53 + 1, 2**53 + 3, 2**53 + 5]
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "users",
                "id",
                {
                    "id": api.DType("int64"),
                    "created": api.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": api.DType("int64"),
                    "user": api.DType("int64"),
                    "time": api.DType("timestamp", unit="s", timezone="UTC"),
                    "note": api.DType("string"),
                    "day": api.DType("date32"),
                },
                time_column="time",
            ),
        ),
        relations=(api.RelationSpec("activity", "users", "id", "events", "user"),),
    )
    users, events, anchors = (
        tmp_path / f"{name}.parquet" for name in ("users", "events", "anchors")
    )
    pq.write_table(
        pa.table({"id": ids, "created": pa.array([0, 0, 0], type=pa.timestamp("s", tz="UTC"))}),
        users,
    )
    pq.write_table(
        pa.table(
            {
                "id": [10, 20, 30],
                "user": [ids[0], ids[0], ids[1]],
                "time": pa.array([10, 20, 30], type=pa.timestamp("s", tz="UTC")),
                "note": ["first", None, "last"],
                "day": [date(2026, 1, 1), None, date(2026, 1, 3)],
            }
        ),
        events,
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec(
            "users", available_from="created" if variant == "availability" else None
        ),
        neighborhoods={
            "graph": api.NeighborhoodSpec(
                (api.GraphLayer((api.GraphSample("activity", 2, strategy="recent"),)),)
            )
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("graph"),
                    api.Graph(
                        node_fields={
                            "events": api.Fields(
                                tuple(
                                    api.Encode(name, api.Raw()) for name in ("note", "day", "time")
                                )
                            ),
                        }
                    ),
                    None,
                )
            },
            batching=api.Batching(max_samples=2),
        ),
    )
    pq.write_table(
        pa.table(
            {
                "sid": [0, 1, 2, 3, 4],
                "entity": [ids[0], ids[1], ids[0], ids[2], ids[0]],
                "cutoff": pa.array(
                    [25000001, 35000001, 15000001, 35000001, 5000001],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "split": ["a", "b", "a", "b", "a"] if variant == "mixed" else ["a"] * 5,
                "metadata": [{"parts": [i, i + 1], "note": None} for i in range(5)],
                "local_time": pa.array(
                    [1699162200000001, None, 1699165800000001, -1, 0],
                    type=pa.timestamp("us", tz="America/New_York"),
                ),
            }
        ),
        anchors,
    )
    table = api.AnchorTable(
        api.Parquet(str(anchors)),
        "entity",
        "cutoff",
        sample_id="sid",
        split="split",
        metadata=("metadata", "local_time"),
    )
    dataset = api.Dataset(
        tables={"users": api.Parquet(str(users)), "events": api.Parquet(str(events))}
    )
    with api.prepare(
        contract, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        with plan.materialize(table, fit_split="a") as task:
            yield task


@pytest.mark.parametrize("world_size", [1, 3, 7])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_inspectable_graph_uses_native_buffers_with_exact_parity(
    graph_task, monkeypatch, world_size, shuffle, drop_last
):
    task = graph_task
    options = dict(
        world_size=world_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=5,
        output_format="inspectable",
    )
    with monkeypatch.context() as patch:
        patch.setattr(NativeAnchors, "_can_batch_inspectable_graph", lambda *args: False)
        expected = [list(task.batches(rank=rank, **options)) for rank in range(world_size)]

    def forbidden(*args, **kwargs):
        raise AssertionError("inspectable graph rebuilt per-anchor requests or samples")

    with monkeypatch.context() as patch:
        patch.setattr(NativeAnchors, "_requests", forbidden)
        patch.setattr(task._assembler._source, "build", forbidden)
        actual = [list(task.batches(rank=rank, **options)) for rank in range(world_size)]
    for rank_batches, expected_batches in zip(actual, expected, strict=True):
        assert len(rank_batches) == len(expected_batches)
        for left, right in zip(rank_batches, expected_batches, strict=True):
            _equal(left.inputs, right.inputs)
            _equal(left.targets, right.targets)
            _equal(left.metadata, right.metadata)
    with pytest.raises(ValueError, match="Raw.*requires inspectable"):
        list(task.batches(output_format="model"))


@pytest.mark.parametrize("graph_task", ["mixed", "availability"], indirect=True)
def test_other_inspectable_contracts_keep_existing_path(graph_task, monkeypatch):
    task = graph_task
    with monkeypatch.context() as patch:
        patch.setattr(NativeAnchors, "_can_batch_inspectable_graph", lambda *args: False)
        expected = list(task.batches())

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported inspectable contract entered typed path")

    monkeypatch.setattr(NativeAnchors, "_typed_batches", forbidden)
    actual = list(task.batches())
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        _equal(left.inputs, right.inputs)
        _equal(left.metadata, right.metadata)
