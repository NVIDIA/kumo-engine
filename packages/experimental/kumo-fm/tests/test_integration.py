# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained payloads over an Arrow-to-Parquet fixture, with explicit expected rows."""
from types import SimpleNamespace

import pytest
from _fixture import LARGE, build_index, when

from kumo import fm
from kumo.pytorch import api


def test_snapshot_batch_survives_close_without_resampling(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        targets={"next": api.Aggregate(api.Rows(
            path=("order_user",), time=api.TimeWindow("at", start=0, stop=7 * 86400)), "count")},
        neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample(
            "order_user", direction="reverse", strategy="recent", fanout=1),)),))})
    with fm.prepare(engine=engine, sample=sample, target="next", task="regression",
                    features={"users": [], "orders": ["at"]}) as loader:
        with loader.materialize(context=fm.Context(pairs=[(LARGE, when(10), 10), (-5, when(10), 11)]),
                                query=fm.Query(pairs=[(LARGE, when(25), 20), (-5, when(25), 21)])) as request:
            batch = request.load()
            loader._graphs_for = lambda *_: (_ for _ in ()).throw(AssertionError("resampled"))
            arrays, ipc = batch.encode("arrays"), batch.encode("arrow_ipc")
        with pytest.raises(RuntimeError, match="closed"):
            request.load()
    assert arrays == batch.encode("arrays") and ipc == batch.encode("arrow_ipc")
    assert batch.tables("context")["instance_table"]["__fm_target"].to_pylist() == [1.0, 0.0]
    assert batch.tables("context")["related_tables"]["orders"]["order_id"].to_pylist() == [12, 13]
    assert batch.tables("predict")["related_tables"]["orders"]["order_id"].to_pylist() == [14, 13]
    assert "__fm_target" not in batch.tables("predict")["instance_table"].column_names
    with pytest.raises(RuntimeError, match="closed"):
        loader.load(query=fm.Query(pairs=[(LARGE, when(25))]))


@pytest.mark.parametrize("operation", ["mean", "count"])
def test_context_targets_report_null_occurrences_without_dropping_rows(tmp_path, operation):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    target = api.Aggregate(api.Rows(path=("order_user",),
                                    time=api.TimeWindow("at", start=0, stop=7 * 86400)),
                           operation, "order_id" if operation == "mean" else None)
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"future": target},
                            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                                api.GraphSample("order_user", direction="reverse", strategy="recent", fanout=1),
                            )),))})
    context = fm.Context(pairs=[(-5, when(10), 202), (LARGE, when(10), 101)])
    query = fm.Query(pairs=[(LARGE, when(25), 909)])
    with fm.prepare(engine=engine, sample=sample, target="future", task="regression",
                    features={"users": [], "orders": []}, batch_size=1) as loader:
        if operation == "mean":
            with pytest.raises(ValueError) as error:
                loader.load(context=context, query=query)
            assert str(error.value) == (
                "FM context target 'future' is null for occurrence IDs [202]; "
                "context targets must be non-null for every occurrence")
        else:
            batch = loader.load(context=context, query=query)
            instance = batch.tables("context")["instance_table"]
            assert instance["instance_id"].to_pylist() == [202, 101]
            assert instance["__fm_target"].to_pylist() == [0.0, 1.0]


def test_binary_labels_stay_aligned_and_map_numeric_keys_across_batches(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    target = api.Compare(api.Aggregate(api.Rows(path=("order_user",),
                                                time=api.TimeWindow("at", start=0, stop=7 * 86400)),
                                       "count"), "greater_than", 0)
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                                api.GraphSample("order_user", direction="reverse", strategy="recent", fanout=1),
                            )),))})
    ids = [-9007199254740993, 9007199254740993, -9]
    context = fm.Context(pairs=[(-5, when(10), ids[0]), (LARGE, when(10), ids[1]), (-5, when(10), ids[2])])
    query = fm.Query(pairs=[(LARGE, when(25), 0)])
    with fm.prepare(engine=engine, sample=sample, target="next", task="binary_classification",
                    classes={0: False, 1: True}, features={"users": [], "orders": []},
                    batch_size=1) as loader:
        batch = loader.load(context=context, query=query)
    instance = batch.tables("context")["instance_table"]
    assert instance["instance_id"].to_pylist() == ids
    assert instance["__fm_target"].to_pylist() == ["False", "True", "False"]
    orders = batch.tables("context")["related_tables"]["orders"]
    assert orders["instance_id"].to_pylist() == ids
    assert orders["order_id"].to_pylist() == [13, 12, 13]
