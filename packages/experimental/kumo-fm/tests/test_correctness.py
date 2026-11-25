# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API tests against hand-computed relational and temporal results."""
import tempfile

import pytest
from _correctness import (
    assert_oracle,
    assert_wire_roundtrip,
    instant,
    make_case,
    prepare,
    sample_for,
)

from kumo import fm


@pytest.fixture(params=[False, True], ids=["int64-keys", "unicode-keys"])
def case(tmp_path, request):
    return make_case(tmp_path, string_keys=request.param)


@pytest.mark.parametrize("parent_first", [False, True], ids=["fk-left", "pk-left"])
@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_temporal_oracle_and_batch_boundaries(case, parent_first, batch_size):
    engine, keys, events = case
    # Alias/reorient the same stored relation through the public schema.
    from kumo.pytorch import api

    schema = api.DataSchema(engine.schema.tables, (
        api.RelationSpec("activity", "users", "id", "events", "user") if parent_first else
        api.RelationSpec("activity", "events", "user", "users", "id"),))
    context = [(keys[1], 10, 2**53 + 1), (keys[0], 10, -7),
               (keys[1], 10, 0), (keys[-1], 10, 2**63 - 1)]
    query = [(keys[-1], 30, -7), (keys[0], 30, 2**53 + 1), (keys[1], 30, 0)]
    with prepare(engine, schema=schema, sample=sample_for(parent_first=parent_first),
                 batch_size=batch_size) as loader:
        batch = loader.load(context=fm.Context(pairs=[(k, instant(t), i) for k, t, i in context]),
                            query=fm.Query(pairs=[(k, instant(t), i) for k, t, i in query]))
    assert_oracle(batch, events, {"context": context, "predict": query})
    assert_wire_roundtrip(batch)


def test_binary_labels_query_only_and_closed_requests(case):
    engine, keys, events = case
    context = [(keys[0], 10, 91), (keys[-1], 10, 92)]
    query = [(keys[0], 20, 91)]  # Exact end of context label horizon is observable.
    with prepare(engine, sample=sample_for(classification=True), task="binary_classification",
                 classes={False: "no", True: "yes"}) as loader:
        with loader.materialize(
                context=fm.Context(pairs=[(k, instant(t), i) for k, t, i in context]),
                query=fm.Query(pairs=[(k, instant(t), i) for k, t, i in query])) as request:
            batch = request.load()
        with pytest.raises(RuntimeError, match="closed"):
            request.load()
        query_only = loader.load(query=fm.Query(entities=[keys[-1]], at=instant(30)))
        assert query_only.splits == ("predict",)
        with pytest.raises(ValueError, match="query cutoff"):
            loader.load(context=fm.Context(entities=[keys[0]], times=[instant(10)]))
        with pytest.raises(ValueError, match="not observable"):
            loader.load(context=fm.Context(entities=[keys[0]], times=[instant(11)]),
                        query=fm.Query(entities=[keys[0]], at=instant(20)))
    loader.close()
    with pytest.raises(RuntimeError, match="closed"):
        loader.materialize(query=fm.Query(entities=[keys[0]], at=instant(30)))
    assert_oracle(batch, events, {"context": context, "predict": query}, classification=True)
    assert_wire_roundtrip(batch)
    assert_wire_roundtrip(query_only)


def test_preparation_failure_releases_prepared_plan(case, monkeypatch):
    from kumo.pytorch import api

    engine, _, _ = case
    closed = []

    class Plan:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append(True)

    calls = []

    def fail_second(**_):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected graph preparation failure")
        return Plan()

    monkeypatch.setattr(api, "prepare", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        prepare(engine)
    assert closed == [True]


def test_repeated_load_does_not_retain_anchor_files(case, tmp_path, monkeypatch):
    engine, keys, _ = case
    directory = tmp_path / "system-temp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    with prepare(engine) as loader:
        for _ in range(3):
            loader.load(query=fm.Query(entities=[keys[0]], at=instant(30)))
            assert not list(directory.iterdir())


@pytest.mark.parametrize("parent_first", [False, True])
@pytest.mark.parametrize("to_children", [False, True])
def test_self_relations_preserve_declared_edges(tmp_path, parent_first, to_children):
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo.pytorch import api

    path = tmp_path / "employees.parquet"
    frame = pa.table({"id": [10, 20, 30], "manager": pa.array([None, 10, 10], pa.int64()),
                      "at": pa.array([instant(1)] * 3, pa.timestamp("us", tz="UTC"))})
    pq.write_table(frame, path)
    left, right = ("id", "manager") if parent_first else ("manager", "id")
    schema = api.DataSchema(tables=(api.TableSpec("employees", "id", {
        f.name: api.DType.from_arrow(f.type) for f in frame.schema}, time_column="at"),),
        relations=(api.RelationSpec("management", "employees", left, "employees", right),))
    direction = "forward" if parent_first == to_children else "reverse"
    sample = api.SampleSpec(entity=api.EntitySpec("employees"), targets={"future": api.Aggregate(
        api.Rows(path=("management",), time=api.TimeWindow("at", start=0, stop=10)), "count")},
        neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample(
            "management", fanout=3, direction=direction, strategy="recent"),)),))})
    contract = api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs={
        "graph": api.Input(api.NeighborhoodSource("history"), api.Graph(), None)}))
    producer = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=contract, dataset=api.Dataset({"employees": api.Parquet(str(path))}),
                     engine=producer):
        pass
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=producer.index, id="self"))
    with prepare(engine, sample=sample, features={"employees": []}, batch_size=1) as loader:
        batch = loader.load(query=fm.Query(pairs=[(10 if to_children else 20, instant(10), i)
                                                for i in (-7, 2**53 + 1)]))
    expected = [(10, None), (20, 10), (30, 10)] if to_children else [(10, None), (20, 10)]
    assert batch.tables("predict")["related_tables"]["employees"].to_pylist() == [
        {"instance_id": i, "id": key, "__fm_edge_management": parent}
        for i in (-7, 2**53 + 1) for key, parent in expected]
    assert_wire_roundtrip(batch)
