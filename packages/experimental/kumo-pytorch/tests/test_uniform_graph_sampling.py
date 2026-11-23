# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph strategy defaults and seed propagation through real native batch paths."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api


@pytest.fixture
def graph_data(tmp_path):
    pytest.importorskip("diskgraph")
    integer = api.DType("int64")
    timestamp = api.DType("timestamp", unit="s", timezone="UTC")
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {"id": integer, "user": integer, "at": timestamp},
                time_column="at",
            ),
        ),
        relations=(api.RelationSpec("activity", "users", "id", "events", "user"),),
    )
    pq.write_table(pa.table({"id": [1, 2]}), tmp_path / "users.parquet")
    pq.write_table(
        pa.table(
            {
                "id": list(range(1, 41)),
                "user": [1] * 40,
                "at": pa.array([i * 10 for i in range(1, 41)], type=pa.timestamp("s", tz="UTC")),
            }
        ),
        tmp_path / "events.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "sid": ["early", "late"],
                "entity": [1, 1],
                "cutoff": pa.array([50, 100], type=pa.timestamp("s", tz="UTC")),
            }
        ),
        tmp_path / "anchors.parquet",
    )
    dataset = api.Dataset(
        {name: api.Parquet(str(tmp_path / f"{name}.parquet")) for name in ("users", "events")}
    )
    anchors = api.AnchorTable(
        api.Parquet(str(tmp_path / "anchors.parquet")), "entity", "cutoff", sample_id="sid"
    )
    requests = (api.SampleRequest("early", 1, 50), api.SampleRequest("late", 1, 100))
    return schema, dataset, anchors, requests


def contract(schema, strategy=None, fanout=5, names=("graph",)):
    return api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("users"),
            neighborhoods={
                name: api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("activity", fanout, strategy=strategy),)),)
                )
                for name in names
            },
        ),
        api.BatchSpec(
            batching=api.Batching(max_samples=2),
            inputs={
                name: api.Input(
                    api.NeighborhoodSource(name),
                    api.Graph(node_fields={"events": api.Fields((api.Encode("id", api.Raw()),))}),
                    None,
                )
                for name in names
            },
        ),
    )


def selections(task, requests=None, **options):
    result = {}
    for batch in task.batches(requests, **options):
        for name, graph in batch.inputs.items():
            if isinstance(graph, api.ColumnarGraph):
                nodes = graph.nodes["events"]
                keys, owners = nodes.ids.to_pylist(), nodes.owners.to_pylist()
            else:
                nodes = graph["nodes"]["events"]
                keys, owners = nodes["ids"].tolist(), nodes["batch"].tolist()
            for owner, sid in enumerate(batch.metadata["sample_ids"]):
                ids = [key for key, sample in zip(keys, owners) if sample == owner]
                assert len(ids) == len(set(ids))
                result[name, sid] = frozenset(ids)
    return result


@pytest.mark.parametrize("clock", [True, False])
@pytest.mark.parametrize("strategy", [None, "uniform", "recent"])
def test_native_strategy_semantics(graph_data, tmp_path, clock, strategy):
    schema, dataset, anchors, requests = graph_data
    if not clock:
        schema = replace(
            schema,
            tables=tuple(replace(table, time_column=api.NON_TEMPORAL) for table in schema.tables),
        )
        anchors = replace(anchors, as_of_time=None)
        requests = tuple(replace(request, as_of_time=None) for request in requests)
    spec = contract(schema, strategy)
    engine = api.DiskGraphEngine(num_threads=1)
    if not clock and strategy == "recent":
        with pytest.raises(api.ValidationError, match="no time column"):
            api.prepare(spec, dataset, engine)
        assert not engine.index.exists()
        return
    with api.prepare(spec, dataset, engine) as plan, plan.materialize(anchors) as task:
        for output_format in ("inspectable", "model"):
            native = selections(task, output_format=output_format)
            assert native == selections(task, requests, output_format=output_format)
            assert all(len(ids) == 5 for ids in native.values())
            if clock and strategy != "uniform":
                assert native["graph", "early"] == frozenset(range(1, 6))
                assert native["graph", "late"] == frozenset(range(6, 11))
            else:
                assert native["graph", "early"] == native["graph", "late"]
                assert max(native["graph", "early"]) > 10  # Real anchor dates are ignored.


@pytest.mark.parametrize("names", [("graph",), ("graph", "other")])
def test_seed_replays_direct_native_and_named_neighborhoods(
    graph_data, tmp_path, monkeypatch, names
):
    import diskgraph

    sampler_engine, runtime_index = diskgraph.SamplerEngine, diskgraph.RuntimeIndex
    sampler_seeds, runtime_seeds = [], []

    def sampler(*args, **kwargs):
        sampler_seeds.append(kwargs["random_seed"])
        return sampler_engine(*args, **kwargs)

    def runtime(*args, **kwargs):
        runtime_seeds.append(kwargs["rnd_seed"])
        return runtime_index(*args, **kwargs)

    monkeypatch.setattr(diskgraph, "SamplerEngine", sampler)
    monkeypatch.setattr(diskgraph, "RuntimeIndex", runtime)
    schema, dataset, anchors, requests = graph_data
    spec = contract(schema, "uniform", names=names)
    answers = []
    engine = api.DiskGraphEngine(num_threads=1)
    for seed in (7, 7, 19):
        engine.random_seed = seed
        with api.prepare(spec, dataset, engine) as plan, plan.materialize(anchors) as task:
            expected = selections(task, requests, batch_size=1)
            for output_format in ("inspectable", "model"):
                assert expected == selections(
                    task,
                    output_format=output_format,
                    batch_size=2,
                    seed=91,
                    epoch=3,
                    shuffle=True,
                )
            if len(names) == 1:
                assert expected == selections(task, output_format="columnar", batch_size=2)
            answers.append(expected)
        assert sampler_seeds and all(value == seed for value in sampler_seeds)
        assert runtime_seeds == [seed]
        sampler_seeds.clear()
        runtime_seeds.clear()
    assert answers[0] == answers[1]


def test_diskgraph_rejects_unresolved_strategy(graph_data, tmp_path):
    schema, dataset, _anchors, _requests = graph_data
    spec = contract(schema)
    with api.prepare(spec, dataset, api.DiskGraphEngine()) as plan:
        with pytest.raises(ValueError, match="resolve GraphSample.strategy"):
            plan._source._sampling_plan(spec.sample.neighborhoods["graph"])


@pytest.mark.parametrize("names", [("graph",), ("a", "b"), ("a", "b", "c")])
def test_inspectable_reuses_every_prefetched_neighborhood(graph_data, tmp_path, monkeypatch, names):
    schema, dataset, anchors, requests = graph_data
    with (
        api.prepare(
            contract(schema, "uniform", names=names),
            dataset,
            api.DiskGraphEngine(num_threads=1),
        ) as plan,
        plan.materialize(anchors) as task,
    ):
        expected = selections(task, requests)

        def forbidden(*args, **kwargs):
            raise AssertionError("prefetched graph fell back to standalone sampling")

        with monkeypatch.context() as patch:
            patch.setattr(task._assembler._source, "_root_ids", forbidden)
            assert selections(task, output_format="inspectable") == expected


@pytest.mark.parametrize("fanout", [40, 100])
def test_uniform_returns_all_neighbors_when_fanout_covers_degree(graph_data, tmp_path, fanout):
    schema, dataset, anchors, requests = graph_data
    with (
        api.prepare(
            contract(schema, "uniform", fanout),
            dataset,
            api.DiskGraphEngine(random_seed=None),
        ) as plan,
        plan.materialize(anchors) as task,
    ):
        assert all(ids == frozenset(range(1, 41)) for ids in selections(task, requests).values())


def test_uniform_empty_neighborhood(graph_data, tmp_path):
    schema, dataset, _anchors, _requests = graph_data
    requests = (api.SampleRequest("empty", 2, 50),)
    with (
        api.prepare(
            contract(schema, "uniform"), dataset, api.DiskGraphEngine()
        ) as plan,
        plan.materialize(requests) as task,
    ):
        assert selections(task, requests) == {("graph", "empty"): frozenset()}


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5])
def test_sampling_seed_is_a_native_u64(seed):
    with pytest.raises(ValueError, match="random_seed"):
        api.DiskGraphEngine(random_seed=seed)
