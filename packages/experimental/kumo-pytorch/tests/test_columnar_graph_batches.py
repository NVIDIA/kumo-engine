# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columnar graphs preserve selected occurrences and bypass tensor assembly."""

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors
from tests.test_inspectable_graph_batches import _equal
from tests.test_inspectable_graph_batches import graph_task as graph_task


def _compare(actual, expected):
    _equal(actual.metadata, expected.metadata)
    _equal(actual.targets, expected.targets)
    graph = actual.inputs["graph"]
    old = expected.inputs["graph"]
    assert isinstance(graph, api.ColumnarGraph)
    assert graph.root_type == "users"
    assert graph.nodes.keys() == old["nodes"].keys()
    assert graph.edges.keys() == old["edges"].keys()
    for name, node in graph.nodes.items():
        previous = old["nodes"][name]
        ids = previous["ids"]
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids.to_pylist()
        assert node.ids.to_pylist() == ids
        assert node.owners.type in (pa.uint32(), pa.uint64())
        assert node.owners.to_pylist() == previous["batch"].tolist()
        assert node.features.column_names == list(previous["features"])
        for field in node.features.column_names:
            values = previous["features"][field]["values"]
            if isinstance(values, torch.Tensor):
                values = pa.array(values.numpy())
            valid = pa.array(previous["features"][field]["valid"].numpy())
            restored = pc.if_else(valid, values, None)
            assert node.features[field].combine_chunks().equals(restored)
    for name, edge in graph.edges.items():
        previous = old["edges"][name]
        assert edge.source_type == previous["source_type"]
        assert edge.destination_type == previous["destination_type"]
        pairs = set(zip(edge.index["source"].to_pylist(), edge.index["destination"].to_pylist()))
        assert pairs == set(map(tuple, previous["index"].numpy().T))
        for left, right in pairs:
            assert (
                graph.nodes[edge.source_type].owners[left].as_py()
                == graph.nodes[edge.destination_type].owners[right].as_py()
            )


@pytest.mark.parametrize("world_size", [1, 3, 7])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_columnar_parity_and_lifetime(graph_task, monkeypatch, world_size, shuffle, drop_last):
    task = graph_task
    options = dict(world_size=world_size, shuffle=shuffle, drop_last=drop_last, seed=5)
    expected = [list(task.batches(rank=rank, **options)) for rank in range(world_size)]

    def forbidden(*args, **kwargs):
        raise AssertionError("columnar output rebuilt requests or encoded a public graph")

    with monkeypatch.context() as patch:
        patch.setattr(NativeAnchors, "_requests", forbidden)
        patch.setattr(task._assembler._source, "_graph_output", forbidden)
        patch.setattr(task._assembler._source, "build_model_batch", forbidden)
        actual = [
            list(task.batches(rank=rank, output_format="columnar", **options))
            for rank in range(world_size)
        ]
    # Comparing after all iterators finish also checks native buffer lifetime.
    for batches, old in zip(actual, expected, strict=True):
        assert len(batches) == len(old)
        for batch, previous in zip(batches, old, strict=True):
            _compare(batch, previous)
            _compare(batch.to("cpu"), previous)


def test_columnar_reuses_native_buffers(graph_task, monkeypatch):
    source = graph_task._assembler._source
    original = source.columnar_graph_batch
    captured = []

    def capture(sample, spec, runtime):
        captured.append(runtime)
        return original(sample, spec, runtime)

    monkeypatch.setattr(source, "columnar_graph_batch", capture)
    batches = list(graph_task.batches(output_format="columnar"))
    for batch, runtime in zip(batches, captured, strict=True):
        graph = batch.inputs["graph"]
        native_edge_buffers = {
            values.buffers()[1].address
            for edge in runtime["graphs"]["main"]["edge_index_dict"]
            for values in (edge["src"], edge["dst"])
            if len(values)
        }
        for edge in graph.edges.values():
            for column in edge.index.columns:
                for chunk in column.chunks:
                    if len(chunk):
                        assert chunk.buffers()[1].address in native_edge_buffers
        for name, node in graph.nodes.items():
            if not len(node.ids):
                continue
            assert (
                node.owners.buffers()[1].address
                == runtime["graphs"]["main"]["batch_dict"][name].buffers()[1].address
            )
            if name == "events":
                raw = pa.Table.from_batches(runtime["raw_features"][name]["record_batches"])
                for field in ("note",):
                    assert (
                        node.features[field].chunk(0).buffers()[1].address
                        == raw[field].chunk(0).buffers()[1].address
                    )


@pytest.mark.parametrize("graph_task", ["mixed", "availability"], indirect=True)
def test_unsupported_columnar_contract_fails_before_iteration(graph_task, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported contract opened native iterator")

    monkeypatch.setattr(NativeAnchors, "_iter", forbidden)
    with pytest.raises(ValueError, match="columnar output requires"):
        graph_task.batches(output_format="columnar")


def test_columnar_rejects_requests(graph_task):
    with pytest.raises(ValueError, match="materialized DiskGraph anchors"):
        graph_task.batches(requests=[], output_format="columnar")


def test_columnar_resume_and_closed_task_buffers(graph_task):
    task = graph_task
    first = task.batches(output_format="columnar", shuffle=True, seed=9)
    next(first)
    state = first.state_dict()
    expected = list(first)
    resumed = task.batches(output_format="columnar", shuffle=True, seed=9)
    resumed.load_state_dict(state)
    actual = list(resumed)
    assert [b.metadata["sample_ids"] for b in actual] == [
        b.metadata["sample_ids"] for b in expected
    ]
    snapshots = [b.inputs["graph"].nodes["events"].features.to_pylist() for b in actual]
    task.close()
    assert [b.inputs["graph"].nodes["events"].features.to_pylist() for b in actual] == snapshots


@pytest.mark.parametrize("pk_left", [True, False])
@pytest.mark.parametrize("directions", [("forward",), ("reverse",), ("forward", "reverse")])
def test_columnar_self_relations_follow_declared_endpoints(tmp_path, pk_left, directions):
    import pyarrow.parquet as pq

    from conformance.assertions import bound_tables

    rows = [
        {"id": 1, "parent_id": None, "at": 1},
        {"id": 2, "parent_id": 1, "at": 10},
        {"id": 3, "parent_id": 2, "at": 20},
    ]
    left_key, right_key = ("id", "parent_id") if pk_left else ("parent_id", "id")
    contract = api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "accounts",
                    "id",
                    {
                        "id": api.DType("int64"),
                        "parent_id": api.DType("int64"),
                        "at": api.DType("timestamp", unit="s", timezone="UTC"),
                    },
                    time_column="at",
                ),
            ),
            (api.RelationSpec("parent", "accounts", left_key, "accounts", right_key),),
        ),
        api.SampleSpec(
            api.EntitySpec("accounts"),
            neighborhoods={
                "family": api.NeighborhoodSpec(
                    tuple(
                        api.GraphLayer(
                            tuple(
                                api.GraphSample("parent", 4, direction=d, strategy="recent")
                                for d in directions
                            )
                        )
                        for _ in range(2)
                    )
                )
            },
        ),
        api.BatchSpec(
            {
                "graph": api.Input(
                    api.NeighborhoodSource("family"),
                    api.Graph({"accounts": api.Fields((api.Encode("parent_id", api.Raw()),))}),
                    None,
                )
            }
        ),
    )
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table({"id": [2, 2], "cutoff": pa.array([100, 5], type=pa.timestamp("s", tz="UTC"))}),
        anchors,
    )
    with (
        bound_tables(contract, {"accounts": rows}, "diskgraph") as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(api.AnchorTable(api.Parquet(str(anchors)), "id", "cutoff")) as task,
    ):
        old = next(iter(task.batches(batch_size=2)))
        graph = next(iter(task.batches(batch_size=2, output_format="columnar"))).inputs["graph"]
    node = graph.nodes["accounts"]
    edge = graph.edges["parent"]
    assert node.ids.to_pylist()[:2] == [2, 2]
    pairs = list(zip(edge.index["source"].to_pylist(), edge.index["destination"].to_pylist()))
    assert set(pairs) == set(map(tuple, old.inputs["graph"]["edges"]["parent"]["index"].numpy().T))
    by_id = {row["id"]: row for row in rows}
    for a, b in pairs:
        assert by_id[node.ids[a].as_py()][left_key] == by_id[node.ids[b].as_py()][right_key]
        assert node.owners[a].as_py() == node.owners[b].as_py()


@pytest.mark.parametrize("unsupported", ["encoding", "tokens"])
def test_columnar_refuses_non_raw_and_token_budget(graph_task, monkeypatch, unsupported):
    from dataclasses import replace

    assembler = graph_task._assembler
    if unsupported == "encoding":
        prepared = assembler._prepared_inputs["graph"]
        graph = prepared.spec.representation
        fields = dict(graph.node_fields)
        fields["events"] = api.Fields((api.Encode("note", api.Vocabulary()),))
        monkeypatch.setitem(
            assembler._prepared_inputs,
            "graph",
            replace(
                prepared,
                spec=replace(prepared.spec, representation=replace(graph, node_fields=fields)),
            ),
        )
    else:
        monkeypatch.setattr(
            assembler,
            "spec",
            replace(
                assembler.spec,
                batching=api.Batching(max_samples=2, max_tokens=10, length_input="graph"),
            ),
        )
    with pytest.raises(ValueError, match="columnar"):
        graph_task.batches(output_format="columnar")
