# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored vectors survive root, neighbor and candidate sampling at each cutoff."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from tests.test_candidate_neighborhoods import REQUESTS, contract, parquet_inputs
from tests.test_scored_walks import star


@pytest.mark.parametrize("list_type", [pa.list_, pa.large_list])
def test_diskgraph_fixed_projection_validates_only_visible_rows(list_type):
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = object.__new__(DiskGraphSource)
    source._pa, source._pc = pa, pc
    dtype = api.DType("float32", shape=(2, 2))
    values = pa.array([[9], None, [1, 2, 3, 4]], type=list_type(pa.float32()))
    assert source._typed_array(values.slice(1), dtype).to_pylist() == [None, [1, 2, 3, 4]]
    with pytest.raises(ValueError, match="expected 4 elements"):
        source._typed_array(values, dtype)
    with pytest.raises(ValueError, match="array elements cannot be null"):
        source._typed_array(pa.array([[1, 2, None, 4]], type=list_type(pa.float32())), dtype)


@pytest.mark.parametrize("rows", [[], [[1, 2]]])
def test_diskgraph_fixed_projection_rejects_mismatched_fixed_width(rows):
    import pyarrow.compute as pc

    from kumo.pytorch.engines.diskgraph import DiskGraphSource

    source = object.__new__(DiskGraphSource)
    source._pa, source._pc = pa, pc
    values = pa.array(rows, type=pa.list_(pa.float32(), 2))
    with pytest.raises(ValueError, match="width|shape"):
        source._typed_array(values, api.DType("float32", shape=(2, 2)))


def vector(pin):
    return None if pin == 2 else [float(pin * 10 + j) for j in range(4)]


def fields(name):
    return api.Fields((api.Encode(name, api.Raw()),))


def fixture(
    tmp_path, *, entity=False, fixed=False, requests=REQUESTS, candidate_anchors=True, mixed=False
):
    scalar_schema, rows = star()
    # Write actual Arrow lists independently of DType.to_arrow().
    dataset, anchors = parquet_inputs(scalar_schema, rows, requests, tmp_path)
    if not candidate_anchors:
        anchors = replace(anchors, candidates={})
    path = tmp_path / "pins.parquet"
    pins = pq.read_table(path)
    values = [vector(pin) for pin in pins["id"].to_pylist()]
    arrow_type = pa.list_(pa.float32(), 4) if fixed else pa.list_(pa.float32())
    # One row group per Pin lets Parquet represent a null fixed-size list between valid rows.
    pq.write_table(
        pins.append_column("image", pa.array(values, type=arrow_type)), path, row_group_size=1
    )
    schema = replace(
        scalar_schema,
        tables=tuple(
            replace(t, columns={**t.columns, "image": api.DType("float32", shape=(2, 2))})
            if t.name == "pins"
            else t
            for t in scalar_schema.tables
        ),
    )
    original = contract(schema)
    if entity:
        values = {"image": api.Field(column="image")}
        targets = {}
        if mixed:
            values["history"] = api.Aggregate(
                api.Rows(("pin_links",), api.TimeWindow("at")), "count"
            )
            targets["future"] = api.Aggregate(
                api.Rows(("pin_links",), api.TimeWindow("at", start=0, stop=None)), "count"
            )
        representation = api.Fields(tuple(api.Encode(name, api.Raw()) for name in values))
        declaration = replace(
            original,
            sample=replace(
                original.sample,
                entity_values=values,
                targets=targets,
                candidate_sets=original.sample.candidate_sets if candidate_anchors else {},
            ),
            batch=replace(
                original.batch,
                inputs={"root": api.Input(api.EntitySource(), representation, api.Separate())},
                targets={name: api.LabelTarget(name, api.Raw()) for name in targets},
            ),
        )
    else:
        inputs = {
            name: replace(input_, representation=api.Graph({"pins": fields("image")}))
            for name, input_ in original.batch.inputs.items()
        }
        if mixed:
            original = replace(
                original,
                sample=replace(original.sample, entity_values={"key": api.Field(column="id")}),
            )
            inputs["root"] = api.Input(api.EntitySource(), fields("key"), api.Separate())
        declaration = replace(original, batch=replace(original.batch, inputs=inputs))
    return api.prepare(declaration, dataset, api.DiskGraphEngine(num_threads=1)), anchors


def check_tensors(payload, ids):
    values, valid = payload["values"], payload["valid"]
    assert values.shape == (len(ids), 2, 2)
    assert values.dtype == torch.float32
    assert valid.dtype == torch.bool and valid.shape == (len(ids),)
    expected = [vector(i) for i in ids]
    assert valid.tolist() == [row is not None for row in expected]
    assert values.reshape(len(ids), 4).tolist() == [row or [0.0] * 4 for row in expected]


@pytest.mark.parametrize("fixed", [False, True])
def test_root_neighbor_and_candidate_arrays(tmp_path, fixed):
    plan, anchors = fixture(tmp_path, fixed=fixed)
    with plan, plan.materialize(anchors) as task:
        batches = [next(task.request_batches(REQUESTS))]
        batches.extend(next(task.batches(output_format=fmt)) for fmt in ("model", "columnar"))
        for batch in batches:
            for graph in (batch.inputs["graph"], batch.inputs["positive_graph"]):
                if isinstance(graph, api.ColumnarGraph):
                    nodes = graph.nodes["pins"]
                    ids = nodes.ids.to_pylist()
                    col = nodes.features["image"]
                    assert col.type == pa.list_(pa.float32(), 4)
                    assert col.to_pylist() == [vector(i) for i in ids]
                else:
                    pins = graph["nodes"]["pins"]
                    ids = pins["ids"].tolist()
                    check_tensors(pins["features"]["image"], ids)
                assert {1, 2, 3, 4} <= set(ids)
        empty = next(task.request_batches([REQUESTS[1]])).inputs["positive_graph"]
        check_tensors(empty["nodes"]["pins"]["features"]["image"], [])


def test_graph_vectors_reused_across_iterators_and_owned_by_each_model_batch(tmp_path, monkeypatch):
    plan, anchors = fixture(tmp_path, mixed=True)
    with plan, plan.materialize(anchors) as task:
        source = task._assembler._source
        fetch = source._read_feature_arrays
        stage_iter = task._anchors._iter
        projections, fetches = [], []

        def traced_iter(*args, **kwargs):
            projections.append(kwargs["projections"]["pins"])
            return stage_iter(*args, **kwargs)

        def traced_fetch(table, ids, names):
            if table == "pins" and "image" in names:
                fetches.append(tuple(ids))
            return fetch(table, ids, names)

        monkeypatch.setattr(task._anchors, "_iter", traced_iter)
        monkeypatch.setattr(source, "_read_feature_arrays", traced_fetch)
        for epoch in range(2):
            batch = next(task.batches(output_format="model", epoch=epoch, shuffle=False))
            assert projections[-1] == ["id"]  # root entity values still use native projection
            assert (
                batch.inputs["root"]["key"]["values"].tolist()
                == batch.metadata["entity_ids"].tolist()
            )
            for name in ("graph", "positive_graph"):
                nodes = batch.inputs[name]["nodes"]["pins"]
                ids = nodes["ids"].tolist()
                assert len(set(ids)) < len(ids)
                feature = nodes["features"]["image"]
                check_tensors(feature, ids)
                feature["values"].fill_(-99)
                feature["valid"].logical_not_()
            if epoch == 0:
                assert fetches and all(len(set(ids)) == len(ids) for ids in fetches)
                fetches.clear()
        assert not fetches


def test_deferred_candidate_fields_project_only_final_occurrences(tmp_path, monkeypatch):
    from kumo.pytorch.engines.diskgraph import _graph_node_ids

    plan, anchors = fixture(tmp_path)
    with plan, plan.materialize(anchors) as task:
        source = task._assembler._source
        bindings = task._assembler._prepared_inputs["graph"].state["pins"]
        dense_ids = np.array([3, 1, 2, 0], dtype=np.uint64)
        names = ("id", "image")
        expected_ids = _graph_node_ids(source._read_feature_arrays("pins", dense_ids, names)["id"])
        fetch = source._read_feature_arrays
        reads = []

        def traced_fetch(table, ids, fields):
            reads.append((table, tuple(ids), tuple(fields)))
            return fetch(table, ids, fields)

        monkeypatch.setattr(source, "_read_feature_arrays", traced_fetch)
        monkeypatch.setattr(
            source,
            "_runtime_batch",
            {
                "_graph_feature_cache": task._anchors._graph_feature_cache,
                "raw_features": {},
            },
            raising=False,
        )
        project = source._deferred_graph_fields("pins", names, bindings)
        assert project is not None and not reads
        indices = np.array([0, 2, 0, 1, 3, 1], dtype=np.int64)
        for positions in (indices, indices[::-1], indices[:0]):
            ids, features = project(dense_ids[positions])
            assert ids.tolist() == expected_ids.numpy()[positions].tolist()
            check_tensors(features["image"], ids.tolist())
            features["image"]["values"].fill_(99)
            features["image"]["valid"].logical_not_()
            ids.fill_(-1)
        assert reads == [("pins", (0, 1, 2, 3), names)]
        # A remapped binding cannot use the direct raw vector representation.
        assert (
            source._deferred_graph_fields(
                "pins",
                names,
                {"image": replace(bindings["image"], destination=object())},
            )
            is None
        )


@pytest.mark.parametrize("candidate_anchors", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_entity_arrays(tmp_path, monkeypatch, candidate_anchors, mixed):
    # Pin 3 has two future memberships at the early cutoff. Pin 2's array is NULL.
    requests = [replace(REQUESTS[0], entity_id=3), *REQUESTS[1:]]
    if not candidate_anchors:
        requests = [replace(request, candidates={}) for request in requests]
    plan, anchors = fixture(
        tmp_path, entity=True, requests=requests, candidate_anchors=candidate_anchors, mixed=mixed
    )
    with plan, plan.materialize(anchors) as task:
        batches = [next(task.request_batches(requests))]

        def no_fallback(*args, **kwargs):
            pytest.fail("native batching reconstructed requests")

        monkeypatch.setattr(task._anchors, "_requests", no_fallback)
        monkeypatch.setattr(task._assembler, "batches", no_fallback)
        batches.append(next(task.batches(output_format="model")))
        for batch in batches:
            ids = batch.metadata["entity_ids"].tolist()
            check_tensors(batch.inputs["root"]["image"], ids)
            if mixed:
                counts = {1: (1, 0), 2: (1, 0), 3: (0, 2)}
                assert batch.inputs["root"]["history"]["values"].tolist() == [
                    counts[i][0] for i in ids
                ]
                assert batch.targets["future"]["values"].tolist() == [counts[i][1] for i in ids]
            if candidate_anchors:
                groups = {
                    r.sample_id: tuple(c.entity_id for c in r.candidates.get("positive", ()))
                    for r in requests
                }
                assert batch.metadata["candidate_entity_ids"]["positive"] == tuple(
                    groups[s] for s in batch.metadata["sample_ids"]
                )


@pytest.mark.parametrize(
    "bad,reason",
    [([1.0], "length|width|size|elements"), ([1.0, None, 3.0, 4.0], "null|NULL")],
)
def test_malformed_non_null_array_fails_even_outside_the_requested_batch(tmp_path, bad, reason):
    import diskgraph

    schema, rows = star()
    dataset, _ = parquet_inputs(schema, rows, REQUESTS, tmp_path)
    path = tmp_path / "pins.parquet"
    table = pq.read_table(path)
    # The bad row is in the last row group, and it is never requested.
    values = [vector(1), None, [0.0] * 4, bad]
    image = pa.array(values, type=pa.list_(pa.float32()))
    pq.write_table(table.append_column("image", image), path, row_group_size=1)
    pins = next(t for t in schema.tables if t.name == "pins")
    schema = replace(
        schema,
        tables=(
            replace(pins, columns={**pins.columns, "image": api.DType("float32", shape=(4,))}),
        ),
        relations=(),
    )
    declaration = api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec("pins"), entity_values={"image": api.Field(column="image")}),
        api.BatchSpec(
            inputs={"root": api.Input(api.EntitySource(), fields("image"), api.Separate())}
        ),
    )
    with pytest.raises(diskgraph.DiskgraphSourceValidationError, match=reason):
        with api.prepare(
            declaration,
            api.Dataset({"pins": dataset.tables["pins"]}),
            api.DiskGraphEngine(num_threads=1),
        ) as plan:
            with plan.materialize((api.SampleRequest("first", 1, 10),)) as task:
                next(task.request_batches((api.SampleRequest("first", 1, 10),)))
