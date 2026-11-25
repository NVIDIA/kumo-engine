# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-supplied positives have independent, ordered point-in-time graphs."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.test_scored_walks import star, walk_contract, write_parquet


def contract(schema):
    base = walk_contract(schema)
    positive = api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods=base.sample.neighborhoods)
    return replace(
        base,
        sample=replace(base.sample, candidate_sets={"positive": positive}),
        batch=replace(
            base.batch,
            inputs={
                **base.batch.inputs,
                "positive_graph": api.Input(
                    api.NeighborhoodSource("graph", candidates="positive"),
                    api.Graph({"pins": api.Fields((api.Encode("id", api.Raw()),))}),
                    None,
                ),
            },
        ),
    )


REQUESTS = (
    api.SampleRequest(
        "early", 4, 10, candidates={"positive": (api.CandidateRequest(1), api.CandidateRequest(4))}
    ),
    api.SampleRequest("empty", 2, 20),
    api.SampleRequest("late", 1, 20, candidates={"positive": (api.CandidateRequest(1),)}),
)


def parquet_inputs(schema, rows, requests, directory):
    dataset, anchors = write_parquet(schema, rows, requests, directory)
    path = directory / "anchors.parquet"
    values = [
        [{"pin": c.entity_id} for c in request.candidates.get("positive", ())]
        for request in requests
    ]
    candidates = pa.array(values, type=pa.list_(pa.struct([("pin", pa.int64())])))
    table = pq.read_table(path).append_column("positives", candidates)
    pq.write_table(table, path)
    return dataset, replace(
        anchors, candidates={"positive": api.CandidateColumn("positives", "pin")}
    )


def _list(array):
    return array.to_pylist() if isinstance(array, (pa.Array, pa.ChunkedArray)) else array.tolist()


def assert_graphs(batch):
    graph = batch.inputs["positive_graph"]
    if isinstance(graph, api.ColumnarGraph):
        offsets = graph.candidate_offsets
        nodes = graph.nodes["pins"]
        pin_ids, owners = _list(nodes.ids), _list(nodes.owners)
        assert _list(nodes.features["id"]) == pin_ids
        edges = graph.edges["walk.pins"]
        src, dst = map(_list, edges.index.columns)
        scores, ranks = _list(edges.score), _list(edges.rank)
    else:
        offsets = graph["candidate_offsets"]
        nodes = graph["nodes"]["pins"]
        pin_ids, owners = _list(nodes["ids"]), _list(nodes["batch"])
        assert _list(nodes["features"]["id"]["values"]) == pin_ids
        edges = graph["edges"]["walk.pins"]
        src, dst = edges["index"].tolist()
        scores, ranks = _list(edges["score"]), _list(edges["rank"])

    groups = {"early": (1, 4), "empty": (), "late": (1,)}
    sample_ids = batch.metadata["sample_ids"]
    expected_groups = tuple(groups[s] for s in sample_ids)
    assert batch.metadata["candidate_entity_ids"]["positive"] == expected_groups
    offsets = _list(offsets)
    assert offsets[0] == 0 and len(offsets) == len(expected_groups) + 1
    assert [b - a for a, b in zip(offsets, offsets[1:])] == list(map(len, expected_groups))
    roots = [pin for group in expected_groups for pin in group]
    assert set(owners) == set(range(len(roots)))
    for occurrence, root in enumerate(roots):
        assert root in [p for p, owner in zip(pin_ids, owners) if owner == occurrence]
    by_owner = {occurrence: {} for occurrence in range(len(roots))}
    for i, j, rank, score in zip(src, dst, ranks, scores, strict=True):
        occurrence = owners[i]
        assert occurrence == owners[j] and pin_ids[i] == roots[occurrence]
        assert rank not in by_owner[occurrence]
        by_owner[occurrence][rank] = (pin_ids[j], score)
    for position, sid in enumerate(sample_ids):
        if sid == "early":
            actual = by_owner[offsets[position]]
            assert set(actual) == {0}
            assert actual[0] == pytest.approx((2, 1 / 12), abs=0.0001)
            assert by_owner[offsets[position] + 1] == {}
        elif sid == "late":
            actual = by_owner[offsets[position]]
            assert set(actual) == {0, 1}
            assert actual[0] == pytest.approx((3, 1 / 12), abs=0.0001)
            assert actual[1] == pytest.approx((2, 1 / 24), abs=0.0001)


def assert_graph_profile(profile, candidates):
    assert profile["samples"] == candidates
    assert isinstance(profile["sampling_ns"], int)
    assert isinstance(profile["total_ns"], int)
    assert 0 <= profile["sampling_ns"] <= profile["total_ns"]


def test_walks_follow_each_candidate_owner_cutoff(tmp_path):
    schema, rows = star()
    declaration = contract(schema)
    declaration = api.DataLoaderSpec.from_json(declaration.to_json())
    dataset, anchors = parquet_inputs(schema, rows, REQUESTS, tmp_path)
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(declaration, dataset, engine) as plan, plan.materialize(anchors) as task:
        assert_graphs(next(task.request_batches(REQUESTS)))
        assert_graphs(next(task.request_batches([REQUESTS[1]])))
        for fmt in ("model", "columnar", "inspectable"):
            batches = list(task.batches(output_format=fmt))
            assert len(batches) == 1
            assert_graphs(batches[0])
            for batch in task.batches(batch_size=1, output_format=fmt):
                assert_graphs(batch)
                (group,) = batch.metadata["candidate_entity_ids"]["positive"]
                assert_graph_profile(plan._source.last_graph_profile, len(group))


def test_candidate_source_validates_its_own_scope_and_round_trips():
    schema, _ = star()
    original = contract(schema)
    declaration = replace(
        original,
        sample=replace(original.sample, neighborhoods={}),
        batch=replace(original.batch, inputs={"graph": original.batch.inputs["positive_graph"]}),
    )
    declaration.validate()
    assert api.DataLoaderSpec.from_json(declaration.to_json()) == declaration.resolve()
    for source, message in (
        (api.NeighborhoodSource("graph"), "unknown neighborhood"),
        (api.NeighborhoodSource("graph", "missing"), "unknown candidate set"),
    ):
        invalid = replace(
            declaration,
            batch=replace(
                declaration.batch, inputs={"graph": api.Input(source, api.Graph(), None)}
            ),
        )
        with pytest.raises(ValueError, match=message):
            invalid.validate()


def test_fit_and_pack_candidate_graphs_with_their_labels(tmp_path):
    schema, rows = star()
    base = contract(schema)
    positive = replace(
        base.sample.candidate_sets["positive"],
        values={"id": api.Field(column="id")},
        targets={"label": api.ExternalTarget(api.DType("int64"))},
    )
    graph = api.Input(
        api.NeighborhoodSource("graph", candidates="positive"),
        api.Graph({"pins": api.Fields((api.Encode("id", api.UseEncoding("shared")),))}),
        None,
    )
    values = api.Input(
        api.CandidateSource("positive"),
        api.Fields((api.Encode("id", api.UseEncoding("shared")),)),
        api.Packed(),
    )
    declaration = replace(
        base,
        sample=replace(base.sample, candidate_sets={"positive": positive}),
        batch=replace(
            base.batch,
            inputs={"positive_graph": graph, "values": values},
            encodings={"shared": api.ZScore()},
            targets={"label": api.CandidateTarget("values", "label", api.Raw())},
        ),
    )
    requests = (
        api.SampleRequest(
            "full",
            4,
            10,
            candidates={
                "positive": (
                    api.CandidateRequest(2, (api.TargetValue("label", 2),)),
                    api.CandidateRequest(1, (api.TargetValue("label", 1),)),
                )
            },
        ),
        api.SampleRequest("empty", 4, 10),
    )

    def check(batch):
        graph = batch.inputs["positive_graph"]
        nodes = graph["nodes"]["pins"]
        # At this cutoff both candidate Pins only see each other through board 10:
        # fitted mean 1.5 and population std 0.5, independent of anchor Pin 4.
        assert nodes["features"]["id"]["values"].tolist() == [
            (value - 1.5) / 0.5 for value in nodes["ids"].tolist()
        ]
        ids = [
            candidate
            for group in batch.metadata["candidate_entity_ids"]["positive"]
            for candidate in group
        ]
        assert ids == [2, 1]
        assert batch.inputs["values"]["id"]["values"].tolist() == [1, -1]
        assert batch.targets["label"]["values"].tolist() == ids
        assert graph["candidate_offsets"].tolist() == batch.targets["label"]["offsets"].tolist()

    dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    anchors = replace(
        anchors,
        candidates={"positive": api.CandidateColumn("positives", "pin", targets={"label": "pin"})},
    )
    engine = api.DiskGraphEngine(num_threads=1)
    with api.prepare(declaration, dataset, engine) as plan, plan.materialize(anchors) as task:
        check(next(task.request_batches(requests)))
        check(next(task.request_batches(requests, output_format="model")))
        for fmt in ("inspectable", "model"):
            check(next(task.batches(output_format=fmt)))


def test_diskgraph_candidate_walk_preserves_nanosecond_cutoffs(tmp_path):
    schema, rows = star()
    timestamp = api.DType("timestamp", unit="ns", timezone="UTC")
    schema = replace(
        schema,
        tables=tuple(
            replace(table, columns={**table.columns, "at": timestamp})
            if table.name == "membership"
            else table
            for table in schema.tables
        ),
    )
    # All facts and cutoffs are inside the same microsecond.
    dataset, anchors = parquet_inputs(schema, rows, REQUESTS, tmp_path)
    path = tmp_path / "anchors.parquet"
    table = pq.read_table(path)
    nanos = pa.array([r.as_of_time for r in REQUESTS], type=timestamp.to_arrow())
    pq.write_table(table.set_column(table.schema.get_field_index("at"), "at", nanos), path)
    requests = tuple(replace(r, as_of_time=at) for r, at in zip(REQUESTS, nanos, strict=True))
    with (
        api.prepare(contract(schema), dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        assert_graphs(next(task.request_batches(requests)))
        assert_graphs(next(task.batches(output_format="model")))
        assert_graphs(next(task.batches(output_format="columnar")))


def test_diskgraph_candidate_hops_use_candidate_table_and_parent_time(tmp_path):
    schema, rows = star()
    graph = api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("pin_links", 10),)),))
    positive = api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods={"links": graph})
    sample = api.SampleSpec(api.EntitySpec("boards"), candidate_sets={"positive": positive})
    graph_input = api.Input(
        api.NeighborhoodSource("links", candidates="positive"), api.Graph(), None
    )
    declaration = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(inputs={"links": graph_input}, batching=api.Batching(max_samples=2)),
    )
    requests = tuple(
        api.SampleRequest(sid, 10, cutoff, candidates={"positive": (api.CandidateRequest(3),)})
        for sid, cutoff in (("early", 10), ("late", 20))
    )
    dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    with (
        api.prepare(declaration, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        for batch in (
            next(task.request_batches(requests)),
            next(task.batches(output_format="columnar")),
        ):
            result = batch.inputs["links"]
            if isinstance(result, api.ColumnarGraph):
                memberships = result.nodes["membership"]
                ids, owners = _list(memberships.ids), _list(memberships.owners)
            else:
                memberships = result["nodes"]["membership"]
                ids, owners = _list(memberships["ids"]), _list(memberships["batch"])
            assert sorted(ids) == [3, 4]
            assert [batch.metadata["sample_ids"][i] for i in owners] == ["late", "late"]

    unsafe = replace(
        graph, layers=(api.GraphLayer((api.GraphSample("pin_links", 10, strategy="uniform"),)),)
    )
    unsafe_sample = replace(
        sample, candidate_sets={"positive": replace(positive, neighborhoods={"links": unsafe})}
    )
    with pytest.raises(ValueError, match="ignores the candidate anchor"):
        replace(declaration, sample=unsafe_sample).validate()


def test_diskgraph_sampled_candidate_hops_keep_native_counts_and_order(tmp_path, monkeypatch):
    schema, rows = star()
    forward = api.GraphSample("pin_links", 10)
    hops = (
        forward,
        api.GraphSample("board_links", 10, direction="reverse"),
        api.GraphSample("board_links", 10),
        api.GraphSample("pin_links", 10, direction="reverse"),
        forward,
    )
    graph = api.NeighborhoodSpec(tuple(api.GraphLayer((hop,)) for hop in hops))
    positive = api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods={"links": graph})
    sample = api.SampleSpec(api.EntitySpec("pins"), candidate_sets={"positive": positive})
    item = api.Input(
        api.NeighborhoodSource("links", "positive"), api.Graph(topology="sampled"), None
    )
    declaration = api.DataLoaderSpec(
        schema, sample, api.BatchSpec(inputs={"graph": item}, batching=api.Batching(max_samples=3))
    )
    requests = tuple(
        api.SampleRequest(sid, 1, cutoff, candidates={"positive": candidates})
        for sid, cutoff, candidates in (
            ("early", 10, (api.CandidateRequest(3),)),
            ("late", 20, (api.CandidateRequest(3),)),
            ("empty", 20, ()),
        )
    )
    monkeypatch.setenv("EDGE_BATCH_SIZE_ROWS", "1")
    dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    with (
        api.prepare(declaration, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        for batches in (
            task.request_batches(requests),
            task.batches(output_format="columnar"),
            task.batches(output_format="model"),
            task.request_batches([requests[-1]]),
        ):
            batch = next(batches)
            ids = batch.metadata["sample_ids"]
            owners = [sid for sid in ids if sid != "empty"]
            assert_graph_profile(plan._source.last_graph_profile, len(owners))
            late = int("late" in owners)
            graph = batch.inputs["graph"]
            if isinstance(graph, api.ColumnarGraph):
                node_counts, edge_counts = graph.num_sampled_nodes, graph.num_sampled_edges
                kinds = {k: (v.relation, v.direction) for k, v in graph.edges.items()}
                endpoints = {k: tuple(map(_list, v.index.columns)) for k, v in graph.edges.items()}
                memberships = _list(graph.nodes["membership"].owners)
            else:
                node_counts, edge_counts = graph["num_sampled_nodes"], graph["num_sampled_edges"]
                kinds = {k: (v["relation"], v["direction"]) for k, v in graph["edges"].items()}
                endpoints = {k: v["index"].tolist() for k, v in graph["edges"].items()}
                memberships = _list(graph["nodes"]["membership"]["batch"])
            assert tuple(node_counts["pins"]) == (len(owners), 0, 0, 0, 2 * late, 0)
            assert tuple(node_counts["membership"]) == (0, 2 * late, 0, 2 * late, 0, 0)
            assert [owners[i] for i in memberships] == ["late"] * (4 * late)
            expected = {
                ("pin_links", "forward"): (2, 0, 0, 0, 2),
                ("board_links", "reverse"): (0, 2, 0, 0, 0),
                ("board_links", "forward"): (0, 0, 4, 0, 0),
                ("pin_links", "reverse"): (0, 0, 0, 2, 0),
            }
            assert set(kinds.values()) == set(expected)
            for key, kind in kinds.items():
                counts = tuple(edge_counts[key])
                assert counts == tuple(count * late for count in expected[kind])
                assert len(endpoints[key][0]) == sum(counts)
                if late and kind == ("pin_links", "forward"):
                    assert endpoints[key][0][:2] == [owners.index("late")] * 2
                    assert len(set(endpoints[key][0][2:])) == 2
