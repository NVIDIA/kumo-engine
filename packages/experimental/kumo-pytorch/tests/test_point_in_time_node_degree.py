# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check every sampled occurrence against the source rows at its owning anchor."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.node_values import DegreeCache
from tests.test_candidate_neighborhoods import parquet_inputs
from tests.test_engine_drawn_candidates import ISOLATED, POSITIVE, QUERY, as_list, data, future
from tests.test_scored_walks import walk_contract, write_parquet


def count(relation, *, start=None):
    return api.Aggregate(
        api.Rows((relation,), time=api.TimeWindow("at", start=start, stop=0)), "count"
    )


def test_exact_degree_cache_reuses_only_matching_nodes_clocks_and_programs():
    cache = DegreeCache()
    calls = []
    name = "entity:degree"

    def read(ids, ticks, program=1):
        def fetch(nodes, cutoffs):
            pairs = list(zip(nodes.tolist(), cutoffs.tolist(), strict=True))
            calls.append(pairs)
            # Keep this reference wide enough to catch unsigned ID or signed tick loss.
            counts = [((i ^ (t % 2**64)) + program) % 2**64 for i, t in pairs]
            return {name: pa.array(counts, type=pa.uint64())}

        result = cache.read(
            program, (name,), np.array(ids, np.uint64), np.array(ticks, np.int64), fetch
        )
        assert result[name].to_pylist() == [
            ((i ^ (t % 2**64)) + program) % 2**64 for i, t in zip(ids, ticks, strict=True)
        ]

    high = 2**63 + 77
    read([high, high], [-2, -2])
    read([high], [-2])
    assert calls == [[(high, -2)]]
    read([high, high], [-2, -1])
    assert calls[-1] == [(high, -1)]
    read([high], [-2], program=2)
    assert calls[-1] == [(high, -2)]
    # Same clock, distinct native IDs forced into one slot. Mixed hits must
    # retain the hit's count while admitting a colliding miss.
    read([0, 65536, 0], [0, 0, 0])
    read([65536, 0], [0, 0])
    read([0, 65536], [0, 0])


def node_columns(graph, table):
    if isinstance(graph, api.ColumnarGraph):
        node = graph.nodes[table]
        return (
            as_list(node.ids),
            as_list(node.owners),
            {name: as_list(node.features[name]) for name in node.features.column_names},
        )
    node = graph["nodes"][table]
    return (
        node["ids"].tolist(),
        node["batch"].tolist(),
        {name: item["values"].tolist() for name, item in node["features"].items()},
    )


@pytest.mark.parametrize("topology", ["relations", "sampled"])
def test_each_query_and_candidate_occurrence_uses_its_parent_cutoff(tmp_path, topology):
    schema, rows = data()
    # More eligible edges for one Pin than a single-hop fanout can expose.
    rows["membership"].append(dict(id=99, pin=POSITIVE, board=10, at=21))
    requests = (
        api.SampleRequest(
            "early",
            QUERY,
            10,
            candidates={
                "positive": (api.CandidateRequest(POSITIVE), api.CandidateRequest(ISOLATED))
            },
        ),
        api.SampleRequest("empty", ISOLATED, 10),
        api.SampleRequest(
            "late", QUERY, 20, candidates={"positive": (api.CandidateRequest(POSITIVE),)}
        ),
    )
    query = walk_contract(schema).sample.neighborhoods["graph"]
    if topology == "sampled":
        query = api.NeighborhoodSpec(
            (
                api.GraphLayer((api.GraphSample("pin_links", 1),)),
                api.GraphLayer((api.GraphSample("board_links", 1, direction="reverse"),)),
            )
        )

    def with_values(hood, start):
        return replace(
            hood,
            node_values={
                table: {"degree": count(relation, start=start)}
                for table, relation in (("pins", "pin_links"), ("boards", "board_links"))
            },
        )

    query = with_values(query, None)
    positive = api.CandidateSetSpec(
        api.EntitySpec("pins"),
        neighborhoods={"graph": with_values(query, -10)},
    )
    sets = {"positive": positive, "future": replace(positive, draw=future())}
    fields = {name: api.Fields((api.Encode("degree", api.Raw()),)) for name in ("pins", "boards")}
    rep = api.Graph(fields, topology=topology)
    spec = api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec("pins"), neighborhoods={"graph": query}, candidate_sets=sets),
        api.BatchSpec(
            inputs={
                name: api.Input(api.NeighborhoodSource("graph", scope), rep, None)
                for name, scope in (("query", None), ("positive", "positive"), ("future", "future"))
            },
            batching=api.Batching(max_samples=3),
        ),
    )
    dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    cutoffs = {r.sample_id: r.as_of_time for r in requests}

    def verify(batch):
        parent_times = [cutoffs[s] for s in batch.metadata["sample_ids"]]
        for name, graph in batch.inputs.items():
            if name == "query":
                times = parent_times
            else:
                groups = batch.metadata["candidate_entity_ids"][name]
                times = [at for at, group in zip(parent_times, groups, strict=True) for _ in group]
            for table, key in (("pins", "pin"), ("boards", "board")):
                ids, owners, features = node_columns(graph, table)
                expected = [
                    sum(
                        row[key] == node_id
                        and row["at"] <= times[owner]
                        and (name == "query" or row["at"] > times[owner] - 10)
                        for row in rows["membership"]
                    )
                    for node_id, owner in zip(ids, owners, strict=True)
                ]
                assert features["degree"] == expected
                if isinstance(graph, api.ColumnarGraph):
                    assert graph.nodes[table].features["degree"].type == pa.int64()
                else:
                    feature = graph["nodes"][table]["features"]["degree"]
                    assert feature["values"].dtype == torch.long
                    assert feature["valid"].all().item()

    with (
        api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        if topology == "relations":
            for fmt in ("inspectable", "model"):
                verify(next(task.request_batches(tuple(reversed(requests)), output_format=fmt)))
        with patch.object(
            task._anchors, "_requests", side_effect=AssertionError("native anchor fallback")
        ):
            for fmt in ("columnar", "model", "inspectable"):
                verify(next(task.batches(output_format=fmt)))
                for batch in task.batches(batch_size=1, shuffle=True, seed=41, output_format=fmt):
                    verify(batch)


def test_nanosecond_degree_on_pins_reached_through_history(tmp_path):
    schema, rows = data()
    nanos = api.DType("timestamp", unit="ns", timezone="UTC")
    schema = replace(
        schema,
        tables=tuple(
            replace(t, columns={**t.columns, "at": nanos}) if "at" in t.columns else t
            for t in schema.tables
        ),
    )
    t0 = 10_000_000_000
    for row in rows["membership"]:
        row["at"] = t0 if row["at"] == 10 else t0 + 50
    reqs = tuple(api.SampleRequest(sid, 10, 10) for sid in ("before", "exact", "after"))
    ds, anchors = write_parquet(schema, rows, reqs, tmp_path)
    instants = pa.array([t0 + 49, t0 + 50, t0 + 51], type=nanos.to_arrow())
    path = tmp_path / "anchors.parquet"
    parquet = pq.read_table(path)
    pq.write_table(parquet.set_column(parquet.schema.get_field_index("at"), "at", instants), path)
    reqs = tuple(replace(r, as_of_time=t) for r, t in zip(reqs, instants, strict=True))
    base = walk_contract(schema, after_hops=True)
    values = {
        "degree": count("pin_links"),
        "twice": api.Derived("add", ("degree", "degree")),
        "id_alias": api.Field("id"),
    }
    hood = replace(base.sample.neighborhoods["graph"], node_values={"pins": values})
    # Only the derived value is selected. The degree remains its dependency.
    fields = api.Fields(tuple(api.Encode(k, api.Raw()) for k in ("twice", "id_alias")))
    graph_input = api.Input(api.NeighborhoodSource("graph"), api.Graph({"pins": fields}), None)
    spec = replace(
        base,
        sample=replace(base.sample, neighborhoods={"graph": hood}),
        batch=replace(base.batch, inputs={"graph": graph_input}),
    )
    with (
        api.prepare(spec, ds, api.DiskGraphEngine(num_threads=2)) as plan,
        plan.materialize(anchors) as task,
    ):
        for stream in (
            task.request_batches(reqs),
            task.batches(output_format="columnar"),
            task.batches(output_format="model"),
        ):
            batch = next(stream)
            ids, owners, columns = node_columns(batch.inputs["graph"], "pins")
            assert set(columns) == {"id_alias", "twice"} and columns["id_alias"] == ids
            expected = [
                2
                * sum(
                    r["pin"] == pin and r["at"] <= instants[owner].value for r in rows["membership"]
                )
                for pin, owner in zip(ids, owners, strict=True)
            ]
            assert columns["twice"] == expected
            assert [value for pin, value in zip(ids, expected, strict=True) if pin == POSITIVE] == [
                4,
                6,
                6,
            ]
