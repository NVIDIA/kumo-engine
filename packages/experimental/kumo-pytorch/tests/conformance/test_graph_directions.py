# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traversal direction and emitted orientation through cycles and self-loops."""

from types import SimpleNamespace

import pytest

import kumo.pytorch.api as r
from tests import graph_semantics
from tests.conformance.fixtures import FLOAT, INT, TIME, batches, fields, materialized, record


def graph_fixture(mode, depth, pk_left):
    rows = [
        {"id": i, "parent": parent, "at": at, "balance": None if i == 6 else -1.25 * i}
        for i, parent, at in [
            (1, 3, 10),
            (2, 1, 20),
            (3, 2, 30),
            (4, 4, 40),
            (5, 1, 101),
            (6, None, 50),
            (7, 1, 100),
            (8, 99, 5),
        ]
    ]
    relation = r.RelationSpec(
        "parent",
        "accounts",
        "id" if pk_left else "parent",
        "accounts",
        "parent" if pk_left else "id",
    )
    directions = ("forward", "reverse") if mode == "both" else (mode,)
    layers = tuple(
        r.GraphLayer(
            tuple(r.GraphSample("parent", 4, direction=direction) for direction in directions)
        )
        for _ in range(depth)
    )
    schema = r.DataSchema(
        (
            r.TableSpec(
                "accounts",
                "id",
                {"id": INT, "parent": INT, "at": TIME, "balance": FLOAT},
                time_column="at",
            ),
        ),
        (relation,),
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("accounts"),
            neighborhoods={"family": r.NeighborhoodSpec(layers)},
        ),
        r.BatchSpec(
            {
                "graph": r.Input(
                    r.NeighborhoodSource("family"),
                    r.Graph({"accounts": fields("balance")}),
                    None,
                )
            },
            batching=r.Batching(max_samples=3),
        ),
    )
    records = [
        record("cycle", 1),
        record("at-boundary", 1, at=20),
        record("self-loop", 4),
        record("isolated", 6),
        record("dangling", 8),
        record("later", 1, at=101),
    ]
    return contract, {"accounts": rows}, records


def expected_graph(rows, relation, root, cutoff, directions, depth):
    """Tiny exhaustive join; fanout four includes every eligible neighbor here."""
    by_id = {row["id"]: row for row in rows}
    logical = {
        (left["id"], right["id"])
        for left in rows
        for right in rows
        if left[relation.left_key] is not None
        and left[relation.left_key] == right[relation.right_key]
    }
    nodes, edges, frontier = {root}, set(), {root}
    for _ in range(depth):
        following = set()
        for node in frontier:
            for direction in directions:
                forward = direction == "forward"
                candidates = {
                    (a, b)
                    for a, b in logical
                    if (a if forward else b) == node and by_id[b if forward else a]["at"] <= cutoff
                }
                assert len(candidates) <= 4, "Fixture now needs a fanout-selection oracle"
                edges.update(candidates)
                following.update(b if forward else a for a, b in candidates)
        nodes.update(following)
        frontier = following
    return nodes, edges


@pytest.mark.parametrize("pk_left", [True, False], ids=["pk-to-fk", "fk-to-pk"])
@pytest.mark.parametrize("depth", [1, 3], ids=["one-hop", "cycle-revisit"])
@pytest.mark.parametrize("mode", ["forward", "reverse", "both"])
def test_directed_traversal_and_bidirectional_expansion(tmp_path, pk_left, depth, mode):
    contract, rows, records = graph_fixture(mode, depth, pk_left)
    expected = {row["sample"]: row for row in records}
    relation = contract.schema.relations[0]
    directions = ("forward", "reverse") if mode == "both" else (mode,)
    with materialized(contract, rows, records, tmp_path, seed=17) as (task, _, _):
        oracle = graph_semantics
        source = oracle.SourceRows(contract.schema, rows)
        delivered = []
        for batch in batches(task):
            identities = list(batch.metadata["sample_ids"])
            requests = [
                SimpleNamespace(
                    sample_id=s,
                    entity_id=expected[s]["entity"],
                    as_of_time=expected[s]["at"],
                )
                for s in identities
            ]
            graph = batch.inputs["graph"]
            oracle.assert_graph(
                graph, contract, "graph", requests, source, task.fitted_state.to_dict()
            )
            nodes = graph["nodes"]["accounts"]
            ids, owners = nodes["ids"].tolist(), nodes["batch"].tolist()
            edge = graph["edges"]["parent"]
            assert edge["source_type"] == edge["destination_type"] == "accounts"
            edge_indices = edge["index"].t().tolist()
            assert all(owners[a] == owners[b] for a, b in edge_indices)
            for owner, identity in enumerate(identities):
                anchor = expected[identity]
                wanted_nodes, wanted_edges = expected_graph(
                    rows["accounts"],
                    relation,
                    anchor["entity"],
                    anchor["at"],
                    directions,
                    depth,
                )
                actual_nodes = [
                    key for key, group in zip(ids, owners, strict=True) if group == owner
                ]
                actual_edges = [(ids[a], ids[b]) for a, b in edge_indices if owners[a] == owner]
                assert (
                    len(actual_nodes) == len(set(actual_nodes))
                    and set(actual_nodes) == wanted_nodes
                ), identity
                assert (
                    len(actual_edges) == len(set(actual_edges))
                    and set(actual_edges) == wanted_edges
                ), identity
            delivered.extend(identities)
        assert sorted(delivered) == sorted(expected)
