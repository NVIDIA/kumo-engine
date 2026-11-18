# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph cutoff policy applies before fanout, independently at every hop."""

import pyarrow as pa
import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api, build


@pytest.mark.parametrize(
    "engine_name,unit,output_format",
    [("in_memory", "s", "inspectable"), ("diskgraph", "s", "inspectable")]
    + [("diskgraph", unit, "model") for unit in ("s", "ms", "us", "ns")],
)
@pytest.mark.parametrize(
    "cutoffs", [(None, None), ("inclusive",) * 2, ("exclusive",) * 2, ("inclusive", "exclusive")]
)
@pytest.mark.parametrize("fanout", [1, 10])
def test_graph_cutoff(engine_name, unit, output_format, cutoffs, fanout):
    dtype = api.DType("timestamp", unit=unit, timezone="UTC")
    # Adjacent ticks at a modern epoch catch both precision loss and rounding.
    base = 1_700_000_000 * {"s": 1, "ms": 1000, "us": 10**6, "ns": 10**9}[unit] + 17
    rows = [{"id": 0, "parent": None, "time": base - 2}]
    for parent in (0, 1, 2, 3):
        rows.extend(
            {"id": parent * 10 + j, "parent": parent, "time": base + j - 2} for j in (1, 2, 3)
        )
    table = build.Table(
        "nodes",
        "id",
        {"id": api.DType("int64"), "parent": api.DType("int64"), "time": dtype},
        time="time",
    )
    link = build.Link("children", table.id, table.parent)
    traversals = [build.Hop(link, fanout, **({"cutoff": c} if c else {})).lower() for c in cutoffs]
    contract = api.DataLoaderSpec(
        api.DataSchema((table.lower(),), (link.lower(),)),
        api.SampleSpec(
            api.EntitySpec("nodes"),
            neighborhoods={
                "tree": api.NeighborhoodSpec(tuple(api.GraphLayer((t,)) for t in traversals))
            },
        ),
        api.BatchSpec({"graph": api.Input(api.NeighborhoodSource("tree"), api.Graph(), None)}),
    )
    contract = api.DataLoaderSpec.from_json(contract.to_json())
    requests = [
        api.SampleRequest(
            str(i), 0, base + i if unit == "s" else pa.scalar(base + i, type=dtype.to_arrow())
        )
        for i in (0, 1)
    ]
    with (
        bound_tables(contract, {"nodes": rows}, engine_name) as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(requests) as task,
    ):
        (batch,) = task.batches(batch_size=2, output_format=output_format)
    graph = batch.inputs["graph"]
    nodes = graph["nodes"]["nodes"]
    ids, owners = nodes["ids"].tolist(), nodes["batch"].tolist()
    edges = graph["edges"]["children"]["index"].tolist()
    assert list(batch.metadata["sample_ids"]) == ["0", "1"]
    assert batch.metadata["as_of_time"].tolist() == [base, base + 1]
    for owner in (0, 1):
        expected_nodes, expected_edges, frontier = {0}, set(), [0]
        for cutoff in cutoffs:
            next_frontier = []
            for parent in frontier:
                eligible = [
                    row
                    for row in rows
                    if row["parent"] == parent
                    and (
                        row["time"] < base + owner
                        if cutoff == "exclusive"
                        else row["time"] <= base + owner
                    )
                ]
                for row in sorted(eligible, key=lambda r: r["time"], reverse=True)[:fanout]:
                    expected_nodes.add(row["id"])
                    expected_edges.add((parent, row["id"]))
                    next_frontier.append(row["id"])
            frontier = next_frontier
        assert {node for node, own in zip(ids, owners) if own == owner} == expected_nodes
        assert {(ids[a], ids[b]) for a, b in zip(*edges) if owners[a] == owner} == expected_edges
    assert all(owners[a] == owners[b] for a, b in zip(*edges))


def test_invalid_graph_cutoff():
    with pytest.raises(ValueError, match="graph cutoff must be"):
        api.GraphSample("children", 1, cutoff="invalid")
