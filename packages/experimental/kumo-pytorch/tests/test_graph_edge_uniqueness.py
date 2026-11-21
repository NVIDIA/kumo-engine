# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G-GRAPH-001 rule 3: an edge appears at most once per sample."""

from collections import Counter
from dataclasses import replace

import pytest

from conformance.assertions import bound_tables
from conformance.graph_cases import generate_graph_case, graph_edge_uniqueness, graph_structure
from kumo.pytorch import api

USERS = [{"user_id": 1, "at": 0}, {"user_id": 2, "at": 0}]
EVENTS = [
    {"event_id": 10, "user_id": 1, "at": 5},
    {"event_id": 11, "user_id": 1, "at": 7},
    {"event_id": 12, "user_id": 1, "at": 9},
    {"event_id": 20, "user_id": 2, "at": 6},
]


def _contract(first_fanout) -> api.DataLoaderSpec:
    def hop(direction: str, fanout=4) -> api.GraphLayer:
        return api.GraphLayer(
            (api.GraphSample("user_events", fanout, direction=direction, strategy="recent"),)
        )

    timestamp = api.DType("timestamp", unit="s", timezone="UTC")
    return api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "users",
                    "user_id",
                    {"user_id": api.DType("int64"), "at": timestamp},
                    time_column="at",
                ),
                api.TableSpec(
                    "events",
                    "event_id",
                    {
                        "event_id": api.DType("int64"),
                        "user_id": api.DType("int64"),
                        "at": timestamp,
                    },
                    time_column="at",
                ),
            ),
            (api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
        ),
        # Layers 2 and 3 reach every layer-1 edge again, once from each end.
        api.SampleSpec(
            api.EntitySpec("users"),
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (hop("forward", first_fanout), hop("reverse"), hop("forward"))
                )
            },
        ),
        api.BatchSpec({"graph": api.Input(api.NeighborhoodSource("history"), api.Graph(), None)}),
    )


@pytest.mark.parametrize(
    "engine_name,first_fanout", [("in_memory", 4), ("diskgraph", 4), ("diskgraph", 1)]
)
def test_revisited_edges_appear_once_per_sample(engine_name, first_fanout):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = api.DataLoaderSpec.from_json(_contract(first_fanout).to_json())
    # Two samples of user 1 share the stored edges; each keeps its own copy.
    requests = [
        api.SampleRequest("early", 1, 6),
        api.SampleRequest("late", 1, 100),
        api.SampleRequest("other", 2, 100),
    ]
    expected = {
        "early": [(1, 10)],
        "late": [(1, 10), (1, 11), (1, 12)],
        "other": [(2, 20)],
    }
    with (
        bound_tables(contract, {"users": USERS, "events": EVENTS}, engine_name) as (
            dataset,
            engine,
        ),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(requests) as task,
    ):
        batches = list(
            task.batches(
                batch_size=len(requests),
                output_format="model" if engine_name == "diskgraph" else "inspectable",
            )
        )
    assert len(batches) == 1
    batch = batches[0]
    assert list(batch.metadata["sample_ids"]) == [request.sample_id for request in requests]
    graph = batch.inputs["graph"]
    users, events = graph["nodes"]["users"], graph["nodes"]["events"]
    edge = graph["edges"]["user_events"]
    assert (edge["source_type"], edge["destination_type"]) == ("users", "events")
    pairs = [tuple(pair) for pair in edge["index"].t().tolist()]
    assert len(pairs) == len(set(pairs))
    actual: dict[str, list[tuple[int, int]]] = {request.sample_id: [] for request in requests}
    for source, destination in pairs:
        owner = users["batch"][source].item()
        assert events["batch"][destination].item() == owner
        actual[requests[owner].sample_id].append(
            (users["ids"][source].item(), events["ids"][destination].item())
        )
    assert {key: sorted(value) for key, value in actual.items()} == expected
    owners = Counter(users["batch"].tolist())
    assert owners == Counter({0: 1, 1: 1, 2: 1})
    assert len(set(zip(events["batch"].tolist(), events["ids"].tolist(), strict=True))) == len(
        events["ids"]
    )


@pytest.mark.parametrize(
    "path,fanout,degrees",
    [
        ("one_hop", 2, (192,) * 6),
        ("many_to_one", 32, (384, 3, 3, 3, 3, 3)),
        ("three_hop", 128, (192,) * 6),
    ],
)
def test_recent_graph_expansion_at_larger_degrees(path, fanout, degrees):
    pytest.importorskip("diskgraph")
    dimensions = {
        "schema_shape.graph_path": path,
        "schema_shape.graph_orientation": "aligned",
        "contract.graph_direction": "forward",
        "contract.graph_fanout": fanout,
        "time.graph_ties": "distinct",
        "time.cutoffs": "shared",
    }
    case = generate_graph_case(11, 0, dimensions, "XS")
    cutoff = case.requests[0].as_of_time
    events = [
        {
            "event_id": user * 1000 + position,
            "user_id": user,
            "shop_id": position % 6 + 1,
            "at": cutoff - 6 * position - (user - 1),
        }
        for user, degree in enumerate(degrees, 1)
        for position in range(-1, degree)  # One future event per user.
    ]
    case = replace(case, tables={**case.tables, "events": events[::-1]})
    graph_structure(case, "diskgraph", 0)
    graph_edge_uniqueness(case, "diskgraph", 0)
