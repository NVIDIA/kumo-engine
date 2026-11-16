# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G-GRAPH-001 rule 3: an edge appears at most once per sample."""

from collections import Counter

import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api

USERS = [{"user_id": 1, "at": 0}, {"user_id": 2, "at": 0}]
EVENTS = [
    {"event_id": 10, "user_id": 1, "at": 5},
    {"event_id": 11, "user_id": 1, "at": 7},
    {"event_id": 20, "user_id": 2, "at": 6},
]


def _contract() -> api.DataLoaderSpec:
    def hop(direction: str) -> api.GraphLayer:
        return api.GraphLayer(
            (api.GraphSample("user_events", 4, direction=direction, strategy="recent"),)
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
                "history": api.NeighborhoodSpec((hop("forward"), hop("reverse"), hop("forward")))
            },
        ),
        api.BatchSpec({"graph": api.Input(api.NeighborhoodSource("history"), api.Graph(), None)}),
    )


@pytest.mark.parametrize("engine_name", ["in_memory", "diskgraph"])
def test_revisited_edges_appear_once_per_sample(engine_name):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = api.DataLoaderSpec.from_json(_contract().to_json())
    # Two samples of user 1 share the stored edges; each keeps its own copy.
    requests = [
        api.SampleRequest("early", 1, 6),
        api.SampleRequest("late", 1, 100),
        api.SampleRequest("other", 2, 100),
    ]
    expected = {
        "early": [(1, 10)],
        "late": [(1, 10), (1, 11)],
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
