# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical rows, graph nodes, and future labels agree at time boundaries."""

from dataclasses import replace

import pytest

from conformance.assertions import bound_tables
from conformance.cases import GUARANTEES, load_case
from kumo.pytorch import api


@pytest.mark.parametrize(
    ("engine_name", "output_format"),
    [("in_memory", "inspectable"), ("diskgraph", "inspectable"), ("diskgraph", "model")],
)
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("clock", ["at", "other_at"])
def test_time_boundaries_across_outputs(engine_name, output_format, bounded, clock):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    case = load_case(GUARANTEES / "G-ROWS-001/golden/C-ROWS-001.json")
    schema = replace(
        case.contract.schema,
        tables=tuple(
            replace(table, columns={**table.columns, "other_at": table.columns["at"]})
            if table.name == "events"
            else table
            for table in case.contract.schema.tables
        ),
    )
    tables = {
        **case.tables,
        "events": [{**row, "other_at": row["at"]} for row in case.tables["events"]],
    }
    rows = api.Rows(
        path=("user_events",),
        time=api.TimeWindow(column=clock, start=-1 if bounded else None, stop=0),
    )
    sample = replace(
        case.contract.sample,
        events={
            "history": api.EventSpec(
                rows=rows,
                fields={
                    "event_id": api.Field(column="event_id"),
                    "age": api.EventTimeDelta(unit="seconds", dtype=api.DType(name="float64")),
                },
            )
        },
        neighborhoods={
            "history": api.NeighborhoodSpec(
                layers=(
                    api.GraphLayer(
                        samples=(
                            api.GraphSample(relation="user_events", fanout=8, strategy="recent"),
                        )
                    ),
                )
            )
        },
        targets={
            "future_count": api.Aggregate(
                source=api.Rows(
                    path=("user_events",), time=api.TimeWindow(column=clock, start=0, stop=2)
                ),
                operation="count",
                dtype=api.DType(name="int64"),
            )
        },
    )
    contract = api.DataLoaderSpec(
        schema=schema,
        sample=sample,
        batch=replace(
            case.contract.batch,
            inputs={
                **case.contract.batch.inputs,
                "history": api.Input(
                    source=api.EventSource(events=("history",)),
                    representation=api.Fields(
                        fields=(
                            api.Encode(field="event_id", encoding=api.Raw()),
                            api.Encode(field="age", encoding=api.Raw()),
                        )
                    ),
                    layout=api.Padded(),
                ),
                "graph": api.Input(
                    source=api.NeighborhoodSource(name="history"),
                    representation=api.Graph(),
                    layout=None,
                ),
            },
            targets={"future_count": api.LabelTarget(source="future_count", encoding=api.Raw())},
            # Model batches use native length counting to obey this budget.
            batching=api.Batching(max_samples=5, max_tokens=4, length_input="history"),
        ),
    )
    # Rows are at 2, 9, 10, 12 for user 1 and 11 for user 2. These lists
    # independently fix equality and future behavior for every request.
    all_ids = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4], []]
    recent_ids = [[2], [3], [], [4], []]
    expected_ids = dict(zip(case.expected, recent_ids if bounded else all_ids, strict=True))
    expected_graph = dict(zip(case.expected, all_ids, strict=True))
    expected_future = dict(zip(case.expected, [1, 1, 1, 0, 1], strict=True))
    cutoffs = {request.sample_id: request.as_of_time for request in case.requests}
    times = {row["event_id"]: row["at"] for row in case.tables["events"]}
    delivered = []
    with bound_tables(contract, tables, engine_name) as (dataset, engine):
        with api.prepare(contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                for batch in task.batches(output_format=output_format):
                    history = batch.inputs["history"]
                    graph = batch.inputs["graph"]["nodes"].get("events")
                    for index, sample_id in enumerate(batch.metadata["sample_ids"]):
                        delivered.append(sample_id)
                        valid = history["valid"][index]
                        ids = history["event_id"]["values"][index][valid].tolist()
                        assert ids == expected_ids[sample_id]
                        assert history["age"]["values"][index][valid].tolist() == [
                            cutoffs[sample_id] - times[event_id] for event_id in ids
                        ]
                        actual_graph = (
                            graph["ids"][graph["batch"] == index].tolist() if graph else []
                        )
                        assert set(actual_graph) == set(expected_graph[sample_id])
                        for name, expected in case.expected[sample_id]["entity_values"].items():
                            assert batch.inputs["entity"][name]["values"][index].item() == expected
                        assert (
                            batch.targets["future_count"]["values"][index].item()
                            == (expected_future[sample_id])
                        )
    assert delivered == list(case.expected)
