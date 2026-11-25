# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from kumo.pytorch import api
from tests.test_engine_drawn_candidates import (
    ISOLATED,
    LATER,
    POSITIVE,
    QUERY,
    data,
    declared,
    future,
)
from tests.test_scored_walks import write_parquet


@pytest.mark.parametrize("available_from,count", [(None, 3), ("released", 1)])
def test_future_actions_draw_only_entities_visible_at_the_anchor(tmp_path, available_from, count):
    schema, rows = data()
    clock = api.DType("timestamp", unit="s", timezone="UTC")
    schema = replace(
        schema,
        tables=tuple(
            replace(
                table,
                columns={**table.columns, "created": clock, "released": clock},
                time_column="created",
            )
            if table.name == "pins"
            else table
            for table in schema.tables
        ),
    )
    created = {QUERY: 1, POSITIVE: 10, ISOLATED: 8, LATER: 11}
    for row in rows["pins"]:
        key = row["id"]
        row.update(created=created[key], released=10 if key == ISOLATED else 9)
    rows["actions"] = [
        {"id": i, "actor": QUERY, "pin": key, "at": 11, "weight": 1}
        for i, key in enumerate((POSITIVE, ISOLATED, LATER))
    ]
    spec = declared(schema, {"future": future(count=count)}, emit_query=False)
    candidate = replace(
        spec.sample.candidate_sets["future"],
        entity=api.EntitySpec("pins", available_from=available_from),
        values={"id": api.Field("id")},
    )
    spec = replace(
        spec,
        sample=replace(spec.sample, candidate_sets={"future": candidate}),
        batch=replace(
            spec.batch,
            inputs={
                **spec.batch.inputs,
                "ids": api.Input(
                    api.CandidateSource("future"),
                    api.Fields((api.Encode("id", api.Raw()),)),
                    api.Packed(),
                ),
            },
        ),
    )
    # At time 10, POSITIVE is on the inclusive table-clock boundary. ISOLATED
    # is on the exclusive available_from boundary. LATER's row is still future.
    expected = {POSITIVE} if available_from else {POSITIVE, ISOLATED}
    requests = [api.SampleRequest(f"seed-{i}", QUERY, 10) for i in range(6)]
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    with (
        api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        for stream in (task.request_batches(requests), task.batches(output_format="model")):
            seen = []
            for batch in stream:
                groups = batch.metadata["candidate_entity_ids"]["future"]
                assert all(set(group) == expected for group in groups)
                assert batch.inputs["ids"]["id"]["values"].tolist() == [
                    key for group in groups for key in group
                ]
                graph = batch.inputs["future"]
                graph_ids = set(graph["nodes"]["pins"]["ids"].tolist())
                assert expected <= graph_ids and LATER not in graph_ids
                if available_from:
                    assert ISOLATED not in graph_ids
                seen.extend(batch.metadata["sample_ids"])
            assert sorted(seen) == sorted(request.sample_id for request in requests)
