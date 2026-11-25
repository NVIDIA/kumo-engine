# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit graph node encoders at the owning cutoff, including on candidate graphs."""

from dataclasses import replace
from statistics import mean, pstdev
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import table_column_encoders
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


def test_point_in_time_graph_encoder_fits_per_occurrence_and_reloads(tmp_path):
    schema, rows = data()
    schema = replace(
        schema,
        tables=tuple(
            replace(table, columns={**table.columns, "strength": api.DType("int64")})
            if table.name == "pins"
            else table
            for table in schema.tables
        ),
    )
    strengths = {QUERY: 3, POSITIVE: 7, ISOLATED: 10, LATER: 11}
    for pin in rows["pins"]:
        pin["strength"] = strengths[pin["id"]]
    requests = [
        api.SampleRequest("later", QUERY, 20, split="valid"),
        api.SampleRequest("early", QUERY, 10, split="train"),
        api.SampleRequest("empty", QUERY, 0, split="train"),
    ]
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pq.read_table(path).append_column("split", pa.array([r.split for r in requests])), path
    )
    anchors = replace(anchors, split="split")

    spec = declared(schema, {"positive": future()})
    values = {
        "degree": api.Aggregate(api.Rows(("pin_links",), api.TimeWindow("at", stop=0)), "count")
    }
    neighborhood = replace(spec.sample.neighborhoods["graph"], node_values={"pins": values})
    sample = replace(
        spec.sample,
        neighborhoods={"graph": neighborhood},
        candidate_sets={
            "positive": replace(
                spec.sample.candidate_sets["positive"], neighborhoods={"graph": neighborhood}
            )
        },
    )
    shared = api.UseEncoding("shared")
    inputs = {
        "graph": replace(
            spec.batch.inputs["graph"],
            representation=api.Graph(
                {
                    "pins": api.Fields(
                        (api.Encode("degree", api.ZScore()), api.Encode("strength", shared))
                    )
                }
            ),
        ),
        "positive": replace(
            spec.batch.inputs["positive"],
            representation=api.Graph({"pins": api.Fields((api.Encode("degree", shared),))}),
        ),
    }
    spec = replace(
        spec,
        sample=sample,
        batch=replace(spec.batch, inputs=inputs, encodings={"shared": api.ZScore()}),
    )
    # Sharing with a physical field still fits on occurrences from the training split.
    assert not table_column_encoders(spec.resolve())

    # The query graphs at t=0/t=10 visit Q / (Q,P). The only candidate
    # graph in the training split is rooted at P at t=10 and visits (P,Q).
    trained_degrees = [0, 1, 2]
    shared_values = [strengths[QUERY], strengths[QUERY], strengths[POSITIVE], 2, 1]
    cutoffs = {r.sample_id: r.as_of_time for r in requests}

    def verify(batches):
        found = set()
        for batch in batches:
            samples = batch.metadata["sample_ids"]
            found.update(samples)
            for head in ("graph", "positive"):
                graph = batch.inputs[head]
                owners = samples
                if head == "positive":
                    owners = [
                        sid
                        for sid, ids in zip(
                            samples, batch.metadata["candidate_entity_ids"][head], strict=True
                        )
                        for _ in ids
                    ]
                pins = graph["nodes"]["pins"]
                degrees = [
                    sum(
                        row["pin"] == pin and row["at"] <= cutoffs[owners[owner]]
                        for row in rows["membership"]
                    )
                    for pin, owner in zip(pins["ids"].tolist(), pins["batch"].tolist(), strict=True)
                ]
                fit = trained_degrees if head == "graph" else shared_values
                assert pins["features"]["degree"]["values"].tolist() == pytest.approx(
                    [(v - mean(fit)) / pstdev(fit) for v in degrees]
                )
                if head == "graph":
                    assert pins["features"]["strength"]["values"].tolist() == pytest.approx(
                        [
                            (strengths[v] - mean(shared_values)) / pstdev(shared_values)
                            for v in pins["ids"].tolist()
                        ]
                    )
        assert found == set(cutoffs)

    with api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan:
        with plan.materialize(anchors, fit_split="train") as task:
            verify(task.batches(output_format="model"))
            task.to_artifact().save(tmp_path / "artifact")
        saved = api.Artifact.load(tmp_path / "artifact")
        assert saved.contract == spec.resolve()
        with (
            patch.object(
                plan._source, "fit_graph_fields", side_effect=AssertionError("unexpected refit")
            ),
            plan.materialize(anchors, fitted=saved.fitted_state) as restored,
        ):
            verify(restored.request_batches(requests, output_format="model"))
