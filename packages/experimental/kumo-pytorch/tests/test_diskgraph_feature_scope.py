# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Walks, candidate graphs and shaped features require the DiskGraph sampler."""

from dataclasses import replace

import pytest

from kumo.pytorch import api
from tests.test_scored_walks import star, walk_contract, write_parquet


@pytest.mark.parametrize(
    "feature", ["scored random-walk sampling", "candidate neighborhoods", "array feature columns"]
)
def test_in_memory_rejects_diskgraph_features_at_preflight(feature, tmp_path):
    schema, rows = star()
    declaration = walk_contract(schema)
    if feature != "scored random-walk sampling":
        hops = api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("pin_links", 2),)),))
        declaration = replace(
            declaration, sample=replace(declaration.sample, neighborhoods={"graph": hops})
        )
        if feature == "candidate neighborhoods":
            candidates = api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods={"links": hops})
            declaration = replace(
                declaration,
                sample=replace(declaration.sample, candidate_sets={"positive": candidates}),
            )
        else:
            pin_table, *others = schema.tables
            pin_table = replace(
                pin_table,
                columns={**pin_table.columns, "unused_embedding": api.DType("float32", shape=(2,))},
            )
            declaration = replace(declaration, schema=replace(schema, tables=(pin_table, *others)))
    tables = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
    message = f"{feature} require"
    assert any(
        message in issue.message for issue in api.check(declaration, tables, api.InMemoryEngine())
    )
    with pytest.raises(ValueError, match=message):
        api.prepare(declaration, tables, api.InMemoryEngine())
    if feature == "array feature columns":
        # The unused array column is declared on the schema rather than the batch.
        with pytest.raises(ValueError, match=message):
            api.ServingRowsSpec.from_contract(declaration)
        return
    requests = [api.SampleRequest("a", 1, 20)]
    dataset, anchors = write_parquet(declaration.schema, rows, requests, tmp_path)
    with (
        api.prepare(declaration, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        artifact = task.to_artifact()
    with pytest.raises(ValueError, match=message):
        api.ServingAssembler(artifact)
