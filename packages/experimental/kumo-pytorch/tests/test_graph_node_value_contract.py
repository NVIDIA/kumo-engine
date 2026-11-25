# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from kumo.pytorch import api
from kumo.pytorch.contract.sample import graph_node_dtype
from tests.test_scored_walks import star, walk_contract


def degree(stop=0):
    return api.Aggregate(api.Rows(("pin_links",), time=api.TimeWindow("at", stop=stop)), "count")


def contract(values, *, field="degree", table="pins", candidate=False):
    schema, _ = star()
    base = walk_contract(schema)
    graph = replace(base.sample.neighborhoods["graph"], node_values={table: values})
    scope = {"graph": graph}
    sample = replace(base.sample, neighborhoods=scope)
    if candidate:
        # Deliberately give the same-named query neighborhood no computed fields.
        sample = replace(
            base.sample,
            candidate_sets={
                "positive": api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods=scope)
            },
        )
    batch = replace(
        base.batch,
        inputs={
            "graph": api.Input(
                api.NeighborhoodSource("graph", candidates="positive" if candidate else None),
                api.Graph({table: api.Fields((api.Encode(field, api.Raw()),))}),
                None,
            )
        },
    )
    return replace(base, sample=sample, batch=batch)


@pytest.mark.parametrize("candidate", [False, True])
def test_computed_field_resolution_starts_at_its_node_table(candidate):
    values = {
        "degree": api.Aggregate(
            api.Rows(("board_links",), time=api.TimeWindow("at", stop=0)), "count"
        ),
        "log_degree": api.Derived("log1p", ("degree",)),
    }
    spec = contract(values, table="boards", field="log_degree", candidate=candidate)
    resolved = spec.resolve()
    resolved.validate()
    scope = resolved.sample.candidate_sets["positive"] if candidate else resolved.sample
    hood = scope.neighborhoods["graph"]
    assert hood.node_values["boards"]["degree"].dtype == api.DType("int64")
    assert graph_node_dtype(spec.schema, hood, "boards", "log_degree") == api.DType("float64")
    assert graph_node_dtype(spec.schema, hood, "boards", "id") == api.DType("int64")
    assert api.DataLoaderSpec.from_json(resolved.to_json()) == resolved


def test_node_value_declaration_identity_and_legacy_serialization():
    values = {"degree": degree()}
    spec = contract(values)
    saved = spec.fingerprint
    values.clear()
    assert spec.fingerprint == saved
    assert "degree" in spec.sample.neighborhoods["graph"].node_values["pins"]
    assert contract({"degree": degree(stop=-1)}).fingerprint != saved
    legacy = walk_contract(star()[0])
    assert '"node_values"' not in legacy.to_json()
    assert api.DataLoaderSpec.from_json(legacy.to_json()).fingerprint == legacy.fingerprint


@pytest.mark.parametrize(
    ("table", "values", "error"),
    [
        ("pins", {"degree": degree(stop=1)}, "at or before the sample time"),
        ("pins", {"id": degree()}, "conflicts with a stored column"),
        ("membership", {"value": api.Field("pin")}, "not reachable"),
        ("unknown", {"value": api.Field("pin")}, "unknown"),
        ("pins", {"degree": api.Derived("log1p", ("missing",))}, "missing"),
        ("pins", {"degree": api.Field("id", selection="all")}, "one value per node"),
    ],
)
def test_invalid_node_values_are_reported(table, values, error):
    with pytest.raises(api.ValidationError, match=error):
        contract(values, table=table).validate()


def test_neighborhoods_have_independent_value_scopes():
    spec = contract({"degree": degree()}, candidate=True)
    value = replace(spec.batch.inputs["graph"], source=api.NeighborhoodSource("graph"))
    with pytest.raises(api.ValidationError, match="degree"):
        replace(spec, batch=replace(spec.batch, inputs={"graph": value})).validate()


def test_node_values_are_diskgraph_only_and_use_diskgraph_preflight():
    stored = api.Field("id", path=("pin_links",))
    spec = contract({"degree": stored}).resolve()
    assert any(
        "computed graph node values require DiskGraphEngine" in issue.message
        for issue in api.InMemoryEngine().check_contract(spec)
    )
    with pytest.raises(ValueError, match="computed graph node values require DiskGraphEngine"):
        api.ServingRowsSpec.from_contract(spec)
    issues = api.DiskGraphEngine(num_threads=1).check_contract(spec)
    assert [(i.path, i.code) for i in issues] == [
        ("sample.neighborhoods.graph.node_values.pins.degree", "DG-ENTITY-VALUE-KIND")
    ]
    via_timed = contract(
        {"degree": api.Aggregate(api.Rows(("board_links", "pin_links")), "count")},
        table="boards",
        candidate=True,
    ).resolve()
    scope = "sample.candidate_sets.positive.neighborhoods.graph.node_values.boards.degree"
    assert any(
        i.code == "DG-TIMED-INTERMEDIATE" and i.path == f"{scope}.source.path[0]"
        for i in api.DiskGraphEngine(num_threads=1).check_contract(via_timed)
    )
