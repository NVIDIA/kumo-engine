# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from kumo.pytorch import api
from tests.test_scored_walks import star, walk_contract, write_parquet


def contract(draw, entity="pins", *, hops=False):
    schema, _ = star()
    base = walk_contract(schema, after_hops=hops)
    candidate = api.CandidateSetSpec(api.EntitySpec(entity), draw=draw)
    return replace(base, sample=replace(base.sample, candidate_sets={"positive": candidate}))


@pytest.mark.parametrize(
    "draw",
    (
        api.DrawRows(api.Rows(("pin_links",), api.TimeWindow("at", start=0, stop=5)), "pin"),
        api.DrawRows(api.Rows(("pin_links",)), "pin", weight="id"),
        api.DrawNeighbors("graph", ("walk.pins",), count=2, weight="score", ranks=(0, 2)),
        api.DrawNeighbors("graph", ("walk.pins",), weight="rank"),
    ),
)
def test_draw_spec_round_trip(draw):
    spec = contract(draw)
    spec.validate()
    assert api.DataLoaderSpec.from_json(spec.to_json()) == spec
    assert spec.fingerprint != contract(None).fingerprint
    assert '"draw"' not in contract(None).to_json()


@pytest.mark.parametrize(
    "draw,entity,error",
    (
        (api.DrawRows(api.Rows(("pin_links",)), "missing"), "pins", "unknown column"),
        (api.DrawRows(api.Rows(("pin_links",)), "at"), "pins", "candidate primary key type"),
        (api.DrawRows(api.Rows(("pin_links",)), "pin", weight="at"), "pins", "numeric scalars"),
        (
            api.DrawRows(api.Rows(("pin_links",), api.TimeWindow("at", start=-1, stop=5)), "pin"),
            "pins",
            "window must start at or after",
        ),
        (api.DrawNeighbors("other", ("walk.pins",)), "pins", "unknown query neighborhood"),
        (api.DrawNeighbors("graph", ("missing",)), "pins", "unknown sampled edge set"),
        (api.DrawNeighbors("graph", ("walk.boards",)), "pins", "must end at candidate table"),
        (api.DrawNeighbors("graph", ("walk.pins",), weight="unknown"), "pins", "has no 'unknown'"),
    ),
)
def test_draw_schema_errors(draw, entity, error):
    with pytest.raises(api.ValidationError, match=error):
        contract(draw, entity).validate()


def test_unscored_hops_supply_only_uniform_destinations():
    hop = api.DrawNeighbors("graph", ("pin_links",))
    contract(hop, hops=True).validate()  # Reverse traversal ends at pins.
    for change, message in (({"ranks": (0, 1)}, "no rank"), ({"weight": "score"}, "no 'score'")):
        with pytest.raises(api.ValidationError, match=message):
            contract(replace(hop, **change), hops=True).validate()


@pytest.mark.parametrize(
    "changes",
    (
        {"edges": ()},
        {"edges": ("g", "g")},
        {"count": 0},
        {"count": True},
        {"ranks": (-1, 2)},
        {"ranks": (2, 2)},
        {"ranks": (0, 1.5)},
    ),
)
def test_draw_rejects_malformed_declarations(changes):
    with pytest.raises(ValueError):
        replace(api.DrawNeighbors("graph", ("g",)), **changes)


def test_draw_checks_time_visibility_on_the_route_to_the_candidate():
    spec = contract(api.DrawNeighbors("graph", ("pin_links",)), hops=True)
    graph = spec.sample.neighborhoods["graph"]
    first, *rest = graph.layers
    uniform = replace(first.samples[0], strategy="uniform")
    graph = replace(graph, layers=(api.GraphLayer((uniform,)), *rest))
    with pytest.raises(api.ValidationError, match="ignores the candidate anchor"):
        replace(spec, sample=replace(spec.sample, neighborhoods={"graph": graph})).validate()


def test_draw_rows_uses_diskgraph_row_preflight():
    rows = api.Rows(("pin_links", "board_links"), filters=(api.RowFilter("id", "not_null"),))
    spec = contract(api.DrawRows(rows, "id"), entity="boards").resolve()
    issues = api.DiskGraphEngine(num_threads=1).check_contract(spec)
    assert {(i.code, i.path) for i in issues} == {
        ("DG-TIMED-INTERMEDIATE", "sample.candidate_sets.positive.draw.rows.path[0]"),
        ("DG-ROW-FILTER", "sample.candidate_sets.positive.draw.rows.filters"),
    }


def test_draw_conflicts_and_engine_scope(tmp_path):
    schema, rows = star()
    draw = api.DrawRows(api.Rows(("pin_links",), api.TimeWindow("at", 0, 1)), "pin")
    spec = contract(draw)
    dataset, anchors = write_parquet(schema, rows, [api.SampleRequest("a", 1, 10)], tmp_path)
    engine = api.DiskGraphEngine(num_threads=1)
    bound = replace(anchors, candidates={"positive": api.CandidateColumn("ids", "pin")})
    assert any(
        i.path == "anchors.candidates.positive"
        for i in api.check(spec, dataset, engine, anchors=bound)
    )
    assert any(
        "training-only" in i.message for i in api.check(spec, dataset, engine, inference=True)
    )
    with api.prepare(spec, dataset, engine) as plan:
        with pytest.raises(api.ValidationError, match="training-only"):
            plan.materialize(anchors, inference=True)
    historical = contract(api.DrawRows(api.Rows(("pin_links",)), "pin"))
    assert not api.check(historical, dataset, engine, inference=True)
    assert any(
        i.path == "sample.candidate_sets.positive.draw"
        for i in api.InMemoryEngine().check_contract(historical)
    )
    with pytest.raises(ValueError, match="engine-drawn candidates require DiskGraphEngine"):
        api.ServingRowsSpec.from_contract(historical)
