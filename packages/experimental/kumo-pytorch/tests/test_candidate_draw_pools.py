# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiskGraph's batched candidate pools and selection edge cases."""

from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

import pyarrow as pa
import pytest

from kumo.pytorch import api
from kumo.pytorch.engine import GraphRoot
from kumo.pytorch.engines.candidate_draws import _eligible_entities, _select, draw_candidate_ids
from tests.test_scored_walks import star, walk_contract, write_parquet


def test_reuses_query_graph_for_multiple_edges_and_resolves_reverse_destinations(tmp_path):
    schema, rows = star()
    # Native node positions and graph IDs must never escape as candidate keys.
    for row in rows["pins"]:
        row["id"] += 100
    for row in rows["membership"]:
        row["pin"] += 100
    base = walk_contract(schema, after_hops=True)
    sets = {
        name: api.CandidateSetSpec(api.EntitySpec("pins"), draw=draw)
        for name, draw in {
            "both": api.DrawNeighbors("graph", ("pin_links", "walk.pins"), count=10),
            "second": api.DrawNeighbors("graph", ("walk.pins",), count=10, ranks=(1, 2)),
            "weighted": api.DrawNeighbors("graph", ("walk.pins",), count=10, weight="rank"),
        }.items()
    }
    spec = replace(base, sample=replace(base.sample, candidate_sets=sets))
    requests = (api.SampleRequest("early", 10, 10), api.SampleRequest("late", 10, 20))
    ds, _ = write_parquet(schema, rows, requests, tmp_path)
    with api.prepare(spec, ds, api.DiskGraphEngine(num_threads=1)) as plan:
        source = plan._source
        roots = [
            GraphRoot(r.sample_id, r.entity_id, source._cutoff(r.as_of_time)) for r in requests
        ]
        with (
            patch.object(
                source, "_sample_graph_events", wraps=source._sample_graph_events
            ) as graph,
            patch.object(source, "_feature_values", wraps=source._feature_values) as keys,
        ):
            result = draw_candidate_ids(source, spec.resolve().sample, roots)
        graph.assert_called_once()
        keys.assert_called_once()
        assert [set(ids) for ids in result["both"]] == [{101, 102}, {101, 102, 103}]
        assert result["second"][0] == result["weighted"][0] == ()
        assert set(result["second"][1]) == set(result["weighted"][1])
        assert result["second"][1]
        assert all(len(ids) == len(set(ids)) for groups in result.values() for ids in groups)


def test_nullable_and_extreme_weights_and_nanosecond_ids():
    values = pa.array([11, 11, 12, 13, None, 14], type=pa.timestamp("ns"))
    roots = [GraphRoot("anchor", 9, 1), GraphRoot("empty", 9, 0)]
    pool = pa.table(
        {
            "owner": [0] * len(values),
            "id": values,
            "weight": [1.7e308, 1.7e308, 5e-324, 0, 1, None],
        }
    )
    draw = api.DrawRows(api.Rows(), "id", count=10, weight="w")
    (selected, empty) = _select(pool, roots, draw, "heads", 99)
    assert [v.value for v in selected] == [11, 12]
    assert empty == ()
    decimal = pa.array([Decimal("1.25"), Decimal("0.00")])
    pool = pa.table({"owner": [0, 1], "id": ["public-a", "public-b"], "weight": decimal})
    assert _select(pool, roots, draw, "heads", 99) == [("public-a",), ()]


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), float("-inf")])
def test_invalid_weights_identify_the_offending_draw(invalid):
    draw = api.DrawRows(api.Rows(), "id", count=2, weight="visits")
    roots = [GraphRoot("anchor", 9, 1)]
    pool = pa.table({"owner": [0, 0], "id": [1, 2], "weight": [1.0, invalid]})
    with pytest.raises(ValueError, match="candidate set 'positives'.*finite nonnegative.*visits"):
        _select(pool, roots, draw, "positives", 0)


def test_multiple_row_draws_use_one_native_frame_and_obey_both_window_boundaries(tmp_path):
    schema, rows = star()
    base = walk_contract(schema)
    window = api.Rows(("pin_links",), time=api.TimeWindow("at", start=1, stop=10))
    sets = {
        name: api.CandidateSetSpec(
            api.EntitySpec("boards"), draw=api.DrawRows(window, "board", count=3, weight=weight)
        )
        for name, weight in (("unweighted", None), ("weighted", "id"))
    }
    spec = replace(base, sample=replace(base.sample, candidate_sets=sets))
    requests = (api.SampleRequest("upper", 3, 10), api.SampleRequest("lower", 3, 19))
    ds, _ = write_parquet(schema, rows, requests, tmp_path)
    with api.prepare(spec, ds, api.DiskGraphEngine(num_threads=1)) as plan:
        source = plan._source
        roots = [
            GraphRoot(r.sample_id, r.entity_id, source._cutoff(r.as_of_time)) for r in requests
        ]
        with patch.object(source, "_query_values", wraps=source._query_values) as query:
            result = draw_candidate_ids(source, spec.resolve().sample, roots)
        query.assert_called_once()
        program = query.call_args.args[3]
        assert len(program["frames"]) == len(program["sequences"]) == 1
        assert result == {"weighted": [(10,), ()], "unweighted": [(10,), ()]}


@pytest.mark.parametrize("native_ids", [False, True])
def test_candidate_entity_clock_and_availability_before_sampling(tmp_path, native_ids):
    schema, rows = star()
    ns = api.DType("timestamp", unit="ns", timezone="UTC")
    pins = next(t for t in schema.tables if t.name == "pins")
    pins = replace(
        pins, columns={**pins.columns, "created": ns, "published": ns}, time_column="created"
    )
    schema = replace(schema, tables=tuple(pins if t.name == "pins" else t for t in schema.tables))
    for row, (created, published) in zip(
        rows["pins"], [(100, 99), (101, 99), (99, 100), (99, None)], strict=True
    ):
        row.update(created=created, published=published)
    base = walk_contract(schema, after_hops=True)
    ds, _ = write_parquet(schema, rows, [api.SampleRequest("setup", 10, 0)], tmp_path)
    entity = api.EntitySpec("pins", available_from="published")
    with api.prepare(base, ds, api.DiskGraphEngine(num_threads=1)) as plan:
        source = plan._source
        roots = [GraphRoot("equal", 10, 100), GraphRoot("later", 10, 101)]
        assert source._clock == "nanoseconds"
        keys = [1, 1, 2, 3, 4, None]
        if native_ids:
            dense = source._sampler.lookup_node_ids("pins", pa.array([1, 2, 3, 4]))
            keys = [None if key is None else dense[key - 1] for key in keys]
        pool = pa.table(
            {"owner": [0] * 6 + [1] * 6, "id": keys * 2, "weight": [1, 1, 1e15, 2, 1, 1] * 2}
        )
        with patch.object(source, "_feature_arrays", wraps=source._feature_arrays) as fetch:
            visible = _eligible_entities(source, pool, entity, roots, native_ids=native_ids)
        fetch.assert_called_once()
        assert len(fetch.call_args.args[1]) == 4
        assert fetch.call_args.args[2] == ("created", "published")
        assert visible["id"].to_pylist() == [keys[0], keys[1], *keys[:4]]
        result = _select(visible, roots, api.DrawRows(api.Rows(), "id", weight="weight"), "p", 1)
        assert result == [(keys[0],), (keys[2],)]
        # The same field can additionally impose the stricter available_from rule.
        same_clock = replace(entity, available_from="created")
        visible = _eligible_entities(source, pool, same_clock, roots, native_ids=native_ids)
        assert visible["id"].to_pylist() == [keys[3], keys[4], keys[0], keys[1], keys[3], keys[4]]
