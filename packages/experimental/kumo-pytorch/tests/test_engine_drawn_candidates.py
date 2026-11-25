# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end checks for ranked neighbors and window-selected training positives."""

from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.test_scored_walks import star, walk_contract, write_parquet

QUERY, POSITIVE, LATER, ISOLATED = 1001, 1022, 1059, 1099


def data():
    schema, rows = star()
    i64 = api.DType("int64")
    timestamp = api.DType("timestamp", unit="s", timezone="UTC")
    schema = replace(
        schema,
        tables=(
            *schema.tables,
            api.TableSpec(
                "actions",
                "id",
                {
                    "id": i64,
                    "actor": i64,
                    "pin": i64,
                    "at": timestamp,
                    "weight": i64,
                },
                time_column="at",
            ),
        ),
        relations=(*schema.relations, api.RelationSpec("acted", "pins", "id", "actions", "actor")),
    )
    rows["pins"] = [{"id": pin} for pin in (QUERY, POSITIVE, LATER, ISOLATED)]
    rows["membership"] = [
        {"id": idx, "pin": pin, "board": 10, "at": at}
        for idx, (pin, at) in enumerate(
            ((QUERY, 10), (POSITIVE, 10), (POSITIVE, 10), (LATER, 11), (POSITIVE, 11)), 1
        )
    ]
    rows["actions"] = [
        {"id": idx, "actor": QUERY, "pin": pin, "at": at, "weight": weight}
        for idx, (at, pin, weight) in enumerate(
            (
                (10, ISOLATED, 1),
                (11, POSITIVE, 1),
                (12, POSITIVE, 2),
                (13, ISOLATED, 1),
                (20, ISOLATED, 1),
                (22, POSITIVE, 1),
            ),
            1,
        )
    ]
    return schema, rows


def future(**kwargs):
    return api.DrawRows(
        api.Rows(path=("acted",), time=api.TimeWindow("at", start=0, stop=2)),
        column="pin",
        **kwargs,
    )


def declared(schema, draws, *, emit_query=True):
    base = walk_contract(schema)
    hoods = base.sample.neighborhoods
    sets = {
        name: api.CandidateSetSpec(api.EntitySpec("pins"), neighborhoods=hoods, draw=draw)
        for name, draw in draws.items()
    }
    inputs = dict(base.batch.inputs) if emit_query else {}
    for name in sets:
        inputs[name] = api.Input(
            api.NeighborhoodSource("graph", candidates=name), api.Graph(), None
        )
    return replace(
        base,
        sample=replace(base.sample, candidate_sets=sets),
        batch=replace(base.batch, inputs=inputs),
    )


def as_list(value):
    return value.to_pylist() if isinstance(value, (pa.Array, pa.ChunkedArray)) else value.tolist()


def verify_cutoffs(batch):
    ids = batch.metadata["sample_ids"]
    for name in ("neighbors", "future"):
        graph = batch.inputs[name]
        groups = batch.metadata["candidate_entity_ids"][name]
        assert groups == tuple(() if s == "empty" else (POSITIVE,) for s in ids)
        if isinstance(graph, api.ColumnarGraph):
            pins = graph.nodes["pins"]
            node_ids, owners, boundaries = pins.ids, pins.owners, graph.candidate_offsets
            edge = graph.edges["walk.pins"].index
            src, dst = (as_list(col) for col in edge.columns)
        else:
            pins = graph["nodes"]["pins"]
            node_ids, owners, boundaries = pins["ids"], pins["batch"], graph["candidate_offsets"]
            src, dst = graph["edges"]["walk.pins"]["index"].tolist()
        node_ids, owners, offsets = map(as_list, (node_ids, owners, boundaries))
        assert offsets == [sum(map(len, groups[:n])) for n in range(len(groups) + 1)]
        found = [set() for _ in range(offsets[-1])]
        for i, j in zip(src, dst, strict=True):
            assert node_ids[i] == POSITIVE and owners[i] == owners[j]
            found[owners[i]].add(node_ids[j])
        for n, s in enumerate(ids):
            if s == "early":
                assert found[offsets[n]] == {QUERY}
            if s == "late":
                assert found[offsets[n]] == {QUERY, LATER}


@pytest.mark.parametrize("emit_query", [True, False])
def test_scored_and_future_draws_expand_at_each_parent_cutoff(tmp_path, emit_query):
    schema, rows = data()
    requests = (
        api.SampleRequest("early", QUERY, 10),
        api.SampleRequest("empty", ISOLATED, 10),
        api.SampleRequest("late", QUERY, 20),
    )
    spec = declared(
        schema,
        {
            "neighbors": api.DrawNeighbors("graph", ("walk.pins",), weight="score", ranks=(0, 1)),
            "future": future(count=3),
        },
        emit_query=emit_query,
    )
    ds, anchors = write_parquet(schema, rows, requests, tmp_path)
    with (
        api.prepare(spec, ds, api.DiskGraphEngine(num_threads=2)) as plan,
        plan.materialize(anchors) as task,
    ):
        verify_cutoffs(next(task.request_batches(requests)))
        verify_cutoffs(next(task.request_batches(list(reversed(requests)), output_format="model")))
        with patch.object(
            task._anchors, "_requests", side_effect=AssertionError("native anchor fallback")
        ):
            for fmt in ("model", "columnar", "inspectable"):
                verify_cutoffs(next(task.batches(output_format=fmt)))
                for batch in task.batches(batch_size=1, shuffle=True, seed=31, output_format=fmt):
                    verify_cutoffs(batch)


def test_future_draw_preserves_original_nanoseconds(tmp_path):
    schema, rows = data()
    nanos = api.DType("timestamp", unit="ns", timezone="UTC")
    schema = replace(
        schema,
        tables=tuple(
            replace(t, columns={**t.columns, "at": nanos}) if "at" in t.columns else t
            for t in schema.tables
        ),
    )
    # Pin LATER becomes visible between the cutoffs, before the selected action.
    for table in ("membership", "actions"):
        for row in rows[table]:
            row["at"] *= 1_000_000_000
    for row in rows["membership"]:
        if row["at"] == 11_000_000_000:
            row["at"] = 10_000_000_050
    rows["actions"] = [dict(id=1, actor=QUERY, pin=POSITIVE, at=10_000_000_090, weight=1)]
    reqs = (api.SampleRequest("early", QUERY, 10), api.SampleRequest("late", QUERY, 10))
    ds, anchors = write_parquet(schema, rows, reqs, tmp_path)
    cutoffs = pa.array([10_000_000_025, 10_000_000_075], type=nanos.to_arrow())
    path = tmp_path / "anchors.parquet"
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.schema.get_field_index("at"), "at", cutoffs), path)
    reqs = tuple(replace(r, as_of_time=at) for r, at in zip(reqs, cutoffs, strict=True))
    spec = declared(
        schema,
        {"neighbors": api.DrawNeighbors("graph", ("walk.pins",), ranks=(0, 1)), "future": future()},
    )
    with (
        api.prepare(spec, ds, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        verify_cutoffs(next(task.request_batches(reqs)))
        verify_cutoffs(next(task.batches(output_format="columnar")))


def test_weighted_future_draws_are_distinct_seeded_and_batch_independent(tmp_path):
    schema, rows = data()
    rows["actions"] = [
        dict(id=i, actor=QUERY, at=11, pin=pin, weight=w)
        for i, (pin, w) in enumerate(((POSITIVE, 1), (POSITIVE, 2), (LATER, 9)), 1)
    ]
    requests = tuple(api.SampleRequest(f"s{i}", QUERY, 10) for i in range(512))
    ds, anchors = write_parquet(schema, rows, requests, tmp_path)
    draws = {
        "uniform": future(),
        "weighted": future(weight="weight"),
        "all": future(count=4, weight="weight"),
    }
    sets = {
        name: api.CandidateSetSpec(
            api.EntitySpec("pins"), values={"pin": api.Field(column="id")}, draw=draw
        )
        for name, draw in draws.items()
    }
    spec = api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec("pins"), candidate_sets=sets),
        api.BatchSpec(
            inputs={
                name: api.Input(
                    api.CandidateSource(name),
                    api.Fields((api.Encode("pin", api.Raw()),)),
                    api.Packed(),
                )
                for name in draws
            },
            batching=api.Batching(max_samples=64),
        ),
    )

    def collect(batches):
        result = {}
        for batch in batches:
            groups = batch.metadata["candidate_entity_ids"]
            for name in draws:
                values = batch.inputs[name]["pin"]["values"].tolist()
                assert values == [pin for group in groups[name] for pin in group]
            for i, sid in enumerate(batch.metadata["sample_ids"]):
                result[sid] = tuple(groups[name][i] for name in draws)
        return result

    outputs = []
    for engine_seed in (13, 29):
        with (
            api.prepare(
                spec, ds, api.DiskGraphEngine(num_threads=1, random_seed=engine_seed)
            ) as plan,
            plan.materialize(anchors) as task,
        ):
            selected = collect(task.batches(output_format="model"))
            outputs.append(selected)
            if engine_seed == 13:
                assert selected == collect(
                    task.batches(batch_size=17, shuffle=True, seed=200, output_format="model")
                )
                shards = [
                    collect(task.batches(rank=rank, world_size=2, output_format="model"))
                    for rank in (0, 1)
                ]
                assert not (shards[0].keys() & shards[1].keys())
                assert (shards[0] | shards[1]) == selected
                iterator = task.batches(output_format="model")
                prefix = collect([next(iterator)])
                resumed = task.batches(output_format="model")
                resumed.load_state_dict(iterator.state_dict())
                assert (prefix | collect(resumed)) == selected
                assert selected == collect(
                    task.request_batches(tuple(reversed(requests)), output_format="model")
                )
        for uniform, weighted, all_ids in selected.values():
            assert len(uniform) == len(weighted) == 1
            assert len(all_ids) == 2 and set(all_ids) == {POSITIVE, LATER}
        assert sum(item[0] == (LATER,) for item in selected.values()) / len(
            requests
        ) == pytest.approx(0.5, abs=0.07)
        assert sum(item[1] == (LATER,) for item in selected.values()) / len(
            requests
        ) == pytest.approx(0.75, abs=0.06)
    assert outputs[0] != outputs[1]


def test_future_draw_fits_candidate_graph_encoders_and_reads_history_at_anchor(tmp_path):
    schema, rows = data()
    requests = (api.SampleRequest("early", QUERY, 10), api.SampleRequest("late", QUERY, 20))
    ds, anchors = write_parquet(schema, rows, requests, tmp_path)
    spec = declared(schema, {"future": future()}, emit_query=False)
    candidate = replace(
        spec.sample.candidate_sets["future"],
        values={
            "id": api.Field("id"),
            "history": api.Aggregate(api.Rows(("pin_links",), api.TimeWindow("at")), "count"),
        },
    )
    spec = replace(
        spec,
        sample=replace(
            spec.sample, candidate_sets={"future": candidate}, entity_values={"id": api.Field("id")}
        ),
        batch=replace(
            spec.batch,
            encodings={"shared": api.ZScore()},
            inputs={
                "owner": api.Input(
                    api.EntitySource(), api.Fields((api.Encode("id", api.Raw()),)), api.Separate()
                ),
                "future": api.Input(
                    api.NeighborhoodSource("graph", candidates="future"),
                    api.Graph({"pins": api.Fields((api.Encode("id", api.UseEncoding("shared")),))}),
                    None,
                ),
                "history": api.Input(
                    api.CandidateSource("future"),
                    api.Fields(
                        (
                            api.Encode("history", api.Raw()),
                            api.Encode("id", api.UseEncoding("shared")),
                        )
                    ),
                    api.Packed(),
                ),
            },
        ),
    )

    def check(batch):
        assert batch.inputs["owner"]["id"]["values"].tolist() == [QUERY, QUERY]
        assert batch.metadata["candidate_entity_ids"]["future"] == ((POSITIVE,), (POSITIVE,))
        assert batch.inputs["history"]["history"]["values"].tolist() == [2, 3]
        pins = batch.inputs["future"]["nodes"]["pins"]
        # Only the early anchor was used to fit. Its drawn pin can see QUERY,
        # while LATER first appears at the next tick.
        # The fitted shared encoder sees POSITIVE as both a graph node and a scalar.
        mid = (2 * POSITIVE + QUERY) / 3
        spread = ((2 * (POSITIVE - mid) ** 2 + (QUERY - mid) ** 2) / 3) ** 0.5
        assert pins["features"]["id"]["values"].tolist() == pytest.approx(
            [(key - mid) / spread for key in pins["ids"].tolist()]
        )

    engine = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, ds, engine) as plan:
        with plan.materialize(requests[:1]) as task:
            check(next(task.request_batches(requests)))
            task.to_artifact().save(tmp_path / "artifact")
        artifact = api.Artifact.load(tmp_path / "artifact")
        assert artifact.contract == spec.resolve()
        with plan.materialize(anchors, fitted=artifact.fitted_state) as recovered:
            check(next(recovered.batches(output_format="model")))
            check(next(recovered.request_batches(requests, output_format="model")))


@pytest.mark.parametrize("fmt", ("inspectable", "model"))
def test_direct_draw_reuses_sampled_query_graph_within_each_batch(tmp_path, fmt):
    schema, rows = data()
    requests = tuple(api.SampleRequest(f"q{i}", QUERY, 10 + i) for i in range(4))
    ds, anchors = write_parquet(schema, rows, requests, tmp_path)
    spec = declared(schema, {"neighbors": api.DrawNeighbors("graph", ("walk.pins",))})
    with (
        api.prepare(spec, ds, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
        patch.object(plan._source, "_root_ids", wraps=plan._source._root_ids) as lookup,
    ):
        for batch in task.request_batches(requests, batch_size=2, output_format=fmt):
            assert len(batch.metadata["sample_ids"]) == 2
        # One root graph lookup and one candidate graph lookup per batch.
        # Fetching the emitted query graph again would add two more.
        assert lookup.call_count == 4


def test_short_draws_and_supplied_sets_share_a_padded_batch(tmp_path):
    schema, rows = data()
    draw = api.CandidateSetSpec(
        api.EntitySpec("pins"), values={"id": api.Field("id")}, draw=future(count=4)
    )
    sets = {"positive": draw, "provided": replace(draw, draw=None)}
    sample = api.SampleSpec(api.EntitySpec("pins"), candidate_sets=sets)
    batch_spec = api.BatchSpec(
        inputs={
            name: api.Input(
                api.CandidateSource(name), api.Fields((api.Encode("id", api.Raw()),)), api.Padded()
            )
            for name in sets
        },
        batching=api.Batching(max_samples=2),
    )
    requests = (
        api.SampleRequest("a", QUERY, 10, candidates={"provided": (api.CandidateRequest(LATER),)}),
        api.SampleRequest("b", ISOLATED, 10),
    )
    ds, _ = write_parquet(schema, rows, requests, tmp_path)
    with (
        api.prepare(
            api.DataLoaderSpec(schema, sample, batch_spec), ds, api.DiskGraphEngine()
        ) as plan,
        plan.materialize(requests) as task,
    ):
        for fmt in ("inspectable", "model"):
            batch = next(task.request_batches(requests, output_format=fmt))
            assert batch.metadata["candidate_entity_ids"] == {
                "positive": ((POSITIVE,), ()),
                "provided": ((LATER,), ()),
            }
            for name in sets:
                assert batch.inputs[name]["valid"].tolist() == [[True], [False]]
            invalid = replace(requests[0], candidates={"positive": ()})
            with pytest.raises(ValueError, match="engine-drawn candidate"):
                next(task.request_batches([invalid], output_format=fmt))


def test_hidden_draw_graph_shares_native_schedule_with_a_different_emitted_graph(tmp_path):
    schema, rows = data()
    request = api.SampleRequest("early", QUERY, 10)
    ds, anchors = write_parquet(schema, rows, (request,), tmp_path)
    spec = declared(schema, {"neighbors": api.DrawNeighbors("graph", ("walk.pins",))})
    spec = replace(
        spec,
        sample=replace(
            spec.sample,
            neighborhoods={
                **spec.sample.neighborhoods,
                "history": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("pin_links", 4, strategy="recent"),)),)
                ),
            },
        ),
        batch=replace(
            spec.batch,
            inputs={
                **spec.batch.inputs,
                "graph": api.Input(api.NeighborhoodSource("history"), api.Graph(), None),
            },
        ),
    )
    with (
        api.prepare(spec, ds, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        for fmt in ("columnar", "model", "inspectable"):
            with patch.object(plan._source, "_root_ids", wraps=plan._source._root_ids) as lookup:
                batch = next(task.batches(output_format=fmt))
                # Only candidate expansion uses the additional sampler.
                assert lookup.call_count == 1
            assert batch.metadata["candidate_entity_ids"]["neighbors"] == ((POSITIVE,),)
            graph = batch.inputs["graph"]
            edges = graph.edges if isinstance(graph, api.ColumnarGraph) else graph["edges"]
            assert set(edges) == {"pin_links"}
            assert batch.metadata["walk_pushes"] > 0
